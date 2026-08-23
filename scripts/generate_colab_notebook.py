"""Generate the checked-in Google Colab research notebook deterministically."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "04_colab_research_experiments.ipynb"


def markdown(source: str) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip() + "\n",
    }


def code(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip() + "\n",
    }


def build_notebook() -> dict[str, object]:
    cells: list[dict[str, object]] = [
        markdown(
            r"""
# CUDA Optimization of Fused Causal Softmax — Colab Experiments

This notebook is the NVIDIA execution companion for the research repository.
It builds and tests the one evolving CUDA source, records raw measurements and
provenance, and refuses to replace missing CUDA evidence with CPU or fabricated
numbers.

# HOW TO USE THIS NOTEBOOK

1. In Colab, choose **Runtime → Change runtime type → NVIDIA GPU**.
2. Read and edit the configuration cell once before executing experiments.
3. Run the artifact and environment sections. Stop if CUDA validation fails.
4. Clone the configured GitHub branch, or select ZIP mode and upload a checkout.
5. Install only the repository's development requirements; keep Colab's
   CUDA-enabled PyTorch installation.
6. Build the extension and require its import check to pass.
7. Run correctness before accepting any custom-kernel measurement.
8. Run historical, launch-configuration, softmax, and attention experiments.
9. Run PyTorch Profiler and let Nsight Compute fail gracefully if Colab blocks
   hardware counters.
10. Generate figures and tables only from the saved CSV files.
11. Read the validation report and artifact inventory.
12. Optionally back up to Drive, create the ZIP, and download it.
13. Send the ZIP back with the paper source so claims can be filled from evidence.

If Colab disconnects, reconnect to a GPU, rerun setup/build, restore `artifacts/`
from Drive if used, set `ARTIFACT_POLICY = "REUSE"`, and continue with the first
incomplete stage. Never combine comparison files from different GPU models or
software environments; if the assigned GPU changes, rerun the whole matched
comparison series.

Each numbered section explains what it does, why it matters, what to expect,
what to save, and how its evidence can be used in the paper.
"""
        ),
        markdown(
            r"""
# 1. Configuration

**What this section does:** Defines every user-controlled path, experiment size,
feature flag, tolerance, and overwrite rule in one place.

**Why necessary:** A reproducible experiment changes controls deliberately, not
through hidden edits across many cells.

**What to expect:** No experiment runs in this cell. Review it before continuing.

**What to save:** The final manifest records these values.

**Paper meaning:** These values define the experimental setup and validity scope.
"""
        ),
        code(
            r'''
REPO_URL = "https://github.com/Arwin-K/cuda-attention-softmax.git"
BRANCH = "main"
SOURCE_MODE = "GITHUB"  # "GITHUB" or "ZIP"

WORK_DIR = "/content/cuda-attention-softmax"
ARTIFACT_DIR = "/content/cuda_softmax_artifacts"
ARTIFACT_POLICY = "ERROR"  # "ERROR", "REUSE", or "OVERWRITE"

WARMUPS = 25
ITERATIONS = 100
SEQUENCE_LENGTHS = [128, 255, 512, 768, 1024, 1536, 2048]
CORRECTNESS_SEQUENCE_LENGTHS = [31, 32, 33, 63, 64, 127, 128, 255, 511, 768, 1023]
CORRECTNESS_INPUT_FAMILIES = [
    "normal", "random_x10", "random_x100", "random_x1000",
    "zeros", "equal", "dominant_positive", "dominant_negative",
]

BATCH = 1
HEADS = 8
BATCH_HEADS = BATCH * HEADS
HEAD_DIM = 64
DTYPE = "float32"
RTOL = 1e-5
ATOL = 1e-6
RANDOM_SEED = 1234
SUPPORTED_BLOCK_SIZES = [128, 256, 512]
PROVISIONAL_BLOCK_SIZE = 256
PROFILE_SEQUENCE_LENGTH = 512

RUN_CORRECTNESS = True
RUN_HISTORICAL_COMMITS = True
RUN_LAUNCH_CONFIG = True
RUN_SOFTMAX_BENCHMARKS = True
RUN_TORCH_COMPILE = True
RUN_ATTENTION_BENCHMARKS = True
RUN_PYTORCH_PROFILER = True
RUN_NSIGHT = True
GENERATE_FIGURES = True
BACKUP_TO_DRIVE = False

# These hashes were discovered from this repository's Git history. The
# historical section verifies that each object and expected subject exist before
# checkout; edit or clear this mapping if your repository history differs.
IMPLEMENTATION_COMMITS = {
    "row_serial": "8f07d7628872550df76990a4fc711d1c84c14c8b",
    "block_shared_tree": "f9420de03a430e9cf768210184ec22fc6b7ddc2a",
    "warp_reduction": "a3736910273bdc2ee52968c49381ec2cf963c876",
}

EXPECTED_COMMIT_SUBJECTS = {
    "row_serial": "record hardware software and git metadata in benchmark CSV",
    "block_shared_tree": "stabilize block-parallel implementation",
    "warp_reduction": "benchmark warp-reduction optimization against prior commit",
}
'''
        ),
        markdown(
            r"""
# 2. Runtime utilities and artifact directories

**What this section does:** Imports standard tools, creates the required artifact
tree, defines checked subprocess/file helpers, and seeds Python and PyTorch.

**Why necessary:** Every stage must save its evidence immediately and must not
silently overwrite an earlier run.

**What to expect:** A printed artifact path and policy. `ERROR` stops on an
existing file; `REUSE` preserves it; `OVERWRITE` must be chosen explicitly.

**What to save:** The complete `artifacts/` directory produced here.

**Paper meaning:** This directory is the evidence bundle behind every claim.
"""
        ),
        code(
            r'''
from __future__ import annotations

import csv
import importlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import socket
import statistics
import subprocess
import sys
import textwrap
import time
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping, Sequence
import zipfile

import torch

ARTIFACTS = Path(ARTIFACT_DIR)
ARTIFACT_SUBDIRECTORIES = [
    "environment", "build", "correctness", "benchmarks/raw",
    "benchmarks/summaries", "attention", "profiler/nsight", "figures",
    "tables", "logs", "metadata",
]
for relative in ARTIFACT_SUBDIRECTORIES:
    (ARTIFACTS / relative).mkdir(parents=True, exist_ok=True)

if ARTIFACT_POLICY not in {"ERROR", "REUSE", "OVERWRITE"}:
    raise ValueError("ARTIFACT_POLICY must be ERROR, REUSE, or OVERWRITE")

RUN_TIMESTAMP = datetime.now(timezone.utc).isoformat()
STATE: dict[str, Any] = {
    "run_timestamp": RUN_TIMESTAMP,
    "failed_experiments": [],
    "skipped_experiments": [],
    "produced_files": [],
}

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def claim_output(path: Path) -> bool:
    """Return whether a stage may write path under the configured policy."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        return True
    if ARTIFACT_POLICY == "REUSE":
        print(f"REUSE: {path}")
        return False
    if ARTIFACT_POLICY == "OVERWRITE":
        print(f"OVERWRITE: {path}")
        return True
    raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")

def register_file(path: Path) -> None:
    relative = str(path.relative_to(ARTIFACTS))
    if relative not in STATE["produced_files"]:
        STATE["produced_files"].append(relative)

def write_text(path: Path, content: str) -> bool:
    if not claim_output(path):
        register_file(path)
        return False
    path.write_text(content, encoding="utf-8")
    register_file(path)
    return True

def write_json(path: Path, payload: Any) -> bool:
    return write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")

def checkpoint_json(path: Path, payload: Any) -> None:
    """Update an already-claimed stage artifact after each completed case."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    register_file(path)

def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> bool:
    if not claim_output(path):
        register_file(path)
        return False
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    register_file(path)
    return True

def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as input_file:
        return list(csv.DictReader(input_file))

def run_command(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    log_path: Path | None = None,
    check: bool = True,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    print("$", " ".join(command))
    started = time.perf_counter()
    result = subprocess.run(
        list(command), cwd=cwd, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    duration = time.perf_counter() - started
    output = result.stdout or ""
    print(output[-8000:])
    if log_path is not None:
        if claim_output(log_path):
            log_path.write_text(output, encoding="utf-8")
        register_file(log_path)
    print(f"return_code={result.returncode} duration_seconds={duration:.3f}")
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed with code {result.returncode}: {command}")
    return result

print(f"Artifacts: {ARTIFACTS}")
print(f"Policy: {ARTIFACT_POLICY}")
'''
        ),
        markdown(
            r"""
# 3. Validate and record the Colab GPU environment

**What this section does:** Runs `nvidia-smi`, `nvcc --version`, and PyTorch CUDA
queries, then writes human- and machine-readable environment reports.

**Why necessary:** CPU or MPS timing cannot answer a CUDA research question, and
Colab may assign different GPUs between sessions.

**What to expect:** `CUDA_READY: True`, one NVIDIA GPU name, compute capability,
and toolkit/runtime versions. The notebook stops here if CUDA is unavailable.

**What to save:** `artifacts/environment/environment.json` and `.txt`.

**Paper meaning:** These values define the hardware/software experimental setup
and should accompany every quantitative table.
"""
        ),
        code(
            r'''
environment_path = ARTIFACTS / "environment/environment.json"
environment_text_path = ARTIFACTS / "environment/environment.txt"

nvidia_smi = run_command(["nvidia-smi"], check=False)
nvcc = run_command(["nvcc", "--version"], check=False)
cxx = run_command(["g++", "--version"], check=False)

CUDA_READY = bool(torch.cuda.is_available() and nvidia_smi.returncode == 0)
device_index = torch.cuda.current_device() if CUDA_READY else None
device_properties = torch.cuda.get_device_properties(device_index) if CUDA_READY else None

ENVIRONMENT = {
    "timestamp": utc_now(),
    "python_version": platform.python_version(),
    "operating_system": platform.platform(),
    "hostname": socket.gethostname(),
    "pytorch_version": torch.__version__,
    "pytorch_cuda_version": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "cuda_device_count": torch.cuda.device_count(),
    "gpu_name": torch.cuda.get_device_name(device_index) if CUDA_READY else None,
    "compute_capability": (
        ".".join(map(str, torch.cuda.get_device_capability(device_index)))
        if CUDA_READY else None
    ),
    "total_gpu_memory_bytes": device_properties.total_memory if device_properties else None,
    "nvidia_smi_return_code": nvidia_smi.returncode,
    "nvcc_return_code": nvcc.returncode,
    "cxx_return_code": cxx.returncode,
    "nvidia_smi_output": nvidia_smi.stdout,
    "nvcc_output": nvcc.stdout,
    "cxx_output": cxx.stdout,
}
if environment_path.exists() and ARTIFACT_POLICY == "REUSE":
    previous_environment = json.loads(environment_path.read_text(encoding="utf-8"))
    comparison_fields = [
        "gpu_name", "compute_capability", "pytorch_version",
        "pytorch_cuda_version", "python_version",
    ]
    mismatches = {
        field: (previous_environment.get(field), ENVIRONMENT.get(field))
        for field in comparison_fields
        if previous_environment.get(field) != ENVIRONMENT.get(field)
    }
    if mismatches:
        raise RuntimeError(
            "REUSE refused because the restored artifacts came from a different "
            f"environment: {mismatches}"
        )
write_json(environment_path, ENVIRONMENT)
write_text(
    environment_text_path,
    "\n".join(f"{key}: {value}" for key, value in ENVIRONMENT.items()) + "\n",
)
STATE["cuda_environment"] = "PASS" if CUDA_READY else "FAIL"
print(json.dumps(ENVIRONMENT, indent=2, default=str))

if not CUDA_READY:
    raise RuntimeError(
        "CUDA is unavailable. In Colab choose Runtime → Change runtime type → "
        "NVIDIA GPU, reconnect, and rerun from the configuration cell. CPU or "
        "Apple MPS measurements are not accepted as CUDA evidence."
    )
'''
        ),
        markdown(
            r"""
# 4. Obtain the repository source

**What this section does:** Clones `REPO_URL` and checks out `BRANCH`, or uploads
and extracts a ZIP when `SOURCE_MODE="ZIP"`.

**Why necessary:** The measured CUDA implementation and its Git revision must be
identifiable. A ZIP without `.git` can run, but cannot support commit history
comparisons or strong provenance.

**What to expect:** The repository root, checked-out revision, and clean/dirty
status are printed. Existing work directories are never deleted silently.

**What to save:** Git state is captured in the next section.

**Paper meaning:** This connects every experiment to reviewable source code.

For a private GitHub repository, do not paste a token into a saved notebook.
Use a temporary Colab Secret or upload a ZIP. Revoke temporary credentials after
the run. The default public URL requires no secret.
"""
        ),
        code(
            r'''
work_dir = Path(WORK_DIR)

if SOURCE_MODE == "GITHUB":
    if not REPO_URL:
        raise ValueError("REPO_URL must be configured for GITHUB source mode")
    if work_dir.exists():
        if ARTIFACT_POLICY != "REUSE":
            raise FileExistsError(
                f"{work_dir} already exists. Use a fresh runtime or choose REUSE."
            )
        print(f"Reusing existing repository: {work_dir}")
    else:
        clone_command = ["git", "clone", REPO_URL, str(work_dir)]
        if BRANCH:
            clone_command[2:2] = ["--branch", BRANCH, "--single-branch"]
        run_command(clone_command, cwd=work_dir.parent)
elif SOURCE_MODE == "ZIP":
    from google.colab import files
    uploaded = files.upload()
    zip_names = [name for name in uploaded if name.lower().endswith(".zip")]
    if len(zip_names) != 1:
        raise ValueError("Upload exactly one repository ZIP")
    if work_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {work_dir}")
    extraction_root = work_dir.parent / "uploaded_repository"
    extraction_root.mkdir(parents=True, exist_ok=False)
    zip_path = extraction_root / zip_names[0]
    zip_path.write_bytes(uploaded[zip_names[0]])
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extraction_root)
    candidates = [path for path in extraction_root.rglob("AGENTS.md")]
    if len(candidates) != 1:
        raise RuntimeError("Could not identify one repository root in the ZIP")
    shutil.move(str(candidates[0].parent), str(work_dir))
else:
    raise ValueError("SOURCE_MODE must be GITHUB or ZIP")

if not (work_dir / "AGENTS.md").is_file():
    raise RuntimeError("The configured source is not the expected repository")

os.chdir(work_dir)
if str(work_dir) not in sys.path:
    sys.path.insert(0, str(work_dir))
print(f"Repository root: {work_dir}")
'''
        ),
        markdown(
            r"""
# 5. Record Git provenance

**What this section does:** Captures `HEAD`, status, and the latest full commit
record before building or measuring.

**Why necessary:** Results from a dirty tree or unknown archive cannot be traced
to a reproducible implementation.

**What to expect:** A 40-character commit hash and preferably an empty short
status. ZIP mode without `.git` prints the required provenance warning.

**What to save:** `artifacts/metadata/git_state.txt` and `git_commit.txt`.

**Paper meaning:** The hash is the implementation identifier in result rows.
"""
        ),
        code(
            r'''
git_directory = work_dir / ".git"
if git_directory.exists():
    git_commit_result = run_command(["git", "rev-parse", "HEAD"], cwd=work_dir)
    git_status_result = run_command(["git", "status", "--short"], cwd=work_dir)
    git_log_result = run_command(
        ["git", "log", "-1", "--format=fuller"], cwd=work_dir
    )
    GIT_COMMIT = git_commit_result.stdout.strip()
    GIT_STATUS = git_status_result.stdout.strip()
    git_state = (
        f"commit: {GIT_COMMIT}\n"
        f"dirty: {bool(GIT_STATUS)}\n"
        f"status:\n{GIT_STATUS or '(clean)'}\n\n"
        f"latest commit:\n{git_log_result.stdout}"
    )
else:
    GIT_COMMIT = "UNAVAILABLE"
    GIT_STATUS = "Git metadata absent from uploaded archive"
    git_state = "WARNING: Git provenance unavailable from uploaded archive.\n"
    print(git_state)

existing_commit_path = ARTIFACTS / "metadata/git_commit.txt"
if existing_commit_path.exists() and ARTIFACT_POLICY == "REUSE":
    previous_commit = existing_commit_path.read_text(encoding="utf-8").strip()
    if previous_commit != GIT_COMMIT:
        raise RuntimeError(
            f"REUSE refused: artifact commit {previous_commit} != source {GIT_COMMIT}"
        )

write_text(ARTIFACTS / "metadata/git_commit.txt", GIT_COMMIT + "\n")
write_text(ARTIFACTS / "metadata/git_state.txt", git_state)
STATE["git_commit"] = GIT_COMMIT
STATE["git_dirty"] = bool(GIT_STATUS and GIT_COMMIT != "UNAVAILABLE")
'''
        ),
        markdown(
            r"""
# 6. Install requirements and build the CUDA extension

**What this section does:** Uses the repository's own development requirements
and `scripts/build_extension.sh`, captures complete logs/metadata, and imports
`cuda_attention._C` explicitly.

**Why necessary:** A successful Python import is the boundary between source
being present and the compiled operator being callable.

**What to expect:** Both commands return zero and the extension import succeeds.
If the build fails, preserve the log and stop before benchmark sections.

**What to save:** Build/install logs and `build_metadata.json`.

**Paper meaning:** This is reproducibility evidence for the compilation stage,
not proof of numerical correctness or performance.
"""
        ),
        code(
            r'''
requirements_path = work_dir / "requirements-dev.txt"
if requirements_path.is_file():
    install_result = run_command(
        [sys.executable, "-m", "pip", "install", "-r", str(requirements_path)],
        cwd=work_dir,
        log_path=ARTIFACTS / "build/dependency_install_log.txt",
        check=False,
    )
else:
    install_result = subprocess.CompletedProcess([], 0, "requirements-dev.txt absent")

build_script = work_dir / "scripts/build_extension.sh"
if not build_script.is_file():
    raise FileNotFoundError(
        "Repository build script scripts/build_extension.sh was not found; "
        "the notebook will not guess a different build mechanism."
    )

build_started = time.perf_counter()
build_result = run_command(
    ["bash", str(build_script)],
    cwd=work_dir,
    log_path=ARTIFACTS / "build/build_log.txt",
    check=False,
)
build_duration = time.perf_counter() - build_started

BUILD_METADATA = {
    "timestamp": utc_now(),
    "git_commit": GIT_COMMIT,
    "command": ["bash", str(build_script)],
    "return_code": build_result.returncode,
    "duration_seconds": build_duration,
    "dependency_install_return_code": install_result.returncode,
}

extension_import_error = None
if build_result.returncode == 0:
    try:
        importlib.invalidate_caches()
        extension = importlib.import_module("cuda_attention._C")
        EXTENSION_READY = hasattr(extension, "fused_causal_softmax")
    except Exception as error:
        EXTENSION_READY = False
        extension_import_error = f"{type(error).__name__}: {error}"
else:
    EXTENSION_READY = False

BUILD_METADATA["extension_import_ready"] = EXTENSION_READY
BUILD_METADATA["extension_import_error"] = extension_import_error
write_json(ARTIFACTS / "build/build_metadata.json", BUILD_METADATA)
STATE["cuda_extension_build"] = "PASS" if EXTENSION_READY else "FAIL"
print(json.dumps(BUILD_METADATA, indent=2))

if not EXTENSION_READY:
    STATE["failed_experiments"].append("CUDA extension build/import")
    raise RuntimeError(
        "CUDA extension build/import failed. Preserve artifacts/build and do "
        "not execute correctness or benchmark sections."
    )
'''
        ),
    ]

    cells.extend(
        [
            markdown(
                r"""
# 7. CUDA correctness gate

**What this section does:** Runs the repository CUDA test module and a structured
matrix over every required sequence length and input family using the fixed FP32
tolerances `rtol=1e-5`, `atol=1e-6`.

**Why necessary:** Performance measurements are rejected until the custom output
matches the trusted PyTorch reference, causal positions are exactly zero, rows
sum to one, and shape/dtype/device invariants hold.

**What to expect:** A passing pytest log, one CSV row per correctness case, and a
summary JSON. Failures remain in the CSV and block later custom benchmarks.

**What to save:** Everything under `artifacts/correctness/`.

**Paper meaning:** These artifacts support the numerical-correctness section;
they are not latency measurements.
"""
            ),
            code(
                r'''
from cuda_attention.operator import fused_causal_softmax
from cuda_attention.reference import causal_allowed_mask, causal_scaled_softmax

pytest_result = run_command(
    [sys.executable, "-m", "pytest", "-q", "tests/test_cuda_operator.py", "-rs"],
    cwd=work_dir,
    log_path=ARTIFACTS / "correctness/pytest_output.txt",
    check=False,
)

CORRECTNESS_FIELDS = [
    "git_commit", "sequence_length", "rows", "input_family", "dtype", "scale",
    "max_absolute_error", "max_relative_error", "max_row_sum_error",
    "masked_positions_exactly_zero", "contains_nan", "contains_inf",
    "shape_correct", "dtype_correct", "device_correct", "pass_fail",
    "rtol", "atol", "gpu_name", "compute_capability", "timestamp",
]
correctness_path = ARTIFACTS / "correctness/correctness_results.csv"

def make_correctness_scores(
    sequence_length: int,
    rows: int,
    family: str,
) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(
        RANDOM_SEED + sequence_length + sum(map(ord, family))
    )
    if family == "zeros":
        scores = torch.zeros(rows, sequence_length, device="cuda", dtype=torch.float32)
    elif family == "equal":
        scores = torch.full(
            (rows, sequence_length), 7.0, device="cuda", dtype=torch.float32
        )
    else:
        scores = torch.randn(
            rows, sequence_length, generator=generator,
            device="cuda", dtype=torch.float32,
        )
        if family.startswith("random_x"):
            scores.mul_(float(family.removeprefix("random_x")))
        elif family == "dominant_positive":
            scores[:, 0] = 1000.0
        elif family == "dominant_negative":
            scores[:, 0] = -1000.0
        elif family != "normal":
            raise ValueError(f"Unknown correctness family: {family}")
    return scores

if RUN_CORRECTNESS and (not correctness_path.exists() or ARTIFACT_POLICY != "REUSE"):
    if not claim_output(correctness_path):
        correctness_rows = read_csv(correctness_path)
    else:
        correctness_rows: list[dict[str, Any]] = []
        scale = 1.0 / math.sqrt(HEAD_DIM)
        for sequence_length in CORRECTNESS_SEQUENCE_LENGTHS:
            rows = 2 * sequence_length
            allowed = causal_allowed_mask(rows, sequence_length, device="cuda")
            for family in CORRECTNESS_INPUT_FAMILIES:
                scores = make_correctness_scores(sequence_length, rows, family)
                expected = causal_scaled_softmax(scores, scale)
                try:
                    actual = fused_causal_softmax(
                        scores, scale, block_size=PROVISIONAL_BLOCK_SIZE
                    )
                    torch.cuda.synchronize()
                    difference = (actual - expected).abs()
                    relative = difference / expected.abs().clamp_min(
                        torch.finfo(torch.float32).tiny
                    )
                    row_sum_error = (actual.sum(dim=-1) - 1.0).abs().max()
                    masked_zero = bool(
                        torch.count_nonzero(actual.masked_select(~allowed)).item() == 0
                    )
                    shape_correct = actual.shape == scores.shape
                    dtype_correct = actual.dtype == scores.dtype
                    device_correct = actual.device == scores.device
                    contains_nan = bool(torch.isnan(actual).any().item())
                    contains_inf = bool(torch.isinf(actual).any().item())
                    try:
                        torch.testing.assert_close(
                            actual, expected, rtol=RTOL, atol=ATOL
                        )
                        close = True
                    except AssertionError:
                        close = False
                    passed = all(
                        [close, masked_zero, shape_correct, dtype_correct,
                         device_correct, not contains_nan, not contains_inf]
                    )
                    record = {
                        "git_commit": GIT_COMMIT,
                        "sequence_length": sequence_length,
                        "rows": rows,
                        "input_family": family,
                        "dtype": str(scores.dtype).removeprefix("torch."),
                        "scale": scale,
                        "max_absolute_error": float(difference.max().item()),
                        "max_relative_error": float(relative.max().item()),
                        "max_row_sum_error": float(row_sum_error.item()),
                        "masked_positions_exactly_zero": masked_zero,
                        "contains_nan": contains_nan,
                        "contains_inf": contains_inf,
                        "shape_correct": shape_correct,
                        "dtype_correct": dtype_correct,
                        "device_correct": device_correct,
                        "pass_fail": "PASS" if passed else "FAIL",
                        "rtol": RTOL,
                        "atol": ATOL,
                        "gpu_name": ENVIRONMENT["gpu_name"],
                        "compute_capability": ENVIRONMENT["compute_capability"],
                        "timestamp": utc_now(),
                    }
                except Exception as error:
                    record = {
                        "git_commit": GIT_COMMIT,
                        "sequence_length": sequence_length,
                        "rows": rows,
                        "input_family": family,
                        "dtype": DTYPE,
                        "scale": scale,
                        "max_absolute_error": "",
                        "max_relative_error": "",
                        "max_row_sum_error": "",
                        "masked_positions_exactly_zero": False,
                        "contains_nan": "",
                        "contains_inf": "",
                        "shape_correct": False,
                        "dtype_correct": False,
                        "device_correct": False,
                        "pass_fail": "FAIL",
                        "rtol": RTOL,
                        "atol": ATOL,
                        "gpu_name": ENVIRONMENT["gpu_name"],
                        "compute_capability": ENVIRONMENT["compute_capability"],
                        "timestamp": utc_now(),
                    }
                    print(f"Correctness failure S={sequence_length} family={family}: {error}")
                correctness_rows.append(record)
                with correctness_path.open("w", newline="", encoding="utf-8") as output_file:
                    writer = csv.DictWriter(output_file, fieldnames=CORRECTNESS_FIELDS)
                    writer.writeheader()
                    writer.writerows(correctness_rows)
        register_file(correctness_path)
elif correctness_path.exists():
    correctness_rows = read_csv(correctness_path)
else:
    correctness_rows = []
    STATE["skipped_experiments"].append("structured CUDA correctness")

CORRECTNESS_READY = bool(
    pytest_result.returncode == 0
    and correctness_rows
    and all(row["pass_fail"] == "PASS" for row in correctness_rows)
)
correctness_summary = {
    "timestamp": utc_now(),
    "git_commit": GIT_COMMIT,
    "pytest_return_code": pytest_result.returncode,
    "case_count": len(correctness_rows),
    "passed_cases": sum(row["pass_fail"] == "PASS" for row in correctness_rows),
    "failed_cases": sum(row["pass_fail"] != "PASS" for row in correctness_rows),
    "rtol": RTOL,
    "atol": ATOL,
    "correctness_ready_for_benchmark": CORRECTNESS_READY,
}
write_json(ARTIFACTS / "correctness/correctness_summary.json", correctness_summary)
STATE["cuda_correctness"] = "PASS" if CORRECTNESS_READY else "FAIL"
print(json.dumps(correctness_summary, indent=2))

if RUN_CORRECTNESS and not CORRECTNESS_READY:
    raise RuntimeError(
        "CUDA correctness did not pass. Do not run custom benchmarks until the "
        "failure is understood without weakening tolerances."
    )
'''
            ),
            markdown(
                r"""
# 8. Benchmark utilities and launch-configuration experiment

**What this section does:** Runs the repository custom benchmark at 128, 256,
and 512 threads for every planned length, retains every CUDA-event sample, and
selects a configuration only after all three matrices are complete.

**Why necessary:** Block size changes parallel work, idle lanes, and cross-warp
coordination. A familiar default is not evidence of optimality.

**What to expect:** Raw and summary CSVs plus `launch_selection.json`. The rule
is the lowest median per-shape relative latency, with per-length wins reported.

**What to save:** Launch raw samples, summaries, and selection JSON.

**Paper meaning:** These support the launch-tuning table and H4. A selected size
is workload/environment-specific, not universally best.
"""
            ),
            code(
                r'''
RAW_SOFTMAX_FIELDS = [
    "git_commit", "implementation", "implementation_description",
    "sequence_length", "rows", "columns", "dtype", "scale", "warmups",
    "iterations", "sample_index", "latency_us", "launch_block_size",
    "compile_warmups", "gpu_name", "compute_capability", "pytorch_version",
    "cuda_version", "timestamp",
]

def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot summarize empty samples")
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight

def classify_implementation(description: str) -> str:
    lowered = description.lower()
    if "torch.compile" in lowered:
        return "torch_compile"
    if "pytorch eager" in lowered:
        return "pytorch_eager"
    if "custom cuda" in lowered:
        return "custom_cuda"
    raise ValueError(f"Unknown implementation description: {description}")

def normalize_repository_samples(
    rows: Sequence[Mapping[str, str]],
    *,
    forced_implementation: str | None = None,
    forced_description: str | None = None,
) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        description = forced_description or row["implementation_description"]
        implementation = forced_implementation or classify_implementation(description)
        normalized.append(
            {
                "git_commit": row["git_commit"],
                "implementation": implementation,
                "implementation_description": description,
                "sequence_length": int(row["sequence_length"]),
                "rows": int(row["rows"]),
                "columns": int(row["columns"]),
                "dtype": row["dtype"],
                "scale": 1.0 / math.sqrt(HEAD_DIM),
                "warmups": int(row["warmups"]),
                "iterations": int(row["iterations"]),
                "sample_index": int(row["sample_index"]),
                "latency_us": float(row.get("sample_us", row.get("latency_us", "nan"))),
                "launch_block_size": row.get("launch_block_size", ""),
                "compile_warmups": int(row.get("compile_warmups", 0) or 0),
                "gpu_name": row["gpu_name"],
                "compute_capability": row["compute_capability"],
                "pytorch_version": row["pytorch_version"],
                "cuda_version": row["cuda_version"],
                "timestamp": row["timestamp"],
            }
        )
    return normalized

def summarize_softmax_rows(
    raw_rows: Sequence[Mapping[str, Any]],
    *,
    baseline: str | None,
) -> list[dict[str, Any]]:
    group_fields = [
        "git_commit", "implementation", "implementation_description",
        "sequence_length", "rows", "columns", "dtype", "warmups", "iterations",
        "launch_block_size", "compile_warmups", "gpu_name", "compute_capability",
        "pytorch_version", "cuda_version",
    ]
    grouped: dict[tuple[Any, ...], list[float]] = {}
    timestamps: dict[tuple[Any, ...], str] = {}
    for row in raw_rows:
        key = tuple(row[field] for field in group_fields)
        latency = float(row["latency_us"])
        if not math.isfinite(latency) or latency <= 0:
            raise ValueError("CUDA latency samples must be finite and positive")
        grouped.setdefault(key, []).append(latency)
        timestamps[key] = str(row["timestamp"])
    summaries = []
    for key, samples in grouped.items():
        group = dict(zip(group_fields, key, strict=True))
        if len(samples) != int(group["iterations"]):
            raise ValueError("Raw sample count does not match iterations")
        median_us = statistics.median(samples)
        summaries.append(
            {
                **group,
                "median_us": median_us,
                "p25_us": percentile(samples, 0.25),
                "p75_us": percentile(samples, 0.75),
                "elements_per_second": (
                    int(group["rows"]) * int(group["columns"]) * 1_000_000.0
                    / median_us
                ),
                "speedup_vs_eager": "",
                "timestamp": timestamps[key],
            }
        )
    if baseline is not None:
        baseline_by_shape = {
            int(row["sequence_length"]): row
            for row in summaries
            if row["implementation"] == baseline
        }
        for row in summaries:
            base = baseline_by_shape.get(int(row["sequence_length"]))
            if base is None:
                row["speedup_vs_eager"] = (
                    "SPEEDUP NOT COMPUTED: incompatible experimental conditions."
                )
                continue
            compatible = all(
                row[field] == base[field]
                for field in [
                    "sequence_length", "rows", "columns", "dtype", "warmups",
                    "iterations", "gpu_name", "compute_capability",
                    "pytorch_version", "cuda_version",
                ]
            )
            row["speedup_vs_eager"] = (
                float(base["median_us"]) / float(row["median_us"])
                if compatible
                else "SPEEDUP NOT COMPUTED: incompatible experimental conditions."
            )
    return sorted(
        summaries,
        key=lambda row: (int(row["sequence_length"]), str(row["implementation"])),
    )

def run_repository_softmax_case(
    *, implementation: str, sequence_length: int, block_size: int,
) -> list[dict[str, Any]]:
    temporary = ARTIFACTS / "logs" / (
        f"temporary_{implementation}_S{sequence_length}_B{block_size}.csv"
    )
    if temporary.exists():
        temporary.unlink()
    command = [
        sys.executable, "benchmarks/benchmark_softmax.py",
        "--implementation", implementation,
        "--sequence-length", str(sequence_length),
        "--warmups", str(WARMUPS), "--iterations", str(ITERATIONS),
        "--block-size", str(block_size), "--output", str(temporary),
    ]
    result = run_command(command, cwd=work_dir, check=False)
    if result.returncode != 0 or not temporary.is_file():
        log = ARTIFACTS / "logs" / f"failed_{implementation}_S{sequence_length}_B{block_size}.txt"
        write_text(log, result.stdout or "No command output\n")
        raise RuntimeError(f"Benchmark failed for {implementation}, S={sequence_length}")
    rows = normalize_repository_samples(read_csv(temporary))
    temporary.unlink()
    return rows

if not CORRECTNESS_READY:
    raise RuntimeError("Launch benchmarking requires the passed correctness gate")

launch_raw_path = ARTIFACTS / "benchmarks/raw/launch_configuration_raw.csv"
launch_summary_path = ARTIFACTS / "benchmarks/summaries/launch_configuration_summary.csv"

if RUN_LAUNCH_CONFIG and (not launch_raw_path.exists() or ARTIFACT_POLICY != "REUSE"):
    if not claim_output(launch_raw_path):
        launch_raw_rows = read_csv(launch_raw_path)
    else:
        launch_raw_rows: list[dict[str, Any]] = []
        for block_size in SUPPORTED_BLOCK_SIZES:
            for sequence_length in SEQUENCE_LENGTHS:
                case_rows = run_repository_softmax_case(
                    implementation="custom",
                    sequence_length=sequence_length,
                    block_size=block_size,
                )
                launch_raw_rows.extend(case_rows)
                with launch_raw_path.open("w", newline="", encoding="utf-8") as output_file:
                    writer = csv.DictWriter(output_file, fieldnames=RAW_SOFTMAX_FIELDS)
                    writer.writeheader()
                    writer.writerows(launch_raw_rows)
        register_file(launch_raw_path)
elif launch_raw_path.exists():
    launch_raw_rows = read_csv(launch_raw_path)
else:
    launch_raw_rows = []
    STATE["skipped_experiments"].append("launch configuration")

if launch_raw_rows:
    launch_summaries = summarize_softmax_rows(launch_raw_rows, baseline=None)
    summary_fields = list(launch_summaries[0])
    write_csv(launch_summary_path, launch_summaries, summary_fields)

    by_size: dict[int, dict[int, float]] = {}
    for row in launch_summaries:
        size = int(row["launch_block_size"])
        by_size.setdefault(size, {})[int(row["sequence_length"])] = float(row["median_us"])
    expected_sizes = set(SUPPORTED_BLOCK_SIZES)
    expected_shapes = set(SEQUENCE_LENGTHS)
    complete = set(by_size) == expected_sizes and all(
        set(by_size[size]) == expected_shapes for size in expected_sizes
    )
    if not complete:
        raise RuntimeError("Launch selection requires complete 128/256/512 matrices")
    relative: dict[int, list[float]] = {size: [] for size in expected_sizes}
    wins = {size: 0 for size in expected_sizes}
    for sequence_length in SEQUENCE_LENGTHS:
        best = min(by_size[size][sequence_length] for size in expected_sizes)
        for size in expected_sizes:
            relative[size].append(by_size[size][sequence_length] / best)
            if by_size[size][sequence_length] == best:
                wins[size] += 1
    scores = {size: statistics.median(relative[size]) for size in expected_sizes}
    SELECTED_BLOCK_SIZE = min(expected_sizes, key=lambda size: (scores[size], size))
    launch_selection = {
        "selected_block_size": SELECTED_BLOCK_SIZE,
        "selection_rule": "lowest median per-shape relative latency",
        "median_relative_latency": scores,
        "per_sequence_wins": wins,
        "sequence_lengths": SEQUENCE_LENGTHS,
        "git_commit": GIT_COMMIT,
        "gpu_name": ENVIRONMENT["gpu_name"],
        "timestamp": utc_now(),
    }
    write_json(ARTIFACTS / "benchmarks/summaries/launch_selection.json", launch_selection)
    STATE["launch_configuration"] = "COMPLETE"
    print(json.dumps(launch_selection, indent=2))
else:
    SELECTED_BLOCK_SIZE = PROVISIONAL_BLOCK_SIZE
    STATE["launch_configuration"] = "INCOMPLETE"
    print("No complete launch data: retaining provisional block size without selection")
'''
            ),
            markdown(
                r"""
# 9. Matched eager, torch.compile, and custom-softmax benchmark

**What this section does:** Measures equivalent scale + causal mask + softmax
work on the same deterministic score convention. Inputs and masks are outside
timing; compilation has a separate untimed first call; CUDA events capture every
steady-state sample.

**Why necessary:** Eager alone is a weak framework baseline. `torch.compile`
tests whether framework-level fusion narrows the custom-kernel advantage.

**What to expect:** `softmax_raw.csv` and `softmax_summary.csv`, with speedup only
when workload/environment controls match. Compilation failure is preserved; the
applicable eager/custom results remain usable.

**What to save:** Both softmax CSVs and any failure logs.

**Paper meaning:** These are the primary kernel latency and throughput results.
Synchronization is necessary because CUDA launches asynchronously; without the
ending event synchronization, Python could return before GPU work completes.
"""
            ),
            code(
                r'''
softmax_raw_path = ARTIFACTS / "benchmarks/raw/softmax_raw.csv"
softmax_summary_path = ARTIFACTS / "benchmarks/summaries/softmax_summary.csv"

if RUN_SOFTMAX_BENCHMARKS and (not softmax_raw_path.exists() or ARTIFACT_POLICY != "REUSE"):
    if not claim_output(softmax_raw_path):
        softmax_raw_rows = read_csv(softmax_raw_path)
    else:
        softmax_raw_rows: list[dict[str, Any]] = []
        requested = "all" if RUN_TORCH_COMPILE else "both"
        for sequence_length in SEQUENCE_LENGTHS:
            try:
                case_rows = run_repository_softmax_case(
                    implementation=requested,
                    sequence_length=sequence_length,
                    block_size=SELECTED_BLOCK_SIZE,
                )
            except RuntimeError as error:
                if RUN_TORCH_COMPILE:
                    STATE["failed_experiments"].append(
                        f"torch.compile or combined softmax at S={sequence_length}: {error}"
                    )
                    print("Combined path failed; retrying applicable eager/custom paths")
                    case_rows = []
                    for fallback in ("eager", "custom"):
                        case_rows.extend(
                            run_repository_softmax_case(
                                implementation=fallback,
                                sequence_length=sequence_length,
                                block_size=SELECTED_BLOCK_SIZE,
                            )
                        )
                else:
                    raise
            softmax_raw_rows.extend(case_rows)
            with softmax_raw_path.open("w", newline="", encoding="utf-8") as output_file:
                writer = csv.DictWriter(output_file, fieldnames=RAW_SOFTMAX_FIELDS)
                writer.writeheader()
                writer.writerows(softmax_raw_rows)
        register_file(softmax_raw_path)
elif softmax_raw_path.exists():
    softmax_raw_rows = read_csv(softmax_raw_path)
else:
    softmax_raw_rows = []
    STATE["skipped_experiments"].append("softmax benchmarks")

if softmax_raw_rows:
    softmax_summaries = summarize_softmax_rows(
        softmax_raw_rows, baseline="pytorch_eager"
    )
    write_csv(softmax_summary_path, softmax_summaries, list(softmax_summaries[0]))
    STATE["softmax_benchmarks"] = "COMPLETE"
    print(f"Saved {len(softmax_raw_rows)} raw softmax samples")
else:
    softmax_summaries = []
    STATE["softmax_benchmarks"] = "INCOMPLETE"
'''
            ),
            markdown(
                r"""
# 10. Optional historical Git-revision experiment

**What this section does:** Verifies configured commit objects and subjects,
checks out each revision, cleans only build products, rebuilds, runs its CUDA
tests, benchmarks it in isolated subprocesses, and restores the original branch.

**Why necessary:** Git history—not duplicate CUDA files—contains the serial,
shared-tree, and warp stages needed to evaluate H1–H3.

**What to expect:** Some historical commits may fail against a newer Colab
toolchain. The notebook applies one recorded header-only compatibility change
when an old revision uses `CUDART_INF_F` without its defining CUDA header.
Other failures are logged and retained instead of silently omitted.

**What to save:** Historical raw/summary CSVs and per-stage build/test logs.

**Paper meaning:** Only verified, correctness-passing commits can appear in the
optimization-evolution comparison.
"""
            ),
            code(
                r'''
historical_raw_path = ARTIFACTS / "benchmarks/raw/historical_raw.csv"
historical_summary_path = ARTIFACTS / "benchmarks/summaries/historical_summary.csv"

def clean_extension_build_products() -> None:
    build_directory = work_dir / "build"
    if build_directory.exists():
        shutil.rmtree(build_directory)
    for extension_path in (work_dir / "cuda_attention").glob("_C*.so"):
        extension_path.unlink()

def verify_historical_commit(stage: str, commit: str) -> str:
    exists = run_command(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=work_dir, check=False,
    )
    if exists.returncode != 0:
        raise ValueError(f"Configured historical commit does not exist: {commit}")
    subject = run_command(
        ["git", "show", "-s", "--format=%s", commit], cwd=work_dir
    ).stdout.strip()
    if subject != EXPECTED_COMMIT_SUBJECTS[stage]:
        raise ValueError(
            f"Historical subject mismatch for {stage}: {subject!r}"
        )
    return subject

def apply_historical_toolchain_compatibility(
    stage: str,
    commit: str,
) -> tuple[Path, str, dict[str, Any]]:
    source_path = work_dir / "csrc/fused_causal_softmax.cu"
    original_source = source_path.read_text(encoding="utf-8")
    patched_source = original_source
    include = "#include <math_constants.h>"
    applied = "CUDART_INF_F" in original_source and include not in original_source
    if applied:
        anchor = "#include <cuda_runtime.h>\n"
        if anchor not in original_source:
            raise RuntimeError(
                f"Cannot apply the recorded CUDA compatibility include to {commit}"
            )
        patched_source = original_source.replace(
            anchor,
            anchor + include + "\n",
            1,
        )
        source_path.write_text(patched_source, encoding="utf-8")
    record = {
        "stage": stage,
        "base_git_commit": commit,
        "applied": applied,
        "file": "csrc/fused_causal_softmax.cu",
        "change": "insert #include <math_constants.h> after cuda_runtime.h",
        "reason": (
            "CUDA 12.8 requires the defining header for CUDART_INF_F; "
            "kernel instructions and benchmark work are unchanged."
        ),
        "git_diff": run_command(
            ["git", "diff", "--", "csrc/fused_causal_softmax.cu"],
            cwd=work_dir,
        ).stdout,
        "timestamp": utc_now(),
    }
    write_json(
        ARTIFACTS / f"metadata/historical_{stage}_compatibility.json",
        record,
    )
    return source_path, original_source, record

if RUN_HISTORICAL_COMMITS and (not historical_raw_path.exists() or ARTIFACT_POLICY != "REUSE"):
    if GIT_COMMIT == "UNAVAILABLE":
        raise RuntimeError("Historical benchmarking requires Git provenance")
    if GIT_STATUS:
        raise RuntimeError("Historical benchmarking refuses a dirty working tree")
    if not claim_output(historical_raw_path):
        historical_raw_rows = read_csv(historical_raw_path)
    else:
        historical_raw_rows: list[dict[str, Any]] = []
        original_branch_result = run_command(
            ["git", "symbolic-ref", "--short", "-q", "HEAD"],
            cwd=work_dir, check=False,
        )
        original_target = original_branch_result.stdout.strip() or GIT_COMMIT
        stage_descriptions = {
            "row_serial": "one-thread-per-row custom CUDA",
            "block_shared_tree": "one-block-per-row shared-tree custom CUDA",
            "warp_reduction": "compact warp-reduction custom CUDA",
        }
        try:
            for stage, commit in IMPLEMENTATION_COMMITS.items():
                verify_historical_commit(stage, commit)
                run_command(["git", "checkout", "--detach", commit], cwd=work_dir)
                clean_extension_build_products()
                source_path, original_source, compatibility = (
                    apply_historical_toolchain_compatibility(stage, commit)
                )
                try:
                    build = run_command(
                        ["bash", "scripts/build_extension.sh"], cwd=work_dir,
                        log_path=ARTIFACTS / f"build/historical_{stage}_build.txt",
                        check=False,
                    )
                    if build.returncode != 0:
                        STATE["failed_experiments"].append(
                            f"historical {stage} build at {commit}"
                        )
                        continue
                    tests = run_command(
                        [sys.executable, "-m", "pytest", "-q", "tests/test_cuda_operator.py", "-rs"],
                        cwd=work_dir,
                        log_path=ARTIFACTS / f"correctness/historical_{stage}_pytest.txt",
                        check=False,
                    )
                    if tests.returncode != 0:
                        STATE["failed_experiments"].append(
                            f"historical {stage} correctness at {commit}"
                        )
                        continue
                    compatibility_suffix = (
                        "; header-only CUDA 12.8 compatibility include applied"
                        if compatibility["applied"] else ""
                    )
                    for sequence_length in SEQUENCE_LENGTHS:
                        temporary = ARTIFACTS / "logs" / f"historical_{stage}_{sequence_length}.csv"
                        if temporary.exists():
                            temporary.unlink()
                        command = [
                            sys.executable, "benchmarks/benchmark_softmax.py",
                            "--implementation", "custom",
                            "--sequence-length", str(sequence_length),
                            "--warmups", str(WARMUPS), "--iterations", str(ITERATIONS),
                            "--output", str(temporary),
                        ]
                        result = run_command(command, cwd=work_dir, check=False)
                        if result.returncode != 0 or not temporary.is_file():
                            STATE["failed_experiments"].append(
                                f"historical {stage} benchmark S={sequence_length}"
                            )
                            continue
                        stage_rows = normalize_repository_samples(
                            read_csv(temporary),
                            forced_implementation=stage,
                            forced_description=(
                                stage_descriptions[stage] + compatibility_suffix
                            ),
                        )
                        historical_raw_rows.extend(stage_rows)
                        temporary.unlink()
                        with historical_raw_path.open(
                            "w", newline="", encoding="utf-8"
                        ) as output_file:
                            writer = csv.DictWriter(output_file, fieldnames=RAW_SOFTMAX_FIELDS)
                            writer.writeheader()
                            writer.writerows(historical_raw_rows)
                finally:
                    source_path.write_text(original_source, encoding="utf-8")
                    clean_extension_build_products()
        finally:
            run_command(["git", "checkout", original_target], cwd=work_dir, check=False)
            clean_extension_build_products()
            restore = run_command(
                ["bash", "scripts/build_extension.sh"], cwd=work_dir,
                log_path=ARTIFACTS / "build/restored_current_build.txt",
                check=False,
            )
            if restore.returncode != 0:
                raise RuntimeError("Failed to restore the current extension build")
        register_file(historical_raw_path)
elif historical_raw_path.exists():
    historical_raw_rows = read_csv(historical_raw_path)
else:
    historical_raw_rows = []
    STATE["skipped_experiments"].append("historical Git revisions")

if historical_raw_rows:
    historical_summaries = summarize_softmax_rows(historical_raw_rows, baseline=None)
    serial_by_shape = {
        int(row["sequence_length"]): row
        for row in historical_summaries if row["implementation"] == "row_serial"
    }
    for row in historical_summaries:
        baseline = serial_by_shape.get(int(row["sequence_length"]))
        compatible = baseline is not None and all(
            row[field] == baseline[field]
            for field in [
                "sequence_length", "rows", "columns", "dtype", "warmups",
                "iterations", "gpu_name", "compute_capability",
                "pytorch_version", "cuda_version",
            ]
        )
        row["speedup_vs_row_serial"] = (
            float(baseline["median_us"]) / float(row["median_us"])
            if compatible
            else "SPEEDUP NOT COMPUTED: incompatible experimental conditions."
        )
    write_csv(
        historical_summary_path,
        historical_summaries,
        list(historical_summaries[0]),
    )
    STATE["historical_benchmarks"] = "COMPLETE"
else:
    historical_summaries = []
    STATE["historical_benchmarks"] = "INCOMPLETE"
'''
            ),
        ]
    )

    cells.extend(
        [
            markdown(
                r"""
# 11. End-to-end transformer attention benchmark

**What this section does:** Uses identical Q/K/V tensors to measure explicit
PyTorch attention, explicit attention with only softmax replaced by the custom
CUDA operator, and PyTorch `scaled_dot_product_attention` with causal semantics.

**Why necessary:** A fast softmax kernel changes only one portion of attention;
both matrix multiplications remain. End-to-end speedup can therefore be much
smaller than kernel speedup.

**What to expect:** Untimed output comparisons at the fixed tolerances followed
by raw CUDA-event samples and summaries. A shape that fails correctness is not
timed and its failure is preserved.

**What to save:** `attention_correctness.json`, `attention_raw.csv`, and
`attention_summary.csv`.

**Paper meaning:** These artifacts support the transformer-attention experiment
and the distinction between microkernel and application performance.
"""
            ),
            code(
                r'''
import torch.nn.functional as F

ATTENTION_RAW_FIELDS = [
    "implementation", "git_commit", "batch", "heads", "sequence_length",
    "head_dimension", "dtype", "warmups", "iterations", "sample_index",
    "latency_us", "gpu_name", "compute_capability", "pytorch_version",
    "cuda_version", "timestamp",
]

def cuda_event_samples(operation: Callable[[], Any]) -> list[float]:
    for _ in range(WARMUPS):
        operation()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(ITERATIONS):
        start.record()
        operation()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)) * 1000.0)
    return samples

def make_attention_operations(
    sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Callable[[], torch.Tensor]]]:
    generator = torch.Generator(device="cuda").manual_seed(
        RANDOM_SEED + sequence_length
    )
    shape = (BATCH, HEADS, sequence_length, HEAD_DIM)
    query = torch.randn(shape, generator=generator, device="cuda", dtype=torch.float32)
    key = torch.randn(shape, generator=generator, device="cuda", dtype=torch.float32)
    value = torch.randn(shape, generator=generator, device="cuda", dtype=torch.float32)
    allowed = torch.ones(
        sequence_length, sequence_length, dtype=torch.bool, device="cuda"
    ).tril()
    scale = 1.0 / math.sqrt(HEAD_DIM)

    def explicit_eager() -> torch.Tensor:
        scores = query @ key.transpose(-2, -1)
        probabilities = torch.softmax(
            (scores * scale).masked_fill(~allowed, -torch.inf), dim=-1
        )
        return probabilities @ value

    def custom_softmax_attention() -> torch.Tensor:
        scores = query @ key.transpose(-2, -1)
        flattened = scores.reshape(-1, sequence_length).contiguous()
        probabilities = fused_causal_softmax(
            flattened, scale, block_size=SELECTED_BLOCK_SIZE
        ).reshape(BATCH, HEADS, sequence_length, sequence_length)
        return probabilities @ value

    def pytorch_sdpa() -> torch.Tensor:
        return F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=True
        )

    return query, key, value, {
        "explicit_pytorch": explicit_eager,
        "custom_softmax_attention": custom_softmax_attention,
        "pytorch_sdpa": pytorch_sdpa,
    }

attention_raw_path = ARTIFACTS / "attention/attention_raw.csv"
attention_summary_path = ARTIFACTS / "attention/attention_summary.csv"
attention_correctness_path = ARTIFACTS / "attention/attention_correctness.json"

if RUN_ATTENTION_BENCHMARKS and (
    not attention_raw_path.exists() or ARTIFACT_POLICY != "REUSE"
):
    if not CORRECTNESS_READY:
        raise RuntimeError("Attention benchmarking requires custom-softmax correctness")
    if not claim_output(attention_raw_path):
        attention_raw_rows = read_csv(attention_raw_path)
        attention_correctness = json.loads(attention_correctness_path.read_text())
    else:
        if attention_correctness_path.exists() and ARTIFACT_POLICY == "ERROR":
            raise FileExistsError(
                f"Refusing to overwrite {attention_correctness_path}"
            )
        attention_raw_rows: list[dict[str, Any]] = []
        attention_correctness: list[dict[str, Any]] = []
        for sequence_length in SEQUENCE_LENGTHS:
            try:
                _, _, _, operations = make_attention_operations(sequence_length)
                eager_output = operations["explicit_pytorch"]()
                custom_output = operations["custom_softmax_attention"]()
                sdpa_output = operations["pytorch_sdpa"]()
                torch.cuda.synchronize()
                custom_difference = (custom_output - eager_output).abs().max().item()
                sdpa_difference = (sdpa_output - eager_output).abs().max().item()
                try:
                    torch.testing.assert_close(
                        custom_output, eager_output, rtol=RTOL, atol=ATOL
                    )
                    custom_close = True
                except AssertionError:
                    custom_close = False
                try:
                    torch.testing.assert_close(
                        sdpa_output, eager_output, rtol=RTOL, atol=ATOL
                    )
                    sdpa_close = True
                except AssertionError:
                    sdpa_close = False
                passed = custom_close and sdpa_close
                attention_correctness.append(
                    {
                        "sequence_length": sequence_length,
                        "custom_max_absolute_error_vs_eager": custom_difference,
                        "sdpa_max_absolute_error_vs_eager": sdpa_difference,
                        "rtol": RTOL,
                        "atol": ATOL,
                        "pass_fail": "PASS" if passed else "FAIL",
                    }
                )
                checkpoint_json(attention_correctness_path, attention_correctness)
                if not passed:
                    STATE["failed_experiments"].append(
                        f"attention correctness S={sequence_length}"
                    )
                    print(f"Skipping attention timing for failed S={sequence_length}")
                    continue
                for implementation, operation in operations.items():
                    samples = cuda_event_samples(operation)
                    timestamp = utc_now()
                    for sample_index, latency_us in enumerate(samples):
                        attention_raw_rows.append(
                            {
                                "implementation": implementation,
                                "git_commit": GIT_COMMIT,
                                "batch": BATCH,
                                "heads": HEADS,
                                "sequence_length": sequence_length,
                                "head_dimension": HEAD_DIM,
                                "dtype": DTYPE,
                                "warmups": WARMUPS,
                                "iterations": ITERATIONS,
                                "sample_index": sample_index,
                                "latency_us": latency_us,
                                "gpu_name": ENVIRONMENT["gpu_name"],
                                "compute_capability": ENVIRONMENT["compute_capability"],
                                "pytorch_version": ENVIRONMENT["pytorch_version"],
                                "cuda_version": ENVIRONMENT["pytorch_cuda_version"],
                                "timestamp": timestamp,
                            }
                        )
                with attention_raw_path.open(
                    "w", newline="", encoding="utf-8"
                ) as output_file:
                    writer = csv.DictWriter(output_file, fieldnames=ATTENTION_RAW_FIELDS)
                    writer.writeheader()
                    writer.writerows(attention_raw_rows)
            except (RuntimeError, torch.cuda.OutOfMemoryError) as error:
                attention_correctness.append(
                    {
                        "sequence_length": sequence_length,
                        "pass_fail": "FAIL",
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                checkpoint_json(attention_correctness_path, attention_correctness)
                STATE["failed_experiments"].append(
                    f"attention S={sequence_length}: {type(error).__name__}"
                )
            finally:
                torch.cuda.empty_cache()
        register_file(attention_raw_path)
        register_file(attention_correctness_path)
elif attention_raw_path.exists():
    attention_raw_rows = read_csv(attention_raw_path)
    attention_correctness = json.loads(attention_correctness_path.read_text())
else:
    attention_raw_rows = []
    attention_correctness = []
    STATE["skipped_experiments"].append("attention benchmarks")

def summarize_attention(
    raw_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[float]] = {}
    representative: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in raw_rows:
        key = (str(row["implementation"]), int(row["sequence_length"]))
        grouped.setdefault(key, []).append(float(row["latency_us"]))
        representative[key] = row
    summaries = []
    for key, samples in grouped.items():
        implementation, sequence_length = key
        source = representative[key]
        summaries.append(
            {
                "implementation": implementation,
                "git_commit": source["git_commit"],
                "batch": int(source["batch"]),
                "heads": int(source["heads"]),
                "sequence_length": sequence_length,
                "head_dimension": int(source["head_dimension"]),
                "dtype": source["dtype"],
                "warmups": int(source["warmups"]),
                "iterations": int(source["iterations"]),
                "median_us": statistics.median(samples),
                "p25_us": percentile(samples, 0.25),
                "p75_us": percentile(samples, 0.75),
                "speedup_vs_explicit_eager": "",
                "speedup_vs_sdpa": "",
                "gpu_name": source["gpu_name"],
                "compute_capability": source["compute_capability"],
                "pytorch_version": source["pytorch_version"],
                "cuda_version": source["cuda_version"],
                "timestamp": source["timestamp"],
            }
        )
    by_shape = {
        sequence_length: {
            row["implementation"]: row
            for row in summaries if row["sequence_length"] == sequence_length
        }
        for sequence_length in {row["sequence_length"] for row in summaries}
    }
    for row in summaries:
        peers = by_shape[row["sequence_length"]]
        eager = peers.get("explicit_pytorch")
        sdpa = peers.get("pytorch_sdpa")
        controls = [
            "git_commit", "batch", "heads", "sequence_length", "head_dimension",
            "dtype", "warmups", "iterations", "gpu_name", "compute_capability",
            "pytorch_version", "cuda_version",
        ]
        row["speedup_vs_explicit_eager"] = (
            float(eager["median_us"]) / float(row["median_us"])
            if eager and all(row[field] == eager[field] for field in controls)
            else "SPEEDUP NOT COMPUTED: incompatible experimental conditions."
        )
        row["speedup_vs_sdpa"] = (
            float(sdpa["median_us"]) / float(row["median_us"])
            if sdpa and all(row[field] == sdpa[field] for field in controls)
            else "SPEEDUP NOT COMPUTED: incompatible experimental conditions."
        )
    return sorted(summaries, key=lambda row: (row["sequence_length"], row["implementation"]))

if attention_raw_rows:
    attention_summaries = summarize_attention(attention_raw_rows)
    write_csv(
        attention_summary_path,
        attention_summaries,
        list(attention_summaries[0]),
    )
    STATE["attention_benchmarks"] = "COMPLETE"
else:
    attention_summaries = []
    STATE["attention_benchmarks"] = "INCOMPLETE"
'''
            ),
            markdown(
                r"""
# 12. Kernel versus attention and Amdahl's Law analysis

**What this section does:** Matches measured softmax and attention rows by shape
and environment, computes measured speedups, and separately computes an Amdahl
model prediction when the measured fraction is valid.

**Why necessary:** Amdahl's Law explains why optimizing one component cannot
automatically accelerate unchanged QKᵀ and PV work by the same factor.

**What to expect:** A CSV distinguishing `MEASURED` quantities from
`MODEL_PREDICTION`. Invalid or incompatible rows report that computation was not
performed.

**What to save:** `artifacts/attention/amdahl_analysis.csv`.

**Paper meaning:** Model predictions must never be described as measured
end-to-end outcomes.
"""
            ),
            code(
                r'''
amdahl_path = ARTIFACTS / "attention/amdahl_analysis.csv"
amdahl_rows: list[dict[str, Any]] = []

if softmax_summaries and attention_summaries:
    softmax_by_shape = {
        int(row["sequence_length"]): row
        for row in softmax_summaries if row["implementation"] == "pytorch_eager"
    }
    custom_softmax_by_shape = {
        int(row["sequence_length"]): row
        for row in softmax_summaries if row["implementation"] == "custom_cuda"
    }
    attention_eager_by_shape = {
        int(row["sequence_length"]): row
        for row in attention_summaries if row["implementation"] == "explicit_pytorch"
    }
    attention_custom_by_shape = {
        int(row["sequence_length"]): row
        for row in attention_summaries if row["implementation"] == "custom_softmax_attention"
    }
    for sequence_length in sorted(
        set(softmax_by_shape)
        & set(custom_softmax_by_shape)
        & set(attention_eager_by_shape)
        & set(attention_custom_by_shape)
    ):
        softmax_eager = softmax_by_shape[sequence_length]
        softmax_custom = custom_softmax_by_shape[sequence_length]
        attention_eager = attention_eager_by_shape[sequence_length]
        attention_custom = attention_custom_by_shape[sequence_length]
        same_environment = len(
            {
                (
                    row["gpu_name"], row["compute_capability"],
                    row["pytorch_version"], row["cuda_version"],
                )
                for row in [
                    softmax_eager, softmax_custom, attention_eager, attention_custom
                ]
            }
        ) == 1
        if not same_environment:
            amdahl_rows.append(
                {
                    "sequence_length": sequence_length,
                    "status": "SPEEDUP NOT COMPUTED: incompatible experimental conditions.",
                }
            )
            continue
        softmax_speedup = float(softmax_eager["median_us"]) / float(
            softmax_custom["median_us"]
        )
        measured_attention_speedup = float(attention_eager["median_us"]) / float(
            attention_custom["median_us"]
        )
        measured_fraction = float(softmax_eager["median_us"]) / float(
            attention_eager["median_us"]
        )
        valid_model = 0.0 <= measured_fraction <= 1.0 and softmax_speedup > 0.0
        predicted = (
            1.0 / ((1.0 - measured_fraction) + measured_fraction / softmax_speedup)
            if valid_model else "MODEL NOT COMPUTED: invalid measured fraction."
        )
        amdahl_rows.append(
            {
                "sequence_length": sequence_length,
                "measured_softmax_speedup": softmax_speedup,
                "measured_attention_speedup": measured_attention_speedup,
                "measured_softmax_fraction_of_explicit_attention": measured_fraction,
                "amdahl_model_prediction": predicted,
                "softmax_values_label": "MEASURED",
                "attention_values_label": "MEASURED",
                "amdahl_value_label": "MODEL_PREDICTION" if valid_model else "UNAVAILABLE",
                "git_commit": GIT_COMMIT,
                "gpu_name": ENVIRONMENT["gpu_name"],
                "timestamp": utc_now(),
                "status": "COMPLETE" if valid_model else "INCOMPLETE",
            }
        )

if amdahl_rows:
    fieldnames = sorted({key for row in amdahl_rows for key in row})
    write_csv(amdahl_path, amdahl_rows, fieldnames)
else:
    STATE["skipped_experiments"].append("Amdahl analysis: matched data absent")
'''
            ),
            markdown(
                r"""
# 13. PyTorch Profiler

**What this section does:** Profiles one configurable shape with labeled eager,
custom-softmax, and SDPA regions; exports a Chrome trace, text table, and event
CSV without turning profiler facts into causal explanations.

**Why necessary:** Timing says how long; profiling helps identify where time is
reported and which kernels/operators are present.

**What to expect:** Profiler artifacts when CUDA profiling is supported. A
failure is logged without deleting completed benchmark data.

**What to save:** Everything under `artifacts/profiler/`.

**Paper meaning:** Use `MEASURED / INTERPRETATION / NEXT EXPERIMENT`; kernel names
and times are measurements, while bottleneck explanations remain hypotheses.
"""
            ),
            code(
                r'''
profiler_table_path = ARTIFACTS / "profiler/pytorch_profiler_table.txt"
profiler_trace_path = ARTIFACTS / "profiler/pytorch_profiler_trace.json"
profiler_events_path = ARTIFACTS / "profiler/pytorch_profiler_events.csv"
profiler_notes_path = ARTIFACTS / "profiler/pytorch_profiler_interpretation.md"

if RUN_PYTORCH_PROFILER:
    try:
        _, _, _, profile_operations = make_attention_operations(
            PROFILE_SEQUENCE_LENGTH
        )
        for operation in profile_operations.values():
            for _ in range(3):
                operation()
        torch.cuda.synchronize()
        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as profile:
            for label, operation in profile_operations.items():
                with torch.profiler.record_function(f"attention::{label}"):
                    operation()
            torch.cuda.synchronize()

        averages = profile.key_averages()
        try:
            table = averages.table(sort_by="self_device_time_total", row_limit=100)
        except KeyError:
            table = averages.table(sort_by="self_cuda_time_total", row_limit=100)
        write_text(profiler_table_path, table + "\n")
        if claim_output(profiler_trace_path):
            profile.export_chrome_trace(str(profiler_trace_path))
        register_file(profiler_trace_path)

        profiler_rows = []
        for event in averages:
            device_total = getattr(
                event, "self_device_time_total",
                getattr(event, "self_cuda_time_total", 0.0),
            )
            profiler_rows.append(
                {
                    "operator": event.key,
                    "count": event.count,
                    "self_cpu_time_total_us": event.self_cpu_time_total,
                    "cpu_time_total_us": event.cpu_time_total,
                    "self_device_time_total_us": device_total,
                }
            )
        write_csv(
            profiler_events_path,
            profiler_rows,
            [
                "operator", "count", "self_cpu_time_total_us",
                "cpu_time_total_us", "self_device_time_total_us",
            ],
        )
        write_text(
            profiler_notes_path,
            "# PyTorch Profiler notes\n\n"
            "## MEASURED\n\n"
            "Profiler table, event CSV, and trace were exported for the configured "
            f"S={PROFILE_SEQUENCE_LENGTH} workload. Read the artifacts before adding facts.\n\n"
            "## INTERPRETATION\n\nTODO(student): Add a hypothesis, not a profiler fact.\n\n"
            "## NEXT EXPERIMENT\n\nTODO(student): State a controlled follow-up.\n",
        )
        STATE["pytorch_profiler"] = "COMPLETE"
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
        write_text(ARTIFACTS / "profiler/pytorch_profiler_failure.txt", failure + "\n")
        STATE["failed_experiments"].append(f"PyTorch Profiler: {failure}")
        STATE["pytorch_profiler"] = "INCOMPLETE"
else:
    STATE["skipped_experiments"].append("PyTorch Profiler")
    STATE["pytorch_profiler"] = "INCOMPLETE"
'''
            ),
            markdown(
                r"""
# 14. Optional NVIDIA Nsight Compute attempt

**What this section does:** Detects the installed `ncu` version and, if present,
attempts a narrowly filtered `--set basic` profile of the custom kernel. The
installed tool—not this notebook—chooses version-valid basic metrics.

**Why necessary:** Nsight can report launch/resource/hardware-counter evidence
that source inspection cannot establish. Colab often restricts performance
counters.

**What to expect:** Either a report/export or the explicit marker
`NSIGHT COMPUTE UNAVAILABLE IN THIS COLAB ENVIRONMENT` with the failure log.

**What to save:** `artifacts/profiler/nsight/`.

**Paper meaning:** Absence of permission is a limitation, not a measured kernel
property. Interpretations require a separate controlled follow-up.
"""
            ),
            code(
                r'''
nsight_directory = ARTIFACTS / "profiler/nsight"
nsight_status: dict[str, Any] = {"timestamp": utc_now(), "status": "UNAVAILABLE"}
ncu_path = shutil.which("ncu")

if RUN_NSIGHT and ncu_path:
    version = run_command([ncu_path, "--version"], check=False)
    nsight_status["version_output"] = version.stdout
    target_path = nsight_directory / "profile_target.py"
    target_source = f"""
import math
import sys
import torch
sys.path.insert(0, {str(work_dir)!r})
from cuda_attention.operator import fused_causal_softmax
scores = torch.randn({BATCH_HEADS * PROFILE_SEQUENCE_LENGTH}, {PROFILE_SEQUENCE_LENGTH}, device="cuda", dtype=torch.float32)
scale = 1.0 / math.sqrt({HEAD_DIM})
for _ in range(3):
    fused_causal_softmax(scores, scale, block_size={SELECTED_BLOCK_SIZE})
torch.cuda.synchronize()
fused_causal_softmax(scores, scale, block_size={SELECTED_BLOCK_SIZE})
torch.cuda.synchronize()
""".strip() + "\n"
    write_text(target_path, target_source)
    report_base = nsight_directory / "fused_causal_softmax"
    report_path = report_base.with_suffix(".ncu-rep")
    if report_path.exists() and ARTIFACT_POLICY == "ERROR":
        raise FileExistsError(f"Refusing to overwrite {report_path}")
    command = [
        ncu_path, "--set", "basic", "--target-processes", "all",
        "--kernel-name", "regex:.*fused_causal_softmax_kernel.*",
        "--export", str(report_base), sys.executable, str(target_path),
    ]
    result = run_command(
        command, cwd=work_dir,
        log_path=nsight_directory / "ncu_run_log.txt", check=False,
    )
    nsight_status["return_code"] = result.returncode
    if result.returncode == 0 and report_path.is_file():
        register_file(report_path)
        export_result = run_command(
            [ncu_path, "--import", str(report_path), "--csv", "--page", "raw"],
            cwd=work_dir,
            log_path=nsight_directory / "ncu_raw_export.csv",
            check=False,
        )
        nsight_status["export_return_code"] = export_result.returncode
        nsight_status["status"] = (
            "COMPLETE" if export_result.returncode == 0 else "PARTIAL"
        )
    else:
        marker = (
            "NSIGHT COMPUTE UNAVAILABLE IN THIS COLAB ENVIRONMENT\n\n"
            "The tool was present, but profiling failed. Inspect ncu_run_log.txt; "
            "hardware-counter permissions are a common Colab limitation.\n"
        )
        write_text(nsight_directory / "UNAVAILABLE.txt", marker)
        STATE["failed_experiments"].append("Nsight Compute execution")
elif RUN_NSIGHT:
    write_text(
        nsight_directory / "UNAVAILABLE.txt",
        "NSIGHT COMPUTE UNAVAILABLE IN THIS COLAB ENVIRONMENT\n"
        "The ncu executable was not installed.\n",
    )
    STATE["skipped_experiments"].append("Nsight Compute: ncu not installed")
else:
    write_text(nsight_directory / "UNAVAILABLE.txt", "Nsight Compute disabled by configuration.\n")
    STATE["skipped_experiments"].append("Nsight Compute disabled")

write_json(nsight_directory / "nsight_status.json", nsight_status)
STATE["nsight_compute"] = nsight_status["status"]
'''
            ),
        ]
    )

    cells.extend(
        [
            markdown(
                r"""
# 15. Generate publication figures from saved measurements

**What this section does:** Reads only measured summaries/profiler CSVs and
creates PDF plus 300-DPI PNG figures for softmax latency, throughput, historical
speedup, launch tuning, kernel-versus-attention speedup, and profiler breakdown.

**Why necessary:** Figures must be reproducible views of stored evidence rather
than manually typed numbers.

**What to expect:** Up to six figure pairs. If required data are absent, the
notebook writes a `*_MISSING.txt` explanation and creates no invented chart.

**What to save:** `artifacts/figures/`.

**Paper meaning:** Every plotted point must be traceable to a CSV row and commit.
"""
            ),
            code(
                r'''
import matplotlib.pyplot as plt

FIGURE_SPECS = {
    "softmax_latency": ("fig_softmax_latency.pdf", "fig_softmax_latency.png"),
    "softmax_throughput": ("fig_softmax_throughput.pdf", "fig_softmax_throughput.png"),
    "historical_speedup": ("fig_historical_speedup.pdf", "fig_historical_speedup.png"),
    "launch_configuration": ("fig_launch_configuration.pdf", "fig_launch_configuration.png"),
    "kernel_vs_attention_speedup": (
        "fig_kernel_vs_attention_speedup.pdf", "fig_kernel_vs_attention_speedup.png"
    ),
    "profiler_breakdown": ("fig_profiler_breakdown.pdf", "fig_profiler_breakdown.png"),
}

def save_figure(name: str, figure: plt.Figure) -> None:
    pdf_name, png_name = FIGURE_SPECS[name]
    pdf_path = ARTIFACTS / "figures" / pdf_name
    png_path = ARTIFACTS / "figures" / png_name
    existing = [path for path in (pdf_path, png_path) if path.exists()]
    if existing and ARTIFACT_POLICY == "ERROR":
        raise FileExistsError(f"Refusing to overwrite figure: {existing[0]}")
    if existing and ARTIFACT_POLICY == "REUSE":
        for path in (pdf_path, png_path):
            if path.exists():
                register_file(path)
        plt.close(figure)
        return
    figure.tight_layout()
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    register_file(pdf_path)
    register_file(png_path)
    plt.close(figure)

def grouped_summary(
    rows: Sequence[Mapping[str, Any]], key: str = "implementation"
) -> dict[str, list[Mapping[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row[key]), []).append(row)
    for values in groups.values():
        values.sort(key=lambda row: int(row["sequence_length"]))
    return groups

if GENERATE_FIGURES and softmax_summaries:
    groups = grouped_summary(softmax_summaries)
    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    for label, rows in groups.items():
        x = [int(row["sequence_length"]) for row in rows]
        median = [float(row["median_us"]) for row in rows]
        lower = [value - float(row["p25_us"]) for value, row in zip(median, rows)]
        upper = [float(row["p75_us"]) - value for value, row in zip(median, rows)]
        axis.errorbar(x, median, yerr=[lower, upper], marker="o", capsize=3, label=label)
    axis.set_xlabel("Sequence length")
    axis.set_ylabel("Median latency (µs)")
    axis.set_title("Causal scaled-softmax latency")
    axis.grid(True, alpha=0.3)
    axis.legend()
    save_figure("softmax_latency", figure)

    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    for label, rows in groups.items():
        axis.plot(
            [int(row["sequence_length"]) for row in rows],
            [float(row["elements_per_second"]) for row in rows],
            marker="o", label=label,
        )
    axis.set_xlabel("Sequence length")
    axis.set_ylabel("Elements per second")
    axis.set_title("Causal scaled-softmax throughput")
    axis.grid(True, alpha=0.3)
    axis.legend()
    save_figure("softmax_throughput", figure)
else:
    write_text(
        ARTIFACTS / "figures/softmax_figures_MISSING.txt",
        "Softmax figures were not generated because measured softmax summaries are absent.\n",
    )

if GENERATE_FIGURES and historical_summaries and len(
    {row["implementation"] for row in historical_summaries}
) >= 2:
    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    for label, rows in grouped_summary(historical_summaries).items():
        usable = [
            row for row in rows
            if isinstance(row.get("speedup_vs_row_serial"), (int, float))
        ]
        if usable:
            axis.plot(
                [int(row["sequence_length"]) for row in usable],
                [float(row["speedup_vs_row_serial"]) for row in usable],
                marker="o", label=label,
            )
    axis.axhline(1.0, color="black", linewidth=1, linestyle="--")
    axis.set_xlabel("Sequence length")
    axis.set_ylabel("Speedup vs one-thread-per-row")
    axis.set_title("CUDA implementation evolution")
    axis.grid(True, alpha=0.3)
    axis.legend()
    save_figure("historical_speedup", figure)
else:
    write_text(
        ARTIFACTS / "figures/historical_speedup_MISSING.txt",
        "Historical speedup figure requires at least two verified measured commits.\n",
    )

if GENERATE_FIGURES and launch_summaries:
    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    launch_groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in launch_summaries:
        launch_groups.setdefault(f"{row['launch_block_size']} threads", []).append(row)
    for label, rows in launch_groups.items():
        rows.sort(key=lambda row: int(row["sequence_length"]))
        axis.plot(
            [int(row["sequence_length"]) for row in rows],
            [float(row["median_us"]) for row in rows],
            marker="o", label=label,
        )
    axis.set_xlabel("Sequence length")
    axis.set_ylabel("Median latency (µs)")
    axis.set_title("Launch-configuration comparison")
    axis.grid(True, alpha=0.3)
    axis.legend()
    save_figure("launch_configuration", figure)
else:
    write_text(
        ARTIFACTS / "figures/launch_configuration_MISSING.txt",
        "Launch figure requires complete measured launch summaries.\n",
    )

complete_amdahl_rows = [
    row for row in amdahl_rows if row.get("status") == "COMPLETE"
]
if GENERATE_FIGURES and complete_amdahl_rows:
    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    x = [int(row["sequence_length"]) for row in complete_amdahl_rows]
    axis.plot(
        x,
        [float(row["measured_softmax_speedup"]) for row in complete_amdahl_rows],
        marker="o", label="Measured softmax speedup",
    )
    axis.plot(
        x,
        [float(row["measured_attention_speedup"]) for row in complete_amdahl_rows],
        marker="s", label="Measured attention speedup",
    )
    axis.set_xlabel("Sequence length")
    axis.set_ylabel("Speedup vs explicit PyTorch")
    axis.set_title("Kernel versus end-to-end attention speedup")
    axis.grid(True, alpha=0.3)
    axis.legend()
    save_figure("kernel_vs_attention_speedup", figure)
else:
    write_text(
        ARTIFACTS / "figures/kernel_vs_attention_MISSING.txt",
        "Kernel-versus-attention figure requires matched measured summaries.\n",
    )

if GENERATE_FIGURES and profiler_events_path.is_file():
    profiler_plot_rows = [
        row for row in read_csv(profiler_events_path)
        if float(row["self_device_time_total_us"]) > 0.0
    ]
    profiler_plot_rows.sort(
        key=lambda row: float(row["self_device_time_total_us"]), reverse=True
    )
    profiler_plot_rows = profiler_plot_rows[:12]
    if profiler_plot_rows:
        figure, axis = plt.subplots(figsize=(8.0, 5.5))
        labels = [row["operator"] for row in reversed(profiler_plot_rows)]
        values = [float(row["self_device_time_total_us"]) for row in reversed(profiler_plot_rows)]
        axis.barh(labels, values)
        axis.set_xlabel("Self device time (µs)")
        axis.set_title("PyTorch Profiler device-time breakdown")
        save_figure("profiler_breakdown", figure)
    else:
        write_text(
            ARTIFACTS / "figures/profiler_breakdown_MISSING.txt",
            "Profiler event CSV contains no positive device-time rows.\n",
        )
else:
    write_text(
        ARTIFACTS / "figures/profiler_breakdown_MISSING.txt",
        "Profiler breakdown requires a measured profiler event CSV.\n",
    )
'''
            ),
            markdown(
                r"""
# 16. Export paper tables and evaluate hypotheses

**What this section does:** Produces correctness, softmax, launch, attention,
and environment tables in CSV and LaTeX-friendly fragments. It evaluates H1–H5
only when the required measurements exist and always separates measurement from
interpretation.

**Why necessary:** Paper numbers should be generated from artifacts, not copied
by hand. Hypotheses require explicit evidence and may remain untested or
inconclusive.

**What to expect:** Table files and `hypothesis_evaluation.md`. No synthetic rows
are inserted to complete missing tables.

**What to save:** `artifacts/tables/` and the hypothesis file.

**Paper meaning:** These files are the direct inputs for Results, Discussion,
and the hypothesis-status narrative.
"""
            ),
            code(
                r'''
def latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%",
        "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
    }
    for source, replacement in replacements.items():
        text = text.replace(source, replacement)
    return text

def write_latex_table(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    if not rows:
        return
    alignment = "l" * len(columns)
    lines = [
        f"\\begin{{tabular}}{{{alignment}}}",
        "\\hline",
        " & ".join(latex_escape(column) for column in columns) + r" \\",
        "\\hline",
    ]
    for row in rows:
        lines.append(" & ".join(latex_escape(row.get(column, "")) for column in columns) + r" \\")
    lines.extend(["\\hline", "\\end{tabular}"])
    write_text(path, "\n".join(lines) + "\n")

correctness_table_columns = [
    "git_commit", "sequence_length", "input_family", "max_absolute_error",
    "max_relative_error", "max_row_sum_error", "contains_nan", "contains_inf",
    "pass_fail",
]
if correctness_rows:
    correctness_table = [
        {column: row.get(column, "") for column in correctness_table_columns}
        for row in correctness_rows
    ]
    write_csv(
        ARTIFACTS / "tables/correctness_table.csv",
        correctness_table, correctness_table_columns,
    )
    write_latex_table(
        ARTIFACTS / "tables/correctness_table.tex",
        correctness_table, correctness_table_columns,
    )

softmax_table_columns = [
    "implementation", "git_commit", "sequence_length", "median_us", "p25_us",
    "p75_us", "elements_per_second", "speedup_vs_eager",
]
if softmax_summaries:
    softmax_table = [
        {column: row.get(column, "") for column in softmax_table_columns}
        for row in softmax_summaries
    ]
    write_csv(
        ARTIFACTS / "tables/softmax_performance_table.csv",
        softmax_table, softmax_table_columns,
    )
    write_latex_table(
        ARTIFACTS / "tables/softmax_performance_table.tex",
        softmax_table, softmax_table_columns,
    )

if launch_summaries:
    launch_by_shape: dict[int, dict[int, float]] = {}
    for row in launch_summaries:
        launch_by_shape.setdefault(int(row["sequence_length"]), {})[
            int(row["launch_block_size"])
        ] = float(row["median_us"])
    launch_table = [
        {
            "sequence_length": sequence_length,
            "128_threads_median_us": values.get(128, ""),
            "256_threads_median_us": values.get(256, ""),
            "512_threads_median_us": values.get(512, ""),
            "selected_configuration": SELECTED_BLOCK_SIZE,
        }
        for sequence_length, values in sorted(launch_by_shape.items())
    ]
    launch_columns = list(launch_table[0])
    write_csv(
        ARTIFACTS / "tables/launch_configuration_table.csv",
        launch_table, launch_columns,
    )
    write_latex_table(
        ARTIFACTS / "tables/launch_configuration_table.tex",
        launch_table, launch_columns,
    )

if attention_summaries:
    attention_table_columns = [
        "implementation", "git_commit", "sequence_length", "median_us", "p25_us",
        "p75_us", "speedup_vs_explicit_eager", "speedup_vs_sdpa",
    ]
    attention_table = [
        {column: row.get(column, "") for column in attention_table_columns}
        for row in attention_summaries
    ]
    write_csv(
        ARTIFACTS / "tables/attention_table.csv",
        attention_table, attention_table_columns,
    )
    write_latex_table(
        ARTIFACTS / "tables/attention_table.tex",
        attention_table, attention_table_columns,
    )

environment_table = [
    {"component": key, "value": value, "evidence_source": "environment.json"}
    for key, value in ENVIRONMENT.items()
    if key not in {"nvidia_smi_output", "nvcc_output"}
]
write_csv(
    ARTIFACTS / "tables/environment_table.csv",
    environment_table, ["component", "value", "evidence_source"],
)
write_latex_table(
    ARTIFACTS / "tables/environment_table.tex",
    environment_table, ["component", "value", "evidence_source"],
)

def hypothesis_entry(
    identifier: str,
    hypothesis: str,
    measured: str,
    interpretation: str,
    status: str,
    next_experiment: str,
) -> str:
    return textwrap.dedent(
        f"""
        ## {identifier}

        HYPOTHESIS:
        {hypothesis}

        MEASURED:
        {measured}

        INTERPRETATION:
        {interpretation}

        STATUS:
        {status}

        NEXT EXPERIMENT:
        {next_experiment}
        """
    ).strip()

hypotheses = []
historical_by_stage = grouped_summary(historical_summaries) if historical_summaries else {}
serial_rows = historical_by_stage.get("row_serial", [])
block_rows = historical_by_stage.get("block_shared_tree", [])
warp_rows = historical_by_stage.get("warp_reduction", [])

if len(serial_rows) >= 2 and len(block_rows) >= 2:
    serial_growth = float(serial_rows[-1]["median_us"]) / float(serial_rows[0]["median_us"])
    block_growth = float(block_rows[-1]["median_us"]) / float(block_rows[0]["median_us"])
    h1_status = "PARTIALLY SUPPORTED" if serial_growth > block_growth else "NOT SUPPORTED"
    h1_measured = (
        f"Measured endpoint latency growth: row-serial={serial_growth:.6g}x, "
        f"block={block_growth:.6g}x."
    )
else:
    h1_status = "UNTESTED"
    h1_measured = "Compatible row-serial and block scaling measurements are absent."
hypotheses.append(hypothesis_entry(
    "H1",
    "One-thread-per-row scales poorly as row work remains serial.",
    h1_measured,
    "Growth comparison can support the observed trend but cannot alone prove the causal mechanism.",
    h1_status,
    "Use profiler instruction/memory evidence on matched early and late lengths.",
))

def later_shape_speedups(candidate_rows: Sequence[Mapping[str, Any]]) -> list[float]:
    return [
        float(row["speedup_vs_row_serial"])
        for row in candidate_rows[-3:]
        if isinstance(row.get("speedup_vs_row_serial"), (int, float))
    ]

block_late = later_shape_speedups(block_rows)
h2_status = (
    "SUPPORTED" if block_late and all(value > 1.0 for value in block_late)
    else "PARTIALLY SUPPORTED" if any(value > 1.0 for value in block_late)
    else "NOT SUPPORTED" if block_late else "UNTESTED"
)
hypotheses.append(hypothesis_entry(
    "H2",
    "One-block-per-row reduces latency at sufficiently long rows despite coordination overhead.",
    f"Measured long-shape speedups vs row serial: {block_late}" if block_late else "Required historical rows are absent.",
    "Status describes only tested lengths and this GPU.", h2_status,
    "Repeat on another NVIDIA architecture and inspect short-row crossover.",
))

if warp_rows and block_rows:
    block_map = {int(row["sequence_length"]): float(row["median_us"]) for row in block_rows}
    warp_vs_block = [
        block_map[int(row["sequence_length"])] / float(row["median_us"])
        for row in warp_rows if int(row["sequence_length"]) in block_map
    ]
else:
    warp_vs_block = []
h3_status = (
    "SUPPORTED" if warp_vs_block and all(value > 1.0 for value in warp_vs_block)
    else "PARTIALLY SUPPORTED" if any(value > 1.0 for value in warp_vs_block)
    else "NOT SUPPORTED" if warp_vs_block else "UNTESTED"
)
hypotheses.append(hypothesis_entry(
    "H3",
    "Warp reductions reduce latency relative to full shared-memory trees.",
    f"Measured block/warp speedups by matched shape: {warp_vs_block}" if warp_vs_block else "Matched block and warp measurements are absent.",
    "Latency shows outcome; Nsight data is needed to attribute it to communication or synchronization.",
    h3_status, "Compare barrier/shared-memory metrics if Nsight permissions allow.",
))

if launch_summaries:
    winners = {
        min(values, key=values.get)
        for values in launch_by_shape.values() if len(values) == 3
    }
    h4_status = "SUPPORTED" if len(winners) > 1 else "NOT SUPPORTED"
    h4_measured = f"Per-shape winning block sizes: {sorted(winners)}."
else:
    h4_status = "UNTESTED"
    h4_measured = "Complete launch measurements are absent."
hypotheses.append(hypothesis_entry(
    "H4",
    "The best block size depends on sequence length and resource tradeoffs.",
    h4_measured,
    "A single observed winner does not establish universality beyond the tested matrix.",
    h4_status, "Repeat launch tuning on a second GPU architecture.",
))

if complete_amdahl_rows:
    comparisons = [
        float(row["measured_softmax_speedup"]) > float(row["measured_attention_speedup"])
        for row in complete_amdahl_rows
    ]
    h5_status = (
        "SUPPORTED" if all(comparisons)
        else "PARTIALLY SUPPORTED" if any(comparisons)
        else "NOT SUPPORTED"
    )
    h5_measured = f"Kernel speedup exceeded attention speedup for {sum(comparisons)}/{len(comparisons)} matched shapes."
else:
    h5_status = "UNTESTED"
    h5_measured = "Matched kernel and attention measurements are absent."
hypotheses.append(hypothesis_entry(
    "H5",
    "Softmax-kernel speedup exceeds complete-attention speedup.",
    h5_measured,
    "Unchanged matrix multiplications limit application-level benefit; measured results remain distinct from the Amdahl model.",
    h5_status, "Profile the fraction of attention time attributable to softmax by shape.",
))

write_text(
    ARTIFACTS / "metadata/hypothesis_evaluation.md",
    "# Hypothesis evaluation\n\n" + "\n\n".join(hypotheses) + "\n",
)
'''
            ),
            markdown(
                r"""
# 17. Automatic validation, manifest, and reproducibility README

**What this section does:** Recomputes statistics from raw samples, checks
schemas/provenance/correctness, inventories files, and creates the validation
report, experiment manifest, and reproduction guide.

**Why necessary:** A polished figure is not trustworthy unless its raw samples,
metadata, formulas, and correctness gate agree.

**What to expect:** Every audit item is labeled `PASS`, `WARNING`, or `FAIL`.
Critical failures must be resolved before paper claims are written.

**What to save:** `validation_report.md`, `experiment_manifest.json`, and
`README_EXPERIMENTS.md`.

**Paper meaning:** This is the final traceability and limitations check.
"""
            ),
            code(
                r'''
validation_items: list[tuple[str, str, str]] = []

def validation(name: str, status: str, detail: str) -> None:
    if status not in {"PASS", "WARNING", "FAIL"}:
        raise ValueError(status)
    validation_items.append((name, status, detail))

validation("CUDA environment", "PASS" if CUDA_READY else "FAIL", str(ENVIRONMENT["gpu_name"]))
validation("CUDA extension build", "PASS" if EXTENSION_READY else "FAIL", str(BUILD_METADATA))
validation("Custom operator import", "PASS" if EXTENSION_READY else "FAIL", "cuda_attention._C")
validation("CUDA correctness", "PASS" if CORRECTNESS_READY else "FAIL", str(correctness_summary))

if correctness_rows:
    no_nonfinite = all(
        str(row["contains_nan"]).lower() == "false"
        and str(row["contains_inf"]).lower() == "false"
        for row in correctness_rows
    )
    masks_zero = all(
        str(row["masked_positions_exactly_zero"]).lower() == "true"
        for row in correctness_rows
    )
    row_sums = all(float(row["max_row_sum_error"]) <= ATOL + RTOL for row in correctness_rows)
    validation("No unexpected NaNs/Infs", "PASS" if no_nonfinite else "FAIL", "correctness CSV")
    validation("Masked probabilities exactly zero", "PASS" if masks_zero else "FAIL", "correctness CSV")
    validation("Softmax row sums", "PASS" if row_sums else "FAIL", f"threshold={ATOL + RTOL}")
else:
    validation("Correctness invariants", "FAIL", "correctness CSV absent")

required_softmax_columns = set(RAW_SOFTMAX_FIELDS)
if softmax_raw_rows:
    raw_columns_ok = all(required_softmax_columns <= set(row) for row in softmax_raw_rows)
    raw_positive = all(float(row["latency_us"]) > 0 for row in softmax_raw_rows)
    hashes_present = all(str(row["git_commit"]) == GIT_COMMIT for row in softmax_raw_rows)
    validation("Softmax raw schema", "PASS" if raw_columns_ok else "FAIL", str(required_softmax_columns))
    validation("Softmax raw samples", "PASS" if raw_positive else "FAIL", f"rows={len(softmax_raw_rows)}")
    validation("Git hashes in benchmark rows", "PASS" if hashes_present else "FAIL", GIT_COMMIT)

    summary_lookup = {
        (str(row["implementation"]), int(row["sequence_length"])): row
        for row in softmax_summaries
    }
    recomputed_ok = True
    throughput_ok = True
    for key, summary in summary_lookup.items():
        samples = [
            float(row["latency_us"])
            for row in softmax_raw_rows
            if (str(row["implementation"]), int(row["sequence_length"])) == key
        ]
        recomputed_ok &= math.isclose(
            statistics.median(samples), float(summary["median_us"]), rel_tol=1e-12
        )
        recomputed_ok &= math.isclose(
            percentile(samples, 0.25), float(summary["p25_us"]), rel_tol=1e-12
        )
        recomputed_ok &= math.isclose(
            percentile(samples, 0.75), float(summary["p75_us"]), rel_tol=1e-12
        )
        expected_throughput = (
            int(summary["rows"]) * int(summary["columns"]) * 1_000_000.0
            / float(summary["median_us"])
        )
        throughput_ok &= math.isclose(
            expected_throughput,
            float(summary["elements_per_second"]),
            rel_tol=1e-12,
        )
    validation("Summary median/quartiles", "PASS" if recomputed_ok else "FAIL", "recomputed from raw")
    validation("Throughput formula", "PASS" if throughput_ok else "FAIL", "rows*columns/median_seconds")
    eager_by_shape = {
        int(row["sequence_length"]): row
        for row in softmax_summaries if row["implementation"] == "pytorch_eager"
    }
    speedup_ok = True
    for row in softmax_summaries:
        eager = eager_by_shape.get(int(row["sequence_length"]))
        recorded = row.get("speedup_vs_eager")
        if eager is None or not isinstance(recorded, (int, float)):
            speedup_ok = False
            continue
        expected_speedup = float(eager["median_us"]) / float(row["median_us"])
        speedup_ok &= math.isclose(expected_speedup, float(recorded), rel_tol=1e-12)
    validation("Matched softmax speedups", "PASS" if speedup_ok else "FAIL", "eager median / candidate median")
else:
    validation("Softmax benchmarks", "WARNING", "raw file absent or stage disabled")

validation(
    "Environment metadata",
    "PASS" if environment_path.is_file() and ENVIRONMENT.get("gpu_name") else "FAIL",
    str(environment_path),
)
figure_files = sorted((ARTIFACTS / "figures").glob("fig_*"))
validation(
    "Figures from measured data",
    "PASS" if figure_files else "WARNING",
    f"generated_files={len(figure_files)}; missing figures have text reports",
)
validation(
    "Git working tree provenance",
    "WARNING" if STATE["git_dirty"] else "PASS",
    GIT_STATUS or "clean",
)

validation_lines = ["# Validation report", ""]
for name, status, detail in validation_items:
    validation_lines.extend([f"## {name}: {status}", "", detail, ""])
write_text(
    ARTIFACTS / "validation_report.md",
    "\n".join(validation_lines).rstrip() + "\n",
)

all_artifact_files = sorted(
    path for path in ARTIFACTS.rglob("*") if path.is_file()
)
manifest = {
    "project_title": "CUDA Optimization of Fused Causal Softmax for Transformer Attention",
    "run_timestamp": RUN_TIMESTAMP,
    "manifest_timestamp": utc_now(),
    "git_commit": GIT_COMMIT,
    "git_dirty": STATE["git_dirty"],
    "gpu": ENVIRONMENT["gpu_name"],
    "compute_capability": ENVIRONMENT["compute_capability"],
    "cuda": ENVIRONMENT["pytorch_cuda_version"],
    "pytorch": ENVIRONMENT["pytorch_version"],
    "python": ENVIRONMENT["python_version"],
    "benchmark_configuration": {
        "warmups": WARMUPS, "iterations": ITERATIONS,
        "sequence_lengths": SEQUENCE_LENGTHS, "batch_heads": BATCH_HEADS,
        "head_dimension": HEAD_DIM, "dtype": DTYPE,
        "selected_block_size": SELECTED_BLOCK_SIZE,
        "random_seed": RANDOM_SEED,
    },
    "correctness_tolerance": {"rtol": RTOL, "atol": ATOL},
    "csv_files": [str(path.relative_to(ARTIFACTS)) for path in all_artifact_files if path.suffix == ".csv"],
    "figures": [str(path.relative_to(ARTIFACTS)) for path in all_artifact_files if path.suffix in {".png", ".pdf"}],
    "profiler_artifacts": [
        str(path.relative_to(ARTIFACTS))
        for path in all_artifact_files if "profiler" in path.parts
    ],
    "failed_experiments": STATE["failed_experiments"],
    "skipped_experiments": STATE["skipped_experiments"],
    "stage_status": {
        key: value for key, value in STATE.items()
        if key in {
            "cuda_environment", "cuda_extension_build", "cuda_correctness",
            "softmax_benchmarks", "historical_benchmarks", "launch_configuration",
            "attention_benchmarks", "pytorch_profiler", "nsight_compute",
        }
    },
}
write_json(ARTIFACTS / "experiment_manifest.json", manifest)

reproducibility_readme = f"""# CUDA softmax experiment artifacts

## Runtime environment

- Git commit: `{GIT_COMMIT}`
- Git dirty: `{STATE['git_dirty']}`
- GPU: `{ENVIRONMENT['gpu_name']}`
- Compute capability: `{ENVIRONMENT['compute_capability']}`
- PyTorch: `{ENVIRONMENT['pytorch_version']}`
- PyTorch CUDA: `{ENVIRONMENT['pytorch_cuda_version']}`
- Python: `{ENVIRONMENT['python_version']}`

## Build

The notebook invoked `bash scripts/build_extension.sh` from the repository root.
See `build/build_log.txt` and `build/build_metadata.json` for the complete result.

## Methodology

Inputs and masks were allocated outside timed regions. Each operation received
{WARMUPS} untimed warmups and {ITERATIONS} CUDA-event samples. The compiled
baseline used an additional untimed first invocation. CUDA synchronization made
device completion part of every sample. Correctness used rtol={RTOL}, atol={ATOL}.

## Output files

See `experiment_manifest.json` for the machine-readable inventory and stage
status. Raw samples are retained rather than replaced by summary values.

## Known limitations

Google Colab is a shared cloud environment. GPU model, clocks, contention,
runtime lifetime, Nsight availability, and performance-counter permissions are
not guaranteed. Results from different GPU environments must not be combined.

## Reproduction

Open `notebooks/04_colab_research_experiments.ipynb`, choose an NVIDIA GPU,
configure the same repository revision and controls, use a new artifact
directory, and run cells in order. Resolve validation failures before quoting
results.
"""
write_text(ARTIFACTS / "README_EXPERIMENTS.md", reproducibility_readme)
'''
            ),
            markdown(
                r"""
# 18. Inventory, optional Drive backup, and artifact ZIP

**What this section does:** Prints the complete artifact tree and stage summary,
optionally copies it to Google Drive, and packages every raw file/log into one
ZIP without excluding failures.

**Why necessary:** Colab `/content` storage is temporary, and the paper handoff
must include raw evidence rather than screenshots alone.

**What to expect:** `cuda_softmax_research_artifacts.zip` plus an optional Drive
folder. Existing destinations obey the configured overwrite policy.

**What to save:** The ZIP and optional Drive backup.

**Paper meaning:** This is the complete evidence handoff for claim verification.
"""
            ),
            code(
                r'''
def artifact_tree(root: Path) -> str:
    lines = [root.name + "/"]
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        indent = "    " * (len(relative.parts) - 1)
        suffix = "/" if path.is_dir() else ""
        lines.append(f"{indent}{relative.name}{suffix}")
    return "\n".join(lines)

print(artifact_tree(ARTIFACTS))
status_summary = {
    "CUDA environment": STATE.get("cuda_environment", "FAIL"),
    "CUDA extension build": STATE.get("cuda_extension_build", "FAIL"),
    "CUDA correctness": STATE.get("cuda_correctness", "FAIL"),
    "Softmax benchmarks": STATE.get("softmax_benchmarks", "INCOMPLETE"),
    "Historical benchmarks": STATE.get("historical_benchmarks", "INCOMPLETE"),
    "Launch configuration": STATE.get("launch_configuration", "INCOMPLETE"),
    "Attention benchmarks": STATE.get("attention_benchmarks", "INCOMPLETE"),
    "PyTorch Profiler": STATE.get("pytorch_profiler", "INCOMPLETE"),
    "Nsight Compute": STATE.get("nsight_compute", "UNAVAILABLE"),
    "Figures generated": (
        f"{len({path.stem for path in (ARTIFACTS / 'figures').glob('fig_*')})}/6"
    ),
}
print(json.dumps(status_summary, indent=2))

if BACKUP_TO_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    backup_root = Path("/content/drive/MyDrive/cuda_softmax_research_runs")
    backup_root.mkdir(parents=True, exist_ok=True)
    backup_destination = backup_root / RUN_TIMESTAMP.replace(":", "-")
    if backup_destination.exists():
        raise FileExistsError(f"Refusing to overwrite Drive backup: {backup_destination}")
    shutil.copytree(ARTIFACTS, backup_destination)
    print(f"Drive backup: {backup_destination}")

zip_path = Path("/content/cuda_softmax_research_artifacts.zip")
if zip_path.exists():
    if ARTIFACT_POLICY == "ERROR":
        raise FileExistsError(f"Refusing to overwrite {zip_path}")
    if ARTIFACT_POLICY == "OVERWRITE":
        zip_path.unlink()

if not zip_path.exists():
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(ARTIFACTS.rglob("*")):
            if path.is_file():
                archive.write(path, Path("artifacts") / path.relative_to(ARTIFACTS))
print(f"Artifact ZIP: {zip_path} ({zip_path.stat().st_size} bytes)")
'''
            ),
            code(
                r'''
# Run this cell only when you are ready for the browser download dialog.
from google.colab import files
files.download("/content/cuda_softmax_research_artifacts.zip")
'''
            ),
            markdown(
                r"""
# WHAT TO SEND TO CHATGPT TO FINISH THE PAPER

Upload `cuda_softmax_research_artifacts.zip`. Also provide the current repository
source/Git history and, if they are not already available, `main.tex` and
`references.bib`.

The ZIP should contain environment and Git reports, build logs, correctness
records, raw and summary softmax data, historical and launch data when run,
attention data, profiler artifacts, supported figures, paper tables, the
validation report, manifest, and hypothesis evaluation.

Ask ChatGPT to replace only claims and placeholders directly supported by these
artifacts. Missing figures, unavailable Nsight counters, failed historical
builds, or inconclusive hypotheses must remain explicit limitations rather than
being repaired with invented values.

## MEASURED

Use facts directly present in the ZIP.

## INTERPRETATION

Explain potential causes separately and cautiously.

## NEXT EXPERIMENT

Propose a controlled measurement that could test each interpretation.
"""
            ),
        ]
    )

    for index, cell in enumerate(cells, start=1):
        cell["id"] = f"cell-{index:03d}"
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def rendered_notebook() -> str:
    return json.dumps(build_notebook(), indent=1, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    rendered = rendered_notebook()
    if arguments.check:
        if not NOTEBOOK_PATH.is_file() or NOTEBOOK_PATH.read_text(encoding="utf-8") != rendered:
            print(f"Notebook is stale; regenerate {NOTEBOOK_PATH}", file=sys.stderr)
            return 1
        print(f"Notebook is current: {NOTEBOOK_PATH}")
        return 0
    NOTEBOOK_PATH.write_text(rendered, encoding="utf-8")
    print(f"Wrote {NOTEBOOK_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
