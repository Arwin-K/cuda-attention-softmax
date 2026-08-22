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
BRANCH = "day-five-pt-2"
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
    "nvidia_smi_output": nvidia_smi.stdout,
    "nvcc_output": nvcc.stdout,
}
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
