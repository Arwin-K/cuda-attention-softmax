"""CPU-safe structural checks for the generated Colab experiment notebook."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks/04_colab_research_experiments.ipynb"
GENERATOR_PATH = PROJECT_ROOT / "scripts/generate_colab_notebook.py"


def load_notebook() -> dict[str, object]:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def notebook_text(notebook: dict[str, object]) -> str:
    cells = notebook["cells"]
    assert isinstance(cells, list)
    return "\n".join(str(cell["source"]) for cell in cells)


def test_checked_in_notebook_matches_deterministic_generator() -> None:
    result = subprocess.run(
        [sys.executable, str(GENERATOR_PATH), "--check"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_notebook_has_valid_format_and_no_saved_experiment_outputs() -> None:
    notebook = load_notebook()

    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] == 5
    cells = notebook["cells"]
    assert isinstance(cells, list)
    assert len(cells) >= 35
    assert len({cell["id"] for cell in cells}) == len(cells)
    for cell in cells:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []


def test_every_python_code_cell_parses_without_colab_magics() -> None:
    notebook = load_notebook()

    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse(str(cell["source"]), filename=f"notebook-cell-{index}")


def test_notebook_contains_complete_numbered_workflow_and_handoff() -> None:
    text = notebook_text(load_notebook())

    required_headings = [
        "# HOW TO USE THIS NOTEBOOK",
        *[f"# {number}." for number in range(1, 19)],
        "# WHAT TO SEND TO CHATGPT TO FINISH THE PAPER",
    ]
    for heading in required_headings:
        assert heading in text


def test_notebook_is_connected_to_repository_without_embedded_secrets() -> None:
    text = notebook_text(load_notebook())

    assert 'REPO_URL = "https://github.com/Arwin-K/cuda-attention-softmax.git"' in text
    assert 'BRANCH = "day-five-pt-2"' in text
    assert "scripts/build_extension.sh" in text
    assert "benchmarks/benchmark_softmax.py" in text
    assert "tests/test_cuda_operator.py" in text
    assert "github_pat_" not in text
    assert "ghp_" not in text
    assert '"gpuType"' not in text


def test_notebook_names_required_evidence_and_integrity_guards() -> None:
    text = notebook_text(load_notebook())
    required_artifacts = [
        "environment/environment.json",
        "build/build_log.txt",
        "correctness/correctness_results.csv",
        "benchmarks/raw/softmax_raw.csv",
        "benchmarks/raw/historical_raw.csv",
        "benchmarks/raw/launch_configuration_raw.csv",
        "attention/attention_raw.csv",
        "profiler/pytorch_profiler_trace.json",
        "validation_report.md",
        "experiment_manifest.json",
        "README_EXPERIMENTS.md",
        "cuda_softmax_research_artifacts.zip",
    ]
    for artifact in required_artifacts:
        assert artifact in text

    assert "SPEEDUP NOT COMPUTED: incompatible experimental conditions." in text
    assert "NSIGHT COMPUTE UNAVAILABLE IN THIS COLAB ENVIRONMENT" in text
    assert 'ARTIFACT_POLICY = "ERROR"' in text
    assert "Do not run custom benchmarks" in text
