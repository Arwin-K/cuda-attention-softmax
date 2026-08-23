from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_cpu_reproducibility.sh"


def test_cpu_workflow_never_invokes_gpu_execution_paths():
    source = SCRIPT.read_text()
    assert 'export CUDA_VISIBLE_DEVICES=""' in source
    assert "build_extension.sh" not in source
    assert "run_benchmarks.sh" not in source
    assert "benchmark_softmax.py" not in source
    assert "ncu" not in source


def test_cpu_workflow_checks_tests_notebook_and_imported_results():
    source = SCRIPT.read_text()
    assert "-m pytest -q tests" in source
    assert "generate_colab_notebook.py --check" in source
    assert "scripts/audit_results.py" in source
    assert "generate_result_provenance" in source
