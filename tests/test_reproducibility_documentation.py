from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCUMENT = (ROOT / "docs/reproducibility.md").read_text()


def test_reproducibility_page_labels_platform_and_execution_status():
    assert "Verified on Apple Silicon" in DOCUMENT
    assert "Recorded T4 execution" in DOCUMENT
    assert "Documented only" in DOCUMENT
    assert "not executable\non the Day 7 Mac" in DOCUMENT


def test_reproducibility_commands_reference_existing_entry_points():
    expected = [
        "scripts/verify_cpu_reproducibility.sh",
        "scripts/audit_results.py",
        "scripts/generate_result_provenance.py",
        "scripts/audit_public_claims.py",
        "scripts/audit_kernel_history.py",
        "scripts/audit_website_journal.py",
        "scripts/build_extension.sh",
        "benchmarks/benchmark_softmax.py",
        "benchmarks/benchmark_attention.py",
        "profiling/profile_pytorch.py",
        "profiling/run_ncu.sh",
    ]
    assert all((ROOT / path).exists() for path in expected)
    assert all(path in DOCUMENT for path in expected)


def test_publication_build_is_not_claimed_as_locally_verified():
    assert "pdflatex" in DOCUMENT
    assert "unavailable on the Day\n7 Mac" in DOCUMENT
