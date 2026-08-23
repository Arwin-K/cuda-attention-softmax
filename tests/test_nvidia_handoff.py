from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HANDOFF = (ROOT / "docs/nvidia_handoff.md").read_text()


def test_handoff_uses_plain_clone_url_and_fixed_tolerances():
    assert "git clone https://github.com/Arwin-K/cuda-attention-softmax.git" in HANDOFF
    assert "git clone [https://" not in HANDOFF
    assert "Do not weaken\ntolerances" in HANDOFF


def test_handoff_orders_correctness_before_benchmarks_and_preserves_artifacts():
    correctness = HANDOFF.index("## 4. Verify and build before measuring")
    benchmark = HANDOFF.index("## 5. Recommended complete notebook run")
    assert correctness < benchmark
    assert "cuda_softmax_research_artifacts.zip" in HANDOFF
    assert "executed notebook" in HANDOFF
    assert "scripts/audit_results.py" in HANDOFF


def test_handoff_distinguishes_exact_replication_from_latest_rerun():
    assert "Exact measured-revision replication" in HANDOFF
    assert "Latest-workflow rerun" in HANDOFF
    assert "ca87722a00ebf585cd788c67949e7c0b32dca788" in HANDOFF
