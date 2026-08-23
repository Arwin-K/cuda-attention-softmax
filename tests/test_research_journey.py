from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JOURNEY = (ROOT / "docs/research_journey.md").read_text()


def test_journey_separates_measurement_scope_and_student_reflection():
    assert "## Measured results" in JOURNEY
    assert "## What the evidence supports" in JOURNEY
    assert "## Student reflection" in JOURNEY
    assert "TODO(student)" in JOURNEY
    assert "one Tesla T4" in JOURNEY


def test_journey_links_existing_evidence_and_keeps_negative_baseline():
    links = [
        "docs/mini_paper.md",
        "docs/kernel_evolution.md",
        "results/runs/2026-08-23_tesla-t4_ca87722/artifacts/figures/fig_historical_speedup.png",
        "results/runs/2026-08-23_tesla-t4_ca87722/artifacts/figures/fig_kernel_vs_attention_speedup.png",
    ]
    assert all((ROOT / path).is_file() for path in links)
    assert "SDPA was faster" in JOURNEY
    assert "not a universal launch constant" in JOURNEY
