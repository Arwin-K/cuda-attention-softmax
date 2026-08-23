from pathlib import Path

from scripts.audit_results import audit_artifacts, missing_fields, percentile


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "results/runs/2026-08-23_tesla-t4_ca87722/artifacts"


def test_missing_fields_reports_actionable_gap():
    assert missing_fields(["git_commit", "latency_us"], ["git_commit", "gpu_name"]) == [
        "gpu_name"
    ]


def test_percentile_matches_linear_interpolation():
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.25) == 1.75
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.75) == 3.25


def test_imported_experiment_passes_complete_schema_audit():
    report = audit_artifacts(ARTIFACTS)
    assert report["status"] == "PASS", [
        item for item in report["checks"] if item["status"] == "FAIL"
    ]
