import json
from pathlib import Path

from scripts.audit_evidence_scope import build_report


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "results/runs/2026-08-23_tesla-t4_ca87722/evidence_scope_audit.json"


def test_evidence_scope_audit_passes():
    report = build_report(ROOT)
    assert report["status"] == "PASS", [
        check for check in report["checks"] if check["status"] == "FAIL"
    ]


def test_checked_in_scope_report_matches_audited_parent_revision():
    checked = json.loads(REPORT.read_text())
    generated = build_report(ROOT)
    assert checked == generated
