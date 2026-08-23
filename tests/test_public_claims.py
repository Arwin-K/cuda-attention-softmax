import json
from pathlib import Path

from scripts.audit_public_claims import build_report


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/runs/2026-08-23_tesla-t4_ca87722"


def test_public_claims_match_artifacts_and_documents():
    report = build_report(ROOT, RUN / "artifacts")
    assert report["status"] == "PASS", report["failures"]
    assert report["claims"]["correctness"]["passed_cases"] == 88
    assert report["claims"]["custom_vs_compile"]["custom_wins"] == 6
    assert report["claims"]["custom_attention_vs_sdpa"]["sdpa_wins"] == 7
    assert report["claims"]["kernel_vs_attention_translation"]["kernel_speedup_greater"] == 6


def test_checked_in_public_claim_audit_is_current():
    checked_in = json.loads((RUN / "public_claim_audit.json").read_text())
    assert checked_in == build_report(ROOT, RUN / "artifacts")
