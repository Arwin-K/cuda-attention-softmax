from pathlib import Path

from scripts.audit_kernel_history import MILESTONES, audit


ROOT = Path(__file__).resolve().parents[1]


def test_documented_kernel_history_matches_git_and_single_source_rule():
    assert audit(ROOT) == []


def test_history_document_names_every_audited_milestone():
    document = (ROOT / "docs/kernel_evolution.md").read_text()
    assert all(revision in document for revision in MILESTONES)
    assert "one primary CUDA implementation" in document
