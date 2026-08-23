#!/usr/bin/env python3
"""Run the publication evidence, scope, structure, and history audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_kernel_history import audit as audit_kernel_history
from scripts.audit_public_claims import build_report as build_claim_report
from scripts.audit_results import audit_artifacts
from scripts.audit_website_journal import audit as audit_website_journal
from scripts.generate_result_provenance import generate as generate_provenance


DAY_7_BEFORE_SCOPE_AUDIT = [
    "complete LaTeX-ready white paper source outline and references",
    "complete technical blog post and recruiter-focused README",
    "add NVIDIA interview notes and project defense questions",
    "complete publication checkpoint before final audit work",
    "audit benchmark schema completeness across all result paths",
    "add result provenance links from figures to source CSV files",
    "add CPU-only reproducibility verification workflow",
    "add NVIDIA execution handoff checklist",
    "document complete kernel evolution through Git milestones",
    "add website-ready research journey narrative",
    "add website-ready per-commit journal index",
    "cross-check website narrative against research artifacts",
    "add reproducibility commands to publication documentation",
    "review educational comments in final CUDA implementation",
]
AUDITED_REPOSITORY_HEAD = "19b2d9db8ac597b917580ea68448bda3e0526803"
PUBLIC_DOCS = [
    "README.md",
    "docs/mini_paper.md",
    "docs/blog_post.md",
    "docs/results.md",
    "docs/discussion.md",
    "docs/limitations.md",
    "docs/research_journey.md",
    "docs/interview_notes.md",
]
STALE_PHRASES = [
    "raw artifact zip is missing",
    "raw zip is missing",
    "raw zip is absent",
    "has not yet been compiled",
    "no cuda correctness",
    "quantitative findings still pending",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def build_report(root: Path) -> dict[str, object]:
    root = root.resolve()
    run = root / "results/runs/2026-08-23_tesla-t4_ca87722"
    artifacts = run / "artifacts"
    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, detail: object) -> None:
        checks.append({"name": name, "status": "PASS" if passed else "FAIL", "detail": detail})

    schema = audit_artifacts(artifacts)
    check("result schema", schema["status"] == "PASS", schema["status"])
    checked_schema = json.loads((run / "schema_audit.json").read_text())
    normalized_schema = json.loads(json.dumps(schema))
    check("checked schema report current", checked_schema == normalized_schema, "exact regeneration")

    provenance = generate_provenance(artifacts)
    checked_provenance = json.loads((run / "figure_provenance.json").read_text())
    check("figure provenance current", checked_provenance == provenance, f"{len(provenance['figures'])} figures")

    claims = build_claim_report(root, artifacts)
    checked_claims = json.loads((run / "public_claim_audit.json").read_text())
    check("public claims current", claims["status"] == "PASS" and checked_claims == claims, claims["failures"])

    kernel_failures = audit_kernel_history(root)
    check("single-source kernel history", not kernel_failures, kernel_failures or "15 milestones")
    journal_failures = audit_website_journal(root / "WEBSITE_JOURNAL.md")
    check("website journal", not journal_failures, journal_failures or "001-112 exactly once")

    tracked = [Path(line) for line in git(root, "ls-files").splitlines()]
    forbidden = {"v1", "v2", "v3", "naive", "optimized", "final"}
    bad_paths = sorted(str(path) for path in tracked if path.stem.lower() in forbidden)
    check("forbidden version files", not bad_paths, bad_paths or "none")

    head = git(root, "rev-parse", AUDITED_REPOSITORY_HEAD)
    # The audited Day 7 commits live on the first-parent chain ending at the
    # fixed pre-audit revision.  Comparing that revision with ``main`` worked
    # only before the day-seven branch was merged: after the merge, ``main``
    # contains ``head`` and the range is empty.  Anchor the window to the fixed
    # audited revision itself so the check remains reproducible from both the
    # topic branch and the merged default branch.
    day7_base = f"{head}~{len(DAY_7_BEFORE_SCOPE_AUDIT)}"
    day7_subjects = git(
        root,
        "log",
        "--first-parent",
        "--reverse",
        "--format=%s",
        f"{day7_base}..{head}",
    ).splitlines()
    check("Day 7 pre-audit sequence", day7_subjects == DAY_7_BEFORE_SCOPE_AUDIT, day7_subjects)

    workflow_notebook = json.loads((root / "notebooks/04_colab_research_experiments.ipynb").read_text())
    workflow_clean = all(
        not cell.get("outputs") and cell.get("execution_count") is None
        for cell in workflow_notebook["cells"]
        if cell["cell_type"] == "code"
    )
    executed_path = run / "executed_notebook.ipynb"
    executed_notebook = json.loads(executed_path.read_text())
    executed_has_evidence = any(
        cell.get("outputs") or cell.get("execution_count") is not None
        for cell in executed_notebook["cells"]
        if cell["cell_type"] == "code"
    )
    check("notebook boundary", workflow_clean and executed_has_evidence, {"workflow_output_free": workflow_clean, "executed_preserved": executed_has_evidence})
    check("executed notebook source hash", sha256(executed_path) == "97badbb4428449fca0e07e3f6a893611b09428ddebf4ccffd6114dbbf4043770", sha256(executed_path))

    stale = []
    for relative in PUBLIC_DOCS:
        text = (root / relative).read_text(encoding="utf-8").lower()
        stale.extend(f"{relative}: {phrase}" for phrase in STALE_PHRASES if phrase in text)
    check("no stale missing-evidence claims", not stale, stale or "none")

    manifest = json.loads((artifacts / "experiment_manifest.json").read_text())
    check("measured revision separated", manifest["git_commit"] != head and manifest["git_dirty"] is False, {"measured": manifest["git_commit"], "audited_head": head})

    limitations = (root / "docs/limitations.md").read_text()
    scope_terms = ["FP32 forward", "one Tesla T4", "one Google\nColab session", "backward", "mixed precision", "SDPA"]
    check("scope limitations explicit", all(term in limitations for term in scope_terms), scope_terms)
    discussion = (root / "docs/discussion.md").read_text()
    check("model separated from measurement", "retained as a model, not relabeled as measurement" in discussion, "Amdahl boundary")

    return {
        "schema_version": 1,
        "audited_repository_head": head,
        "measured_git_commit": manifest["git_commit"],
        "status": "PASS" if all(item["status"] == "PASS" for item in checks) else "FAIL",
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_report(args.repository_root)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
