import json
from pathlib import Path

from scripts.generate_result_provenance import generate, sha256


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/runs/2026-08-23_tesla-t4_ca87722"
ARTIFACTS = RUN / "artifacts"


def test_sha256_detects_byte_changes(tmp_path):
    artifact = tmp_path / "sample.csv"
    artifact.write_bytes(b"a,b\n1,2\n")
    first = sha256(artifact)
    artifact.write_bytes(b"a,b\n1,3\n")
    assert sha256(artifact) != first


def test_every_declared_figure_has_hashed_source_provenance():
    generated = generate(ARTIFACTS)
    declared = json.loads((ARTIFACTS / "experiment_manifest.json").read_text())["figures"]
    assert {row["figure"] for row in generated["figures"]} == set(declared)
    for row in generated["figures"]:
        assert len(row["figure_sha256"]) == 64
        assert row["sources"]
        assert row["measured_git_commits"]


def test_checked_in_provenance_is_current():
    checked_in = json.loads((RUN / "figure_provenance.json").read_text())
    assert checked_in == generate(ARTIFACTS)
