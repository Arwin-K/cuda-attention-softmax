from pathlib import Path

from scripts.audit_website_journal import audit, parse_entries


ROOT = Path(__file__).resolve().parents[1]
JOURNAL = ROOT / "WEBSITE_JOURNAL.md"


def test_website_journal_has_every_commit_exactly_once_with_status():
    assert audit(JOURNAL) == []
    assert [number for number, _ in parse_entries(JOURNAL)] == list(range(1, 113))


def test_day_seven_preserves_future_status_until_work_is_done():
    entries = dict(parse_entries(JOURNAL))
    assert all(entries[number].startswith("Completed") for number in range(97, 111))
    assert all(entries[number].startswith("Planned") for number in range(111, 113))
