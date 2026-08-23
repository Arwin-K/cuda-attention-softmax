#!/usr/bin/env python3
"""Require one status-aware website entry for every planned commit."""

from __future__ import annotations

import re
from pathlib import Path


ENTRY = re.compile(r"^(?P<number>\d{3})\.\s+(?P<body>.+)$")


def parse_entries(path: Path) -> list[tuple[int, str]]:
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = ENTRY.match(line)
        if match:
            entries.append((int(match.group("number")), match.group("body")))
    return entries


def audit(path: Path) -> list[str]:
    entries = parse_entries(path)
    failures = []
    counts = {number: sum(candidate == number for candidate, _ in entries) for number in range(1, 113)}
    missing = [number for number, count in counts.items() if count == 0]
    duplicates = [number for number, count in counts.items() if count > 1]
    out_of_range = sorted(number for number, _ in entries if number not in counts)
    if missing:
        failures.append(f"missing entries: {missing}")
    if duplicates:
        failures.append(f"duplicate entries: {duplicates}")
    if out_of_range:
        failures.append(f"out-of-range entries: {out_of_range}")
    for number, body in entries:
        if not (body.startswith("Completed") or body.startswith("Planned")):
            failures.append(f"{number:03d} lacks explicit Completed/Planned status")
    return failures


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    path = root / "WEBSITE_JOURNAL.md"
    failures = audit(path)
    if failures:
        print("WEBSITE_JOURNAL_AUDIT: FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("WEBSITE_JOURNAL_AUDIT: PASS (001-112 exactly once, explicit status)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
