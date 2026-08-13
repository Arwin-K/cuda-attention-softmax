"""Print the host and optional accelerator capabilities as JSON."""

from __future__ import annotations

import json
from pathlib import Path
import sys


# Running ``python scripts/check_environment.py`` puts ``scripts/`` rather than
# the repository root on sys.path. Adding the root makes this source-checkout
# workflow work before the package is installed.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cuda_attention import detect_environment


def main() -> None:
    """Emit a stable, readable capability report."""

    print(json.dumps(detect_environment().as_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
