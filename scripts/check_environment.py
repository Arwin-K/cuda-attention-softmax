"""Print the host and optional accelerator capabilities as JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys


# Running ``python scripts/check_environment.py`` puts ``scripts/`` rather than
# the repository root on sys.path. Adding the root makes this source-checkout
# workflow work before the package is installed.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cuda_attention import detect_environment


def environment_report() -> dict[str, object]:
    """Combine Python runtime information with required command discovery."""

    report = detect_environment().as_dict()
    nvcc_path = shutil.which("nvcc")
    nvidia_smi_path = shutil.which("nvidia-smi")
    report.update(
        {
            "nvcc_path": nvcc_path,
            "nvidia_smi_path": nvidia_smi_path,
            "cuda_build_ready": bool(
                report["operating_system"] == "Linux"
                and report["pytorch_available"]
                and report["pytorch_cuda_version"]
                and nvcc_path
            ),
            "cuda_execution_ready": bool(
                report["cuda_available"] and nvidia_smi_path
            ),
        }
    )
    return report


def main() -> int:
    """Emit a stable report and optionally require a build/run CUDA host."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="return a nonzero status unless CUDA build and execution are ready",
    )
    arguments = parser.parse_args()
    report = environment_report()
    print(json.dumps(report, indent=2, sort_keys=True))

    if arguments.require_cuda and not (
        report["cuda_build_ready"] and report["cuda_execution_ready"]
    ):
        print(
            "CUDA build/execution requirements are not satisfied on this host.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
