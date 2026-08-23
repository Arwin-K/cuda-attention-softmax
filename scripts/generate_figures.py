"""Generate evidence figures from a measured benchmark summary CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cuda_attention.plotting import (
    load_speedup_comparison_csv,
    load_summary_csv,
    plot_kernel_attention_speedup,
    plot_latency,
    plot_speedup,
    plot_throughput,
)


def generate_figures(
    summary_path: Path,
    output_directory: Path,
) -> tuple[Path, Path, Path]:
    """Create final softmax figures only from nonempty measured summaries."""

    records = load_summary_csv(summary_path)
    latency_path = output_directory / "softmax_latency.png"
    throughput_path = output_directory / "softmax_throughput.png"
    speedup_path = output_directory / "softmax_speedup.png"
    outputs = (latency_path, throughput_path, speedup_path)
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite figures: {existing}")
    plot_latency(records, latency_path)
    plot_throughput(records, throughput_path)
    plot_speedup(records, speedup_path)
    return outputs


def generate_kernel_attention_figure(
    comparison_path: Path,
    output_directory: Path,
) -> Path:
    """Generate the application-translation figure from matched speedups."""

    records = load_speedup_comparison_csv(comparison_path)
    output_path = output_directory / "kernel_vs_attention_speedup.png"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite figure: {output_path}")
    plot_kernel_attention_speedup(records, output_path)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--speedup-comparison", type=Path)
    parser.add_argument("--output-directory", type=Path, default=Path("figures"))
    arguments = parser.parse_args()

    try:
        if arguments.summary is None and arguments.speedup_comparison is None:
            raise ValueError("provide --summary and/or --speedup-comparison")
        generated: tuple[Path, ...] = ()
        if arguments.summary is not None:
            generated += generate_figures(
                arguments.summary,
                arguments.output_directory,
            )
        if arguments.speedup_comparison is not None:
            generated += (
                generate_kernel_attention_figure(
                    arguments.speedup_comparison,
                    arguments.output_directory,
                ),
            )
    except (FileExistsError, FileNotFoundError, ValueError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        return 2
    for path in generated:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
