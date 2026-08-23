"""CPU-safe plotting helpers driven only by benchmark summary CSV files."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REQUIRED_SUMMARY_FIELDS = {
    "git_commit",
    "implementation_description",
    "sequence_length",
    "median_us",
    "elements_per_second",
    "gpu_name",
    "compute_capability",
    "pytorch_version",
    "cuda_version",
}

PROVENANCE_FIELDS = (
    "gpu_name",
    "compute_capability",
    "pytorch_version",
    "cuda_version",
)

REQUIRED_SPEEDUP_COMPARISON_FIELDS = {
    "git_commit",
    "sequence_length",
    "kernel_speedup",
    "attention_speedup",
    "translation_ratio",
    *PROVENANCE_FIELDS,
}


@dataclass(frozen=True)
class PlotSeries:
    """One commit-specific implementation series ordered by sequence length."""

    label: str
    x: tuple[int, ...]
    y: tuple[float, ...]


def load_summary_csv(path: Path) -> list[dict[str, str]]:
    """Load nonempty summary data with the fields required for figures."""

    if not path.is_file():
        raise FileNotFoundError(f"summary CSV does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        fields = set(reader.fieldnames or ())
        if not REQUIRED_SUMMARY_FIELDS.issubset(fields):
            raise ValueError("summary CSV is missing required plotting fields")
        records = list(reader)
    if not records:
        raise ValueError("summary CSV contains no measurements")
    return records


def load_speedup_comparison_csv(path: Path) -> list[dict[str, str]]:
    """Load matched kernel/attention ratios without accepting partial schema."""

    if not path.is_file():
        raise FileNotFoundError(f"speedup comparison CSV does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        if not REQUIRED_SPEEDUP_COMPARISON_FIELDS.issubset(reader.fieldnames or ()):
            raise ValueError("speedup comparison CSV is missing required fields")
        records = list(reader)
    if not records:
        raise ValueError("speedup comparison CSV contains no measurements")
    return records


def metric_series(
    records: Iterable[dict[str, str]],
    metric: str,
) -> tuple[PlotSeries, ...]:
    """Group a numeric metric by implementation and exact Git revision."""

    if metric not in {"median_us", "elements_per_second"}:
        raise ValueError(f"unsupported plot metric: {metric}")
    grouped: dict[tuple[str, str], list[tuple[int, float]]] = {}
    for record in records:
        key = (record["implementation_description"], record["git_commit"])
        grouped.setdefault(key, []).append(
            (int(record["sequence_length"]), float(record[metric]))
        )
    if not grouped:
        raise ValueError("cannot plot an empty measurement set")

    series = []
    for (description, commit), points in sorted(grouped.items()):
        ordered = sorted(points)
        series.append(
            PlotSeries(
                label=f"{description} ({commit[:8]})",
                x=tuple(point[0] for point in ordered),
                y=tuple(point[1] for point in ordered),
            )
        )
    return tuple(series)


def speedup_series(records: Iterable[dict[str, str]]) -> tuple[PlotSeries, ...]:
    """Calculate candidate speedup over matched eager PyTorch medians."""

    materialized = list(records)
    baselines: dict[tuple[str, ...], float] = {}
    for record in materialized:
        if record["implementation_description"].startswith("PyTorch eager"):
            key = (
                *(record[field] for field in PROVENANCE_FIELDS),
                record["sequence_length"],
            )
            if key in baselines:
                raise ValueError("speedup plot requires one eager baseline per case")
            baseline = float(record["median_us"])
            if baseline <= 0.0:
                raise ValueError("speedup plot requires positive median latency")
            baselines[key] = baseline
    if not baselines:
        raise ValueError("speedup plot requires measured eager baselines")

    grouped: dict[tuple[str, str], list[tuple[int, float]]] = {}
    for record in materialized:
        description = record["implementation_description"]
        if description.startswith("PyTorch eager"):
            continue
        key = (
            *(record[field] for field in PROVENANCE_FIELDS),
            record["sequence_length"],
        )
        if key not in baselines:
            raise ValueError("speedup candidate has no matched eager baseline")
        candidate = float(record["median_us"])
        if candidate <= 0.0:
            raise ValueError("speedup plot requires positive median latency")
        series_key = (description, record["git_commit"])
        grouped.setdefault(series_key, []).append(
            (int(record["sequence_length"]), baselines[key] / candidate)
        )
    if not grouped:
        raise ValueError("speedup plot requires at least one candidate")

    return tuple(
        PlotSeries(
            label=f"{description} vs eager ({commit[:8]})",
            x=tuple(point[0] for point in sorted(points)),
            y=tuple(point[1] for point in sorted(points)),
        )
        for (description, commit), points in sorted(grouped.items())
    )


def kernel_attention_speedup_series(
    records: Iterable[dict[str, str]],
) -> tuple[PlotSeries, PlotSeries]:
    """Prepare isolated-kernel and full-attention speedups on matched axes."""

    materialized = list(records)
    if not materialized:
        raise ValueError("cannot plot an empty speedup comparison")
    provenance = {
        tuple(record[field] for field in ("git_commit", *PROVENANCE_FIELDS))
        for record in materialized
    }
    if len(provenance) != 1:
        raise ValueError("kernel/attention figure requires one matched environment")

    points: dict[int, tuple[float, float]] = {}
    for record in materialized:
        sequence_length = int(record["sequence_length"])
        kernel_speedup = float(record["kernel_speedup"])
        attention_speedup = float(record["attention_speedup"])
        if min(kernel_speedup, attention_speedup) <= 0.0:
            raise ValueError("speedup values must be positive")
        if sequence_length in points:
            raise ValueError("speedup comparison requires one row per sequence length")
        points[sequence_length] = (kernel_speedup, attention_speedup)

    ordered = sorted(points.items())
    x = tuple(point[0] for point in ordered)
    return (
        PlotSeries(
            label="Isolated fused softmax",
            x=x,
            y=tuple(point[1][0] for point in ordered),
        ),
        PlotSeries(
            label="Complete explicit attention",
            x=x,
            y=tuple(point[1][1] for point in ordered),
        ),
    )


def plot_summary_metric(
    records: list[dict[str, str]],
    *,
    metric: str,
    output_path: Path,
) -> None:
    """Render a commit-aware line plot using a noninteractive backend."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "plot generation requires the matplotlib development dependency"
        ) from error

    prepared_series = metric_series(records, metric)
    labels = {
        "median_us": ("Median latency (microseconds)", "Softmax latency"),
        "elements_per_second": ("Elements per second", "Softmax throughput"),
    }
    ylabel, title = labels[metric]
    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    for series in prepared_series:
        axis.plot(series.x, series.y, marker="o", label=series.label)
    axis.set_xlabel("Sequence length")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_latency(records: list[dict[str, str]], output_path: Path) -> None:
    plot_summary_metric(records, metric="median_us", output_path=output_path)


def plot_throughput(records: list[dict[str, str]], output_path: Path) -> None:
    plot_summary_metric(
        records,
        metric="elements_per_second",
        output_path=output_path,
    )


def plot_speedup(records: list[dict[str, str]], output_path: Path) -> None:
    """Render eager-relative speedups, where one means equal median latency."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "plot generation requires the matplotlib development dependency"
        ) from error

    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    for series in speedup_series(records):
        axis.plot(series.x, series.y, marker="o", label=series.label)
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axis.set_xlabel("Sequence length")
    axis.set_ylabel("Speedup over PyTorch eager (x)")
    axis.set_title("Fused causal softmax speedup")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_kernel_attention_speedup(
    records: list[dict[str, str]],
    output_path: Path,
) -> None:
    """Plot microkernel and application speedup from matched measurements."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "plot generation requires the matplotlib development dependency"
        ) from error

    figure, axis = plt.subplots(figsize=(7.0, 4.5))
    for series in kernel_attention_speedup_series(records):
        axis.plot(series.x, series.y, marker="o", label=series.label)
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axis.set_xlabel("Sequence length")
    axis.set_ylabel("Speedup over equivalent eager path (x)")
    axis.set_title("Kernel speedup versus complete-attention speedup")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
