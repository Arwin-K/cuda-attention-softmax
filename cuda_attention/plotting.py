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
