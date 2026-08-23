"""Generate reproducible LaTeX result tables from measured benchmark CSVs."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.compare_speedups import (
    load_attention_raw_csv,
    summarize_attention_records,
)
from benchmarks.config import BENCHMARK_SEQUENCE_LENGTHS
from cuda_attention.plotting import (
    load_speedup_comparison_csv,
    load_summary_csv,
)


PROVENANCE_FIELDS = (
    "git_commit",
    "gpu_name",
    "compute_capability",
    "pytorch_version",
    "cuda_version",
)


def _latex_escape(value: object) -> str:
    text = str(value)
    for source, replacement in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("_", r"\_"),
        ("#", r"\#"),
    ):
        text = text.replace(source, replacement)
    return text


def render_latex_table(
    *,
    headers: Sequence[str],
    rows: Sequence[Sequence[object]],
    caption: str,
    label: str,
) -> str:
    """Render a compact table whose values already came from validated CSVs."""

    if not rows:
        raise ValueError("cannot render an empty results table")
    if any(len(row) != len(headers) for row in rows):
        raise ValueError("table row width does not match headers")
    column_specification = "l" + "r" * (len(headers) - 1)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{_latex_escape(caption)}}}",
        f"\\label{{{_latex_escape(label)}}}",
        f"\\begin{{tabular}}{{{column_specification}}}",
        r"\hline",
        " & ".join(_latex_escape(header) for header in headers) + r" \\",
        r"\hline",
    ]
    lines.extend(
        " & ".join(_latex_escape(value) for value in row) + r" \\"
        for row in rows
    )
    lines.extend((r"\hline", r"\end{tabular}", r"\end{table}", ""))
    return "\n".join(lines)


def _role(description: str, mapping: Sequence[tuple[str, str]]) -> str | None:
    for prefix, role in mapping:
        if description.startswith(prefix):
            return role
    return None


def _complete_role_rows(
    records: Sequence[Mapping[str, object]],
    *,
    description_field: str,
    role_mapping: Sequence[tuple[str, str]],
    required_lengths: Sequence[int],
) -> list[tuple[int, str, Mapping[str, object]]]:
    expected_roles = {role for _, role in role_mapping}
    selected: dict[tuple[int, str], Mapping[str, object]] = {}
    for record in records:
        role = _role(str(record[description_field]), role_mapping)
        if role is None:
            continue
        key = (int(record["sequence_length"]), role)
        if key in selected:
            raise ValueError("results tables require one row per path and shape")
        selected[key] = record
    expected = {
        (sequence_length, role)
        for sequence_length in required_lengths
        for role in expected_roles
    }
    if set(selected) != expected:
        raise ValueError("results tables require every planned path and shape")
    return [
        (sequence_length, role, selected[(sequence_length, role)])
        for sequence_length in required_lengths
        for role in sorted(expected_roles)
    ]


def _require_matched_provenance(
    groups: Sequence[Sequence[Mapping[str, object]]],
) -> None:
    provenance = {
        tuple(str(record[field]) for field in PROVENANCE_FIELDS)
        for group in groups
        for record in group
    }
    if len(provenance) != 1:
        raise ValueError("results tables require one matched Git/GPU environment")


def generate_results_tables(
    *,
    softmax_summary_path: Path,
    attention_raw_path: Path,
    speedup_comparison_path: Path,
    output_directory: Path,
    required_lengths: Sequence[int] = BENCHMARK_SEQUENCE_LENGTHS,
) -> tuple[Path, Path, Path]:
    """Validate complete evidence and write three independently includable tables."""

    softmax = load_summary_csv(softmax_summary_path)
    attention = summarize_attention_records(
        load_attention_raw_csv(attention_raw_path)
    )
    comparison = load_speedup_comparison_csv(speedup_comparison_path)
    softmax_roles = _complete_role_rows(
        softmax,
        description_field="implementation_description",
        role_mapping=(
            ("PyTorch eager", "Eager"),
            ("torch.compile", "Compiled"),
            ("warp-reduction custom CUDA", "Custom CUDA"),
        ),
        required_lengths=required_lengths,
    )
    attention_roles = _complete_role_rows(
        attention,
        description_field="implementation",
        role_mapping=(
            ("explicit_eager", "Explicit eager"),
            ("custom_cuda", "Custom CUDA"),
            ("pytorch_sdpa", "PyTorch SDPA"),
        ),
        required_lengths=required_lengths,
    )
    if {int(row["sequence_length"]) for row in comparison} != set(required_lengths):
        raise ValueError("speedup table requires every planned sequence length")
    _require_matched_provenance(
        (
            [record for _, _, record in softmax_roles],
            [record for _, _, record in attention_roles],
            comparison,
        )
    )

    outputs = (
        output_directory / "softmax_results.tex",
        output_directory / "attention_results.tex",
        output_directory / "kernel_attention_speedup.tex",
    )
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite results tables: {existing}")
    output_directory.mkdir(parents=True, exist_ok=True)

    softmax_rows = [
        (
            role,
            length,
            f'{float(record["median_us"]):.3f}',
            f'{float(record["p25_us"]):.3f}',
            f'{float(record["p75_us"]):.3f}',
            f'{float(record["elements_per_second"]):.3e}',
        )
        for length, role, record in softmax_roles
    ]
    attention_rows = [
        (
            role,
            length,
            f'{float(record["median_us"]):.3f}',
            f'{float(record["p25_us"]):.3f}',
            f'{float(record["p75_us"]):.3f}',
        )
        for length, role, record in attention_roles
    ]
    comparison_rows = [
        (
            int(record["sequence_length"]),
            f'{float(record["kernel_speedup"]):.3f}',
            f'{float(record["attention_speedup"]):.3f}',
            f'{float(record["translation_ratio"]):.3f}',
        )
        for record in sorted(comparison, key=lambda row: int(row["sequence_length"]))
    ]
    documents = (
        render_latex_table(
            headers=("Path", "S", "Median us", "P25 us", "P75 us", "Elements/s"),
            rows=softmax_rows,
            caption="Fused causal softmax benchmark results.",
            label="tab:softmax-results",
        ),
        render_latex_table(
            headers=("Path", "S", "Median us", "P25 us", "P75 us"),
            rows=attention_rows,
            caption="Complete causal attention benchmark results.",
            label="tab:attention-results",
        ),
        render_latex_table(
            headers=("S", "Kernel speedup", "Attention speedup", "Translation ratio"),
            rows=comparison_rows,
            caption="Isolated-kernel and complete-attention speedup.",
            label="tab:speedup-translation",
        ),
    )
    for path, document in zip(outputs, documents, strict=True):
        path.write_text(document, encoding="utf-8")
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--softmax-summary", type=Path, required=True)
    parser.add_argument("--attention-raw", type=Path, required=True)
    parser.add_argument("--speedup-comparison", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        outputs = generate_results_tables(
            softmax_summary_path=arguments.softmax_summary,
            attention_raw_path=arguments.attention_raw,
            speedup_comparison_path=arguments.speedup_comparison,
            output_directory=arguments.output_directory,
        )
    except (FileExistsError, FileNotFoundError, KeyError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
