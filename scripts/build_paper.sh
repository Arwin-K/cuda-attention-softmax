#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"
paper_dir="${project_root}/docs"

cd "${paper_dir}"

if command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf -bibtex -file-line-error -halt-on-error -interaction=nonstopmode paper.tex
elif command -v tectonic >/dev/null 2>&1; then
  tectonic --keep-intermediates --keep-logs paper.tex
else
  printf '%s\n' \
    'No supported LaTeX compiler was found.' \
    'Install latexmk (recommended) or Tectonic, then rerun ./scripts/build_paper.sh.' >&2
  exit 2
fi

printf 'Local paper build complete: %s\n' "${paper_dir}/paper.pdf"
