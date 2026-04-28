#!/usr/bin/env bash


set -euo pipefail

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "Missing OPENAI_API_KEY"
  exit 1
fi

if [[ -z "${OPENAI_BASE_URL:-}" ]]; then
  echo "Missing OPENAI_BASE_URL"
  exit 1
fi

cd "$(dirname "$0")"

INPUT_CSV="${1:-data/processed/year_month_sample.csv}"
OUTPUT_CSV="${2:-data/final/pair_level_gpt41mini.csv}"

python -m src.llm_pair_extraction \
  --input-csv "$INPUT_CSV" \
  --output-csv "$OUTPUT_CSV"
