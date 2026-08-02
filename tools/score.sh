#!/usr/bin/env bash
# Dev loop: run the pipeline on training PDFs and score against public labels.
# Usage: tools/score.sh [subset_dir]  (defaults to full data/train)
set -euo pipefail

CHALLENGE=${CHALLENGE:-"$HOME/Desktop/mib-doc-challenge"}
SOLUTION="$(cd "$(dirname "$0")/.." && pwd)"
INPUT="${1:-$CHALLENGE/data/train}"
OUT=/tmp/mib-dev
mkdir -p "$OUT"

PYTHONPATH="$SOLUTION" "$SOLUTION/.venv/bin/python" -m solution.main "$INPUT" "$OUT/predictions.jsonl"
python3 "$CHALLENGE/scripts/evaluate.py" \
  --truth "$CHALLENGE/data/train_labels.csv" \
  --submission "$OUT/predictions.jsonl" \
  --output-json "$OUT/evaluation.json" \
  --case-scores-jsonl "$OUT/case_scores.jsonl" || true
python3 - "$OUT/evaluation.json" << 'EOF'
import json, sys
e = json.load(open(sys.argv[1]))
for k in ("total_score", "classification_score", "extraction_score", "calibration_score", "missing_penalty"):
    if k in e:
        print(f"{k}: {e[k]}")
EOF
