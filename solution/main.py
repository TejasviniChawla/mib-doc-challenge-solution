"""Entry point: process a directory of MIB case-packet PDFs into predictions.jsonl."""

import json
import multiprocessing as mp
import os
import sys
import traceback
from pathlib import Path


def process_pdf(pdf_path: str) -> dict | None:
    """Extract fields and adjudicate one case packet. Returns a prediction dict or None to omit."""
    from solution.pipeline import run_case

    try:
        return run_case(pdf_path)
    except Exception:
        sys.stderr.write(f"[error] {pdf_path}\n{traceback.format_exc()}\n")
        return None


def main() -> int:
    if len(sys.argv) != 3:
        sys.stderr.write("usage: python -m solution.main <input_pdf_dir> <output_path>\n")
        return 2

    input_dir = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    pdfs = sorted(str(p) for p in input_dir.rglob("*.pdf"))
    sys.stderr.write(f"[info] {len(pdfs)} PDFs in {input_dir}\n")

    workers = min(int(os.environ.get("MIB_WORKERS", "4")), mp.cpu_count())
    if workers > 1 and len(pdfs) > 1:
        with mp.get_context("spawn").Pool(workers) as pool:
            results = pool.map(process_pdf, pdfs, chunksize=8)
    else:
        results = [process_pdf(p) for p in pdfs]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(output_path, "w") as f:
        for record in results:
            if record is not None:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
    sys.stderr.write(f"[info] wrote {written}/{len(pdfs)} predictions to {output_path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
