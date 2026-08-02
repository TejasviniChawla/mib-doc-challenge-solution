# MIB Doc Challenge — Solution

Offline document-processing pipeline for the [8090 MIB Doc Challenge](https://github.com/8090-inc/mib-doc-challenge):
reads a directory of adversarial PDF case packets, extracts applicant records, and adjudicates
each case as `APPROVED` / `DENIED` / `NEEDS_REVIEW` with calibrated confidence.

Fully offline: PyMuPDF + Tesseract + OpenCV + rapidfuzz + hand-written rules and a fitted
lookup table. No LLMs, no network, no GPU.

## Run

```bash
docker build -t mib-submission .
docker run --rm --network none \
  --mount type=bind,src=/path/to/pdfs,dst=/input,readonly \
  --mount type=bind,src=/path/to/out,dst=/output \
  mib-submission /input /output/predictions.jsonl
```

~1.2 s/PDF on 4 vCPUs (budget is 6 s/PDF).

## How it works

1. **Trust-classified text** — text-layer spans are filtered by color/position/size metadata
   (white-on-white, off-crop, and micro-font spans are hidden prompt-injection payloads and are
   never parsed). Scanned pages are rasterized and OCR'd — rendering is inherently
   injection-proof. OCR escalates through bounded retries (raw 300 dpi psm6 → OSD rotation →
   deskew/contrast variant).
2. **Page-type parsing** — six document types classified by fuzzy header matching; per-type
   parsing of label/value pairs, sponsor-letter prose, `Finding:` decisions (with a fuzzy
   uppercase-stamp matcher for OCR-mangled stamps), manual-correction grammar, and
   `COPY ARTIFACT` decoy detection.
3. **Closed-vocabulary snapping** — every categorical field (species, world, visa, purpose, fee,
   flags — and applicant names, a 12×12 token generator) is fuzzy-snapped onto vocabularies
   mined from the training labels, plus a context-free scavenging pass for pages whose labels
   were destroyed.
4. **Evidence-precedence merging** — fields merge across pages per the field manual's trust
   order, adjusted by source quality; names are majority-voted; corrections override all;
   fee falls back to the receipt's Amount line ($809 = paid, $0+waiver = waived).
5. **Derived flags** — identity_conflict, sponsor_mismatch, and illegible_biometrics are
   derived from cross-page conflicts and destroyed biometric slips.
6. **Adjudication** — a visible adjudicator note wins (100% agreement with train labels across
   287 recovered notes); otherwise a rule engine encoding the field manual plus edge rules
   mined from the 1,000 public training labels, with a small fitted policy layer that maximizes
   expected classification points for evidence-starved cases.
7. **Confidence** — empirical accuracy of the decision bucket (rule × OCR-quality ×
   missing-fields), fit on train (`tools/fit_calibration.py` → `solution/calib.py`).

See `MEMO.md` for the full technical write-up, failure modes, and roadmap.

## Layout

```
solution/         runtime package (installed into the Docker image)
  pipeline.py     per-case orchestration and cross-page merging
  pdfio.py        text-layer span trust classification
  ocr.py          rendering + bounded OCR escalation
  pagedoc.py      page-type classification and parsing
  adjudicate.py   policy rule engine
  vocab.py        closed vocabularies mined from train labels
  calib.py        generated: decision policy + confidence table
tools/            dev-only: scoring loop, calibration fitting
```

## License

MIT. Solution code reuses only its listed open-source dependencies.
