"""Per-case orchestration: PDF -> trusted page text -> PageDocs -> merged
fields -> adjudication -> prediction record."""

import re

from solution import extract, pagedoc, pdfio
from solution.adjudicate import adjudicate
from solution.vocab import ALL_RISK_FLAGS

MIN_TEXT_CHARS = 120  # below this, a page is treated as scanned and OCR'd


def gather_pages(pdf_path: str) -> list[pagedoc.PageDoc]:
    doc = pdfio.open_pdf(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        spans = pdfio.classify_spans(page, i)
        trusted = [s for s in spans if s.trusted]
        visible = "\n".join(s.text for s in trusted)
        if len(visible.strip()) >= MIN_TEXT_CHARS:
            pd = pagedoc.parse_page(i, visible, "text")
        else:
            from solution import ocr

            text, conf, img = ocr.ocr_page_full(page)
            # Footer text layer still carries the packet id on scanned pages.
            footer = "\n".join(s.text for s in trusted)
            pd = pagedoc.parse_page(i, text + "\n" + footer, "ocr", ocr_conf=conf)
            # Targeted digit-whitelist re-OCR of ID tokens, only when the
            # full-page pass saw an SPN token but no valid SPN-#### value.
            if "SPN" in text.upper() and not extract.find_sponsor_ids(text):
                for line in ocr.precise_ids(img):
                    ids = extract.find_sponsor_ids(line)
                    if ids:
                        pd.fields["sponsor_id"] = ids[0]
                        break
        pages.append(pd)
    doc.close()
    return pages


# Trust precedence per field source page type (lower = stronger).
FIELD_PRECEDENCE = {
    "intake": 2,
    "biometric": 3,
    "sponsor": 4,
    "registry": 5,
    "fee": 2,
    "note": 2,
    "unknown": 6,
}

NORMALIZERS = {
    "species_code": extract.snap_species,
    "home_world": extract.snap_world,
    "visa_class": extract.snap_visa,
    "declared_purpose": extract.snap_purpose,
    "fee_status": extract.snap_fee,
}


def _norm(key: str, raw: str) -> str | None:
    raw = raw.strip()
    if not raw:
        return None
    if key == "arrival_date":
        return extract.normalize_date(raw)
    if key == "sponsor_id":
        ids = extract.find_sponsor_ids(raw)
        return ids[0] if ids else None
    if key == "case_id":
        ids = extract.find_case_ids(raw)
        return ids[0] if ids else None
    if key in NORMALIZERS:
        return NORMALIZERS[key](raw)
    return raw


def merge_case(pages: list[pagedoc.PageDoc]) -> dict:
    """Merge page-level fields into one record using trust precedence."""
    merged: dict = {}
    provenance: dict = {}

    # Active case id: packet footer is present on every page incl. scans.
    packet_ids = [p.packet_id for p in pages if p.packet_id]
    case_id = max(set(packet_ids), key=packet_ids.count) if packet_ids else None

    field_keys = [
        "applicant_name",
        "species_code",
        "home_world",
        "visa_class",
        "sponsor_id",
        "arrival_date",
        "declared_purpose",
        "fee_status",
    ]

    def value_rank(p: pagedoc.PageDoc) -> float:
        """Lower is stronger: page-type precedence plus an OCR-quality penalty.

        A clean text-layer page always beats OCR output, and high-confidence
        OCR beats low-confidence OCR, regardless of document rank.
        """
        rank = FIELD_PRECEDENCE.get(p.ptype, 9)
        if p.source == "ocr":
            conf = p.ocr_conf or 0.0
            rank += 10 if conf >= 60 else (20 if conf >= 40 else 30)
        return rank

    for p in sorted(pages, key=value_rank):
        if p.decoy:
            continue  # COPY ARTIFACT pages describe a different applicant
        # Skip pages that belong to a different applicant's case.
        if case_id and p.packet_id and p.packet_id != case_id:
            continue
        page_case = _norm("case_id", p.fields.get("case_id", ""))
        if case_id and page_case and page_case != case_id:
            continue
        for key in field_keys:
            if key in merged:
                continue
            raw = p.fields.get(key)
            if raw is None:
                continue
            val = _norm(key, raw)
            if val:
                merged[key] = val
                provenance[key] = (p.ptype, p.source, p.ocr_conf)

    # Manual corrections override everything (rank-1 evidence).
    for p in pages:
        for key, raw in p.corrections.items():
            val = _norm(key, raw)
            if val:
                merged[key] = val
                provenance[key] = ("correction", p.source, p.ocr_conf)

    # Risk flags: printed observations plus derived cross-page conflicts.
    flags: set = set()
    for p in pages:
        if case_id and p.packet_id and p.packet_id != case_id:
            continue
        flags |= p.observed_flags

    merged["risk_flags"] = flags
    merged["case_id"] = case_id
    merged["_provenance"] = provenance
    merged["_pages"] = pages
    return merged


def decide(merged: dict) -> tuple[str, dict]:
    """Adjudicate: a visible adjudicator note wins; else the rule engine."""
    pages = merged["_pages"]
    signals = {}

    findings = []
    for p in pages:
        case_id = merged.get("case_id")
        if case_id and p.packet_id and p.packet_id != case_id:
            continue
        for decision, reason in p.findings:
            findings.append((p.index, decision, reason, p.source, p.ocr_conf))
    if findings:
        # Latest note wins (rescinded denials are superseded by later notes).
        decision = findings[-1][1]
        signals["note"] = True
        signals["note_source"] = findings[-1][3]
        return decision, signals

    ocr_pages = [p for p in pages if p.source == "ocr"]
    low_conf = [p for p in ocr_pages if (p.ocr_conf or 0) < 45]
    fields = {
        **{k: merged.get(k) for k in (
            "visa_class", "fee_status", "sponsor_id", "home_world", "arrival_date",
        )},
        "risk_flags": merged["risk_flags"],
        "evidence_problem": False,
        "fee_waiver_visible": any(
            (p.fields.get("waiver_code") or "").strip() not in ("", "N/A", "n/a")
            for p in pages if p.ptype == "fee"
        ),
    }
    signals["low_conf_pages"] = len(low_conf)
    return adjudicate(fields), signals


def confidence_for(merged: dict, decision: str, signals: dict) -> float:
    """Heuristic confidence v1; replaced by trained calibrator later."""
    if signals.get("note") and signals.get("note_source") == "text":
        return 0.98
    if signals.get("note"):
        return 0.9
    conf = 0.85
    if signals.get("low_conf_pages"):
        conf -= 0.15
    missing = sum(1 for k in ("visa_class", "fee_status", "sponsor_id", "arrival_date") if not merged.get(k))
    conf -= 0.08 * missing
    return round(max(0.3, min(conf, 0.97)), 2)


def run_case(pdf_path: str) -> dict | None:
    pages = gather_pages(pdf_path)
    merged = merge_case(pages)

    case_id = merged.get("case_id")
    if not case_id:
        ids = [extract.find_case_ids(p.text) for p in pages]
        flat = [i for sub in ids for i in sub]
        case_id = max(set(flat), key=flat.count) if flat else None
    if not case_id:
        return None

    decision, signals = decide(merged)
    conf = confidence_for(merged, decision, signals)

    return {
        "case_id": case_id,
        "applicant_name": merged.get("applicant_name") or "",
        "species_code": merged.get("species_code") or "",
        "home_world": merged.get("home_world") or "",
        "visa_class": merged.get("visa_class") or "",
        "sponsor_id": merged.get("sponsor_id") or "",
        "arrival_date": merged.get("arrival_date") or "",
        "declared_purpose": merged.get("declared_purpose") or "",
        "risk_flags": "|".join(sorted(merged["risk_flags"])) or "none",
        "fee_status": merged.get("fee_status") or "unknown",
        "adjudication": decision,
        "confidence": conf,
    }
