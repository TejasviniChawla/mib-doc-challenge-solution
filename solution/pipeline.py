"""Per-case orchestration: PDF -> trusted text -> fields -> adjudication.

NOTE: label/anchor logic is a first skeleton; refined against real training
packet layouts during recon.
"""

import re
from pathlib import Path

from solution import extract, pdfio
from solution.adjudicate import adjudicate
from solution.vocab import ALL_RISK_FLAGS


def gather_text(pdf_path: str) -> dict:
    """Collect trusted (visible) and untrusted (hidden) text per page.

    Pages with a usable text layer use trusted spans; image-only or
    low-text pages fall back to OCR of the rendered image.
    """
    doc = pdfio.open_pdf(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        spans = pdfio.classify_spans(page, i)
        visible = pdfio.page_visible_text(spans)
        hidden = pdfio.page_hidden_text(spans)
        ocr_text, ocr_conf = None, None
        if len(visible.strip()) < 40:  # scanned or image-only page
            from solution import ocr

            ocr_text, ocr_conf = ocr.ocr_page(page)
        pages.append(
            {
                "index": i,
                "visible": visible,
                "hidden": hidden,
                "ocr": ocr_text,
                "ocr_conf": ocr_conf,
            }
        )
    doc.close()
    return {"path": pdf_path, "pages": pages}


def trusted_text(case: dict) -> str:
    parts = []
    for p in case["pages"]:
        parts.append(p["ocr"] if p["ocr"] is not None else p["visible"])
    return "\n".join(parts)


LABEL_PATTERNS = {
    "applicant_name": r"(?:applicant\s*name|name\s*of\s*applicant)\s*[:\-]\s*(.+)",
    "species_code": r"species(?:\s*code)?\s*[:\-]\s*(.+)",
    "home_world": r"home\s*world\s*[:\-]\s*(.+)",
    "visa_class": r"visa\s*class\s*[:\-]\s*(.+)",
    "arrival_date": r"arrival\s*date\s*[:\-]\s*(.+)",
    "declared_purpose": r"(?:declared\s*)?purpose\s*[:\-]\s*(.+)",
    "fee_status": r"fee\s*(?:status)?\s*[:\-]\s*(.+)",
    "receipt_date": r"(?:receipt|received)\s*(?:date)?\s*[:\-]\s*(.+)",
}


def extract_fields(text: str) -> dict:
    fields: dict = {}
    for key, pat in LABEL_PATTERNS.items():
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            fields[key] = m.group(1).strip()

    ids = extract.find_case_ids(text)
    fields["case_id"] = ids[0] if ids else None
    sponsors = extract.find_sponsor_ids(text)
    fields["sponsor_id"] = sponsors[0] if sponsors else None

    # Normalize / snap onto vocabularies.
    out = {
        "case_id": fields.get("case_id"),
        "applicant_name": (fields.get("applicant_name") or "").strip() or None,
        "species_code": extract.snap_species(fields.get("species_code", "")),
        "home_world": extract.snap_world(fields.get("home_world", "")),
        "visa_class": extract.snap_visa(fields.get("visa_class", "")),
        "sponsor_id": fields.get("sponsor_id"),
        "arrival_date": extract.normalize_date(fields.get("arrival_date", "") or ""),
        "declared_purpose": extract.snap_purpose(fields.get("declared_purpose", "")),
        "fee_status": extract.snap_fee(fields.get("fee_status", "")),
        "receipt_date": extract.normalize_date(fields.get("receipt_date", "") or ""),
    }

    flags = set()
    low = text.lower()
    for flag in ALL_RISK_FLAGS:
        if flag.replace("_", " ") in low or flag in low:
            flags.add(flag)
    out["risk_flags"] = flags
    return out


def run_case(pdf_path: str) -> dict | None:
    case = gather_text(pdf_path)
    text = trusted_text(case)
    fields = extract_fields(text)

    case_id = fields.get("case_id")
    if not case_id:
        # Fall back to filename convention if present (e.g. MIB-000123.pdf).
        m = re.search(r"MIB-\d{6}", Path(pdf_path).stem)
        if m:
            case_id = m.group(0)
        else:
            return None

    decision = adjudicate(fields)
    confidence = 0.7  # placeholder until the calibrator exists

    return {
        "case_id": case_id,
        "applicant_name": fields.get("applicant_name") or "",
        "species_code": fields.get("species_code") or "",
        "home_world": fields.get("home_world") or "",
        "visa_class": fields.get("visa_class") or "",
        "sponsor_id": fields.get("sponsor_id") or "",
        "arrival_date": fields.get("arrival_date") or "",
        "declared_purpose": fields.get("declared_purpose") or "",
        "risk_flags": "|".join(sorted(fields["risk_flags"])) or "none",
        "fee_status": fields.get("fee_status") or "unknown",
        "adjudication": decision,
        "confidence": confidence,
    }
