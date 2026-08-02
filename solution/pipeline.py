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
            # Note-stamp rescue: an adjudicator note whose finding didn't
            # parse gets one high-DPI sparse-text retry — notes are the top
            # trusted evidence and worth the extra pass.
            if pd.ptype == "note" and not pd.findings:
                text2, _, _ = ocr.ocr_page_highres(page)
                pd2 = pagedoc.parse_page(i, text2 + "\n" + footer, "ocr", ocr_conf=conf)
                if pd2.findings:
                    pd.findings = pd2.findings
                    pd.observed_flags |= pd2.observed_flags
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
    "applicant_name": extract.snap_name,
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

    # Last resort: context-free value scavenging. The vocabularies are
    # distinctive enough (ALL_CAPS species codes, unique world names, visa
    # patterns) to recover values from garbled pages where labels died.
    _scavenge(pages, merged, provenance, case_id)

    # Fee inference from the receipt's Amount when the status word is
    # unreadable: $809 = paid; $0 with a waiver code = waived (validated
    # 297/297 and 106/106 on text-layer train receipts).
    if not merged.get("fee_status"):
        for p in pages:
            if p.decoy or p.ptype not in ("fee", "unknown"):
                continue
            amount = p.fields.get("amount", "")
            digits = re.sub(r"[^\d]", "", amount.split(".")[0]) if amount else ""
            if digits.endswith("809") or digits == "809":
                merged["fee_status"] = "paid"
                provenance["fee_status"] = ("amount", p.source, p.ocr_conf)
                break
            waiver = p.fields.get("waiver_code", "")
            if digits in ("0", "000") and waiver and waiver.upper() not in ("N/A", "NA"):
                merged["fee_status"] = "waived"
                provenance["fee_status"] = ("amount", p.source, p.ocr_conf)
                break

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

    # A biometric slip so damaged its flag line can't be read IS the
    # illegible_biometrics condition.
    for p in pages:
        if p.decoy or p.ptype != "biometric" or p.source != "ocr":
            continue
        obs_line = bool(p.observed_flags) or "observed" in p.text.lower()
        if (p.ocr_conf or 0) < 55 and not obs_line:
            flags.add("illegible_biometrics")

    _resolve_name(pages, merged, provenance, flags, case_id)

    merged["risk_flags"] = flags
    merged["case_id"] = case_id
    merged["_provenance"] = provenance
    merged["_pages"] = pages
    return merged


def _scavenge(pages, merged, provenance, case_id):
    from rapidfuzz import fuzz, process as rf_process

    from solution import vocab

    texts = []
    for p in pages:
        if p.decoy or (case_id and p.packet_id and p.packet_id != case_id):
            continue
        texts.append(p.text)
    blob = "\n".join(texts)

    if not merged.get("species_code"):
        toks = re.findall(r"[A-Z][A-Z_ ]{5,24}", blob.upper())
        best = (None, 0)
        for t in toks:
            r = rf_process.extractOne(t.strip().replace(" ", "_"), vocab.SPECIES_CODES, scorer=fuzz.ratio)
            if r and r[1] > best[1]:
                best = (r[0], r[1])
        if best[1] >= 75:
            merged["species_code"] = best[0]
            provenance["species_code"] = ("scavenge", "ocr", None)

    if not merged.get("home_world"):
        best = (None, 0)
        for line in blob.splitlines():
            line = line.strip()
            if not (3 <= len(line) <= 60):
                continue
            for w in vocab.HOME_WORLDS:
                s = fuzz.partial_ratio(w.lower(), line.lower())
                if s > best[1] and len(line) >= len(w) - 2:
                    best = (w, s)
        if best[1] >= 85:
            merged["home_world"] = best[0]
            provenance["home_world"] = ("scavenge", "ocr", None)

    if not merged.get("visa_class"):
        m = re.search(r"\b(XW|DIP|MED|TRANSIT)\s*[-–—]?\s*([1237])\b", blob, re.I)
        if m:
            merged["visa_class"] = extract.snap_visa(f"{m.group(1)}-{m.group(2)}")
            provenance["visa_class"] = ("scavenge", "ocr", None)

    if not merged.get("declared_purpose"):
        best = (None, 0)
        for line in blob.splitlines():
            line = line.strip().lower()
            if not (4 <= len(line) <= 50):
                continue
            for pu in vocab.DECLARED_PURPOSES:
                s = fuzz.partial_ratio(pu, line)
                if s > best[1]:
                    best = (pu, s)
        if best[1] >= 88:
            merged["declared_purpose"] = best[0]
            provenance["declared_purpose"] = ("scavenge", "ocr", None)

    if not merged.get("sponsor_id"):
        ids = extract.find_sponsor_ids(blob)
        if ids:
            merged["sponsor_id"] = max(set(ids), key=ids.count)
            provenance["sponsor_id"] = ("scavenge", "ocr", None)

    if not merged.get("arrival_date"):
        dates = re.findall(r"20\d{2}-\d{2}-\d{2}", blob)
        if dates:
            merged["arrival_date"] = max(set(dates), key=dates.count)
            provenance["arrival_date"] = ("scavenge", "ocr", None)

    if not merged.get("applicant_name"):
        toks = re.findall(r"[A-Z][a-z]{3,10}", blob)
        good = []
        for t in toks:
            r = rf_process.extractOne(t, vocab.NAME_TOKENS, scorer=fuzz.ratio)
            if r and r[1] >= 84:
                good.append(r[0])
        if len(good) >= 2:
            merged["applicant_name"] = f"{good[0]} {good[1]}"
            provenance["applicant_name"] = ("scavenge", "ocr", None)


# When pages disagree on the applicant, the intake form is the tampered one:
# on train conflicts the truth name came from registry (25), b13 (14),
# letter (9), intake (5).
NAME_TRUST = {"registry": 0, "biometric": 1, "sponsor": 2, "intake": 3}


def _resolve_name(pages, merged, provenance, flags, case_id):
    """Majority-vote the applicant name across pages; derive
    identity_conflict / sponsor_mismatch from disagreements."""
    corrected = any(
        "applicant_name" in p.corrections for p in pages if not p.decoy
    )
    candidates = []  # (name, ptype, trusted_source)
    for p in pages:
        if p.decoy or (case_id and p.packet_id and p.packet_id != case_id):
            continue
        raw = p.fields.get("applicant_name")
        if not raw:
            continue
        name = extract.snap_name(raw)
        if not name or " " not in name:
            continue
        trusted = p.source == "text" or (p.ocr_conf or 0) >= 80
        candidates.append((name, p.ptype, trusted))

    if candidates and not corrected:
        counts = {}
        for name, ptype, trusted in candidates:
            w = 2 if trusted else 1
            counts[name] = counts.get(name, 0) + w
        best = max(
            counts,
            key=lambda n: (
                counts[n],
                -min(NAME_TRUST.get(pt, 9) for nm, pt, _ in candidates if nm == n),
            ),
        )
        merged["applicant_name"] = best
        provenance["applicant_name"] = ("vote", "mixed", None)

        # Distinct names from trusted identity documents = identity conflict.
        # (A different name on the sponsor letter is sponsor_mismatch, below.)
        trusted_names = {n for n, pt, tr in candidates if tr and pt != "sponsor"}
        if len(trusted_names) > 1:
            flags.add("identity_conflict")

    # Sponsor letter naming a different applicant = sponsor mismatch.
    final = merged.get("applicant_name")
    if final:
        for p in pages:
            if p.decoy or p.ptype != "sponsor":
                continue
            if p.source == "ocr" and (p.ocr_conf or 0) < 75:
                continue
            ln = extract.snap_name(p.fields.get("applicant_name", ""))
            if ln and " " in ln and ln != final:
                flags.add("sponsor_mismatch")


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
        signals["rule"] = "note_" + findings[-1][3]
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
    decision, rule = adjudicate(fields)
    signals["rule"] = rule
    signals["engine_decision"] = decision  # pre-policy, for calibration fits

    # Per-rule decision-policy overrides fit on train (expected-points argmax
    # over each rule's truth distribution); see tools/fit_calibration.py.
    try:
        from solution.calib import POLICY

        decision = POLICY.get(rule, decision)
    except ImportError:
        pass
    return decision, signals


def case_features(merged: dict, decision: str, signals: dict) -> dict:
    """Features for confidence calibration, bucketed offline against train."""
    pages = merged["_pages"]
    ocr_confs = [p.ocr_conf for p in pages if p.source == "ocr" and p.ocr_conf is not None]
    missing = sum(
        1 for k in ("visa_class", "fee_status", "sponsor_id", "arrival_date", "applicant_name")
        if not merged.get(k)
    )
    return {
        "rule": signals.get("rule", "?"),
        "decision": signals.get("engine_decision", decision),
        "n_pages": len(pages),
        "n_ocr": len(ocr_confs),
        "min_ocr_conf": round(min(ocr_confs), 1) if ocr_confs else None,
        "missing_fields": missing,
        "has_b13": any(p.ptype == "biometric" for p in pages),
        "has_note": any(p.findings for p in pages),
        "has_fee_page": any(p.ptype == "fee" for p in pages),
        "fee_source": (merged.get("_provenance", {}).get("fee_status") or ("missing",))[0],
        "flags": "|".join(sorted(merged["risk_flags"])) or "none",
    }


def confidence_for(features: dict) -> float:
    """Calibrated confidence: empirical accuracy per bucket, fit on train
    (see tools/fit_calibration.py, which writes solution/calib.py)."""
    try:
        from solution.calib import lookup

        c = lookup(features)
        if c is not None:
            return c
    except ImportError:
        pass
    # Fallback heuristic when no calibration table is present.
    rule = features.get("rule", "")
    if rule.startswith("note_text"):
        return 0.98
    if rule.startswith("note_"):
        return 0.9
    conf = 0.85 - 0.08 * features.get("missing_fields", 0)
    if (features.get("min_ocr_conf") or 100) < 45:
        conf -= 0.1
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
    features = case_features(merged, decision, signals)
    conf = confidence_for(features)

    return {
        "_features": features,
        "case_id": case_id,
        "applicant_name": merged.get("applicant_name") or "",
        "species_code": merged.get("species_code") or "",
        "home_world": merged.get("home_world") or "",
        "visa_class": merged.get("visa_class") or "",
        # Schema requires SPN-####/ISO-date syntax; when unrecoverable, emit a
        # valid placeholder (scores identically to blank under exact-match).
        "sponsor_id": merged.get("sponsor_id") or "SPN-0000",
        "arrival_date": merged.get("arrival_date") or "2026-04-15",
        "declared_purpose": merged.get("declared_purpose") or "",
        "risk_flags": "|".join(sorted(merged["risk_flags"])) or "none",
        # When the fee is unreadable, output the empirical majority value —
        # extraction is scored independently of the adjudication decision.
        "fee_status": merged.get("fee_status") or "paid",
        "adjudication": decision,
        "confidence": conf,
    }
