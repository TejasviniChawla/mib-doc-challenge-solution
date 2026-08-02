"""Per-page parsing: classify page type and pull structured fields.

Handles both clean text-layer pages and noisy OCR text from scanned pages.
Trust rules: white text (ffffff) is hidden/untrusted and never parsed;
the red SAMPLE DENIAL watermark is a decoy; COPY ARTIFACT pages are
duplicated decoy pages for a different applicant and are excluded from
field merging.
"""

import re
from dataclasses import dataclass, field

from rapidfuzz import fuzz

from solution import extract

FINDING_RE = re.compile(r"Finding:\s*(APPROVED|DENIED|NEEDS_REVIEW)[.\s]*(?:Reason:\s*(.*?))?(?:$|Packet)", re.S)
CORRECTION_RE = re.compile(
    r"Manual correction:\s*(applicant|sponsor|visa class|fee status|arrival date|species|home world)\s*is\s*([^.\n]+)\.",
    re.I,
)
OBSERVED_FLAGS_RE = re.compile(r"(?:Observed|Ohserved|Obseryed|0bserved)\s*[fl]l?ags?\s*[:\-]?\s*([a-zA-Z_|,\s]+)", re.I)
SPONSOR_ATTEST_RE = re.compile(r"Sponsor\s+(SPN[-\s]?\w{4})\s+attests\s+that\s+(.+?)\s+is expected on Earth for\s+(.+?)[.\n]", re.S)
CLASS_COMPLIANCE_RE = re.compile(r"class\s+([A-Z]{2,7}[-\s]?\d)\s+compliance", re.I)
DISQ_REASON_RE = re.compile(r"risk flag:\s*([a-z_]+)", re.I)

# Stamp/marker words that contaminate OCR'd values.
STAMP_WORDS = [
    "CASEWORK", "COPY ARTIFACT", "SCAN TAB", "SAMPLE DENIAL",
    "REGISTRY IMAGE", "PASSPORT IMAGE", "MIB Eyes Only",
]


def _label(pat: str) -> re.Pattern:
    return re.compile(pat + r"\s*[:\-]?\s*\n?[ \t]*([^\n]+)", re.I)


L = {
    "case_id": _label(r"\bCase\s*.?[DI]{1,2}\b"),
    "applicant_name": _label(r"\b(?:Applicant|Registry Name)\b"),
    "species_code": _label(r"\bSpecies\s*(?:Code|Match)\b"),
    "home_world": _label(r"\bH[oa]m?e?\s*W[oa]rld\b"),
    "visa_class": _label(r"\bVis[a]?\s*Cl[a]?ss\b"),
    "sponsor_id": _label(r"\bSponsor\s*.?[DI]{1,2}\b"),
    "arrival_date": _label(r"\bArr[il]va[l1]?\s*Date\b"),
    "declared_purpose": _label(r"\b(?:Decl?[a]?red\s*)?Purpose\b"),
    "fee_status": _label(r"\bFee\s*St[a]?tus\b"),
    "waiver_code": _label(r"\bWa[il]ver\s*Code\b"),
    "registry_status": _label(r"\bRegistry\s*Status\b"),
    "biometric_confidence": _label(r"\bBiometric\s*confidence\b"),
}

# Fuzzy page-header cues: (page type, reference phrase, threshold).
HEADER_CUES = [
    ("fee", "MIB Fee Receipt", 75),
    ("registry", "Planetary Registry Extract", 75),
    ("intake", "FORM I-8090 Extraterrestrial Work Authorization Intake", 70),
    ("biometric", "FORM B-13 Biometric Scan Slip", 70),
    ("sponsor", "Sponsor Attestation Letter", 75),
    ("note", "Manual Adjudicator Note", 75),
]

DECISION_WORDS = {"APPROVED": "APPROVED", "DENIED": "DENIED", "REVIEW": "NEEDS_REVIEW", "NEEDS_REVIEW": "NEEDS_REVIEW"}


@dataclass
class PageDoc:
    index: int
    ptype: str
    source: str  # "text" | "ocr"
    text: str
    fields: dict = field(default_factory=dict)
    corrections: dict = field(default_factory=dict)
    observed_flags: set = field(default_factory=set)
    findings: list = field(default_factory=list)  # [(decision, reason)]
    ocr_conf: float | None = None
    packet_id: str | None = None
    decoy: bool = False  # COPY ARTIFACT duplicate for another applicant


def classify(text: str) -> str:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    head = " ".join(lines[:4])[:200]
    best, best_score = "unknown", 0.0
    for ptype, ref, thresh in HEADER_CUES:
        score = fuzz.partial_ratio(ref.lower(), head.lower())
        if score >= thresh and score > best_score:
            best, best_score = ptype, score
    if best != "unknown":
        return best
    # Field-level cues when the header is destroyed.
    low = text.lower()
    if re.search(r"fee\s*st[a]?tus|waiver\s*code|fee\s*receip", low):
        return "fee"
    if re.search(r"species\s*match|biometric|observed\s*[fl]l?ags", low):
        return "biometric"
    if re.search(r"registry\s*(status|extract)", low):
        return "registry"
    if "attests" in low or "attestation" in low:
        return "sponsor"
    if re.search(r"ad[ji]udicat\w*\s*note|finding:", low):
        return "note"
    if re.search(r"i-?8090|intake", low):
        return "intake"
    return "unknown"


def _clean_value(val: str) -> str:
    for w in STAMP_WORDS:
        idx = val.upper().find(w.upper())
        if idx >= 0:
            val = val[:idx]
    # Strip trailing OCR junk: pipes, brackets, stray punctuation runs.
    val = re.sub(r"[|\[\]{}<>~_=]+", " ", val)
    val = re.sub(r"\s{2,}", " ", val).strip(" .,:;-")
    return val.strip()


def _fuzzy_findings(text: str) -> list:
    """Recover Finding decisions from garbled OCR of note pages."""
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or len(line) > 80:
            continue
        for tok in re.split(r"[^A-Za-z_]+", line):
            if len(tok) < 5:
                continue
            # Stamp words are typeset uppercase; require the OCR token to be
            # mostly uppercase before fuzzy-matching at a permissive level.
            upper_frac = sum(c.isupper() for c in tok) / len(tok)
            if upper_frac < 0.7:
                continue
            up = tok.upper()
            for word, decision in DECISION_WORDS.items():
                if up == word or fuzz.ratio(up, word) >= 66:
                    out.append((decision, ""))
                    break
    return out


def parse_page(index: int, text: str, source: str, ocr_conf: float | None = None) -> PageDoc:
    ptype = classify(text)
    pd = PageDoc(index=index, ptype=ptype, source=source, text=text, ocr_conf=ocr_conf)

    if re.search(r"COPY\s*ARTIFACT", text, re.I):
        pd.decoy = True

    m = re.search(r"Packet\s+(MIB[-\s]?\d{6})", text)
    if m:
        pd.packet_id = "MIB-" + re.sub(r"\D", "", m.group(1))[-6:]

    for key, pat in L.items():
        mm = pat.search(text)
        if mm:
            val = _clean_value(mm.group(1))
            if val and not val.startswith("["):
                pd.fields[key] = val

    for who, val in CORRECTION_RE.findall(text):
        key = {
            "applicant": "applicant_name",
            "sponsor": "sponsor_id",
            "visa class": "visa_class",
            "fee status": "fee_status",
            "arrival date": "arrival_date",
            "species": "species_code",
            "home world": "home_world",
        }[who.lower()]
        pd.corrections[key] = _clean_value(val)

    for m in OBSERVED_FLAGS_RE.finditer(text):
        for tok in re.split(r"[|,\s]+", m.group(1)):
            f = extract.snap_flag(tok)
            if f:
                pd.observed_flags.add(f)

    if ptype == "sponsor":
        m = SPONSOR_ATTEST_RE.search(text)
        if m:
            pd.fields["sponsor_id"] = m.group(1).replace(" ", "-").upper()
            pd.fields["applicant_name"] = _clean_value(m.group(2))
            pd.fields["declared_purpose"] = _clean_value(m.group(3))
        else:
            pd.fields.pop("case_id", None)  # letter prose confuses the label regex
        m = CLASS_COMPLIANCE_RE.search(text)
        if m:
            pd.fields["visa_class"] = m.group(1).replace(" ", "-").upper()

    if ptype == "note":
        for decision, reason in FINDING_RE.findall(text):
            pd.findings.append((decision, (reason or "").strip()))
        if not pd.findings and source == "ocr":
            pd.findings.extend(_fuzzy_findings(text))
        for m in DISQ_REASON_RE.finditer(text):
            f = extract.snap_flag(m.group(1))
            if f:
                pd.observed_flags.add(f)

    return pd
