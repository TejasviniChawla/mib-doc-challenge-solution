"""MIB adjudication policy engine.

Encodes the public FIELD_MANUAL rules plus edge rules mined from the 1,000
labeled training cases (validated at 97.2% on gold fields; residual errors are
cases whose NEEDS_REVIEW status comes from packet-level evidence quality,
handled by the caller via `evidence_problem`).

Mined rules beyond the public manual:
- Additional revoked sponsors: SPN-2718, SPN-7331, SPN-9090 (same DIP-1
  exemption signature as the three published ones).
- Eris Relay and TRAPPIST-1e are fully embargoed: always carry the
  planetary_embargo flag and always deny.
- Wolf-1061c is embargoed for all visa classes except DIP-1, usually without
  the planetary_embargo flag being set.
- Revoked sponsor and stale arrival date only deny non-DIP-1 cases.
"""

from datetime import date, timedelta

from solution.vocab import DISQUALIFYING_FLAGS, REVIEW_FLAGS

REVOKED_SPONSORS = {
    "SPN-0007",
    "SPN-0139",
    "SPN-4040",
    "SPN-2718",
    "SPN-7331",
    "SPN-9090",
}

EMBARGO_FULL = {"Eris Relay", "TRAPPIST-1e"}
EMBARGO_NON_DIP = {"Wolf-1061c"}

STALE_DAYS = 180


def is_stale(arrival_date: str | None, receipt_date: str | None) -> bool | None:
    """True if arrival is more than 180 days before packet receipt.

    Returns None when it cannot be determined (missing/unparseable dates).
    Falls back to a conservative fixed cutoff when no receipt date is known:
    every training arrival before 2026-01-28 was stale for non-DIP cases.
    """
    if not arrival_date:
        return None
    try:
        arrival = date.fromisoformat(arrival_date)
    except ValueError:
        return None
    if receipt_date:
        try:
            receipt = date.fromisoformat(receipt_date)
            return arrival < receipt - timedelta(days=STALE_DAYS)
        except ValueError:
            pass
    return arrival < date(2026, 1, 28)


def adjudicate(fields: dict) -> tuple[str, str]:
    """Decide APPROVED / DENIED / NEEDS_REVIEW from extracted fields.

    Returns (decision, rule_name) — the rule name feeds confidence
    calibration.

    `fields` keys: visa_class, fee_status, sponsor_id, home_world,
    arrival_date, risk_flags (set), receipt_date (optional),
    evidence_problem (bool: packet illegible/contradictory/hidden-only values),
    fee_waiver_visible (bool: a visible hardship-waiver note exists).
    """
    flags = set(fields.get("risk_flags") or ())
    visa = fields.get("visa_class") or ""
    fee = fields.get("fee_status") or ""
    home = fields.get("home_world") or ""
    sponsor = fields.get("sponsor_id") or ""
    dip = visa == "DIP-1"

    # Hard denials — trusted disqualifying evidence beats everything else.
    if flags & DISQUALIFYING_FLAGS:
        return "DENIED", "disq_flag"
    if home in EMBARGO_FULL:
        return "DENIED", "embargo_world"
    if visa == "TRANSIT-7":
        return "DENIED", "transit"
    if fee == "unpaid" and not (fields.get("fee_waiver_visible") and not dip):
        return "DENIED", "unpaid"
    if home in EMBARGO_NON_DIP and not dip:
        return "DENIED", "wolf_embargo"
    if sponsor in REVOKED_SPONSORS and not dip:
        return "DENIED", "revoked_sponsor"
    stale = is_stale(fields.get("arrival_date"), fields.get("receipt_date"))
    if stale and not dip:
        return "DENIED", "stale_date"

    # Review conditions.
    if fee == "unknown":
        return "NEEDS_REVIEW", "fee_unknown"  # receipt literally says unknown
    if not fee:
        return "NEEDS_REVIEW", "fee_missing"  # receipt absent or unreadable
    if flags & REVIEW_FLAGS:
        return "NEEDS_REVIEW", "review_flag"
    if fields.get("evidence_problem"):
        return "NEEDS_REVIEW", "evidence_problem"
    if not fields.get("arrival_date"):
        return "NEEDS_REVIEW", "no_arrival"
    if not sponsor and not dip:
        return "NEEDS_REVIEW", "no_sponsor"

    return "APPROVED", "clean"
