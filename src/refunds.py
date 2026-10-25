"""Refund timelines read from the knowledge base (REF-2) at runtime + deterministic date arithmetic.
Nothing is hard-coded: if the policy document changes, the answers change with it."""
import re
from datetime import date
from functools import lru_cache

from . import config, datetools


@lru_cache(maxsize=1)
def _table():
    from .kb import load_chunks
    sec = next((c for c in load_chunks() if c.cid == "REF-2"), None)
    rows = []
    for line in (sec.text.splitlines() if sec else []):
        if not line.startswith("|") or set(line.replace("|", "").strip()) <= set("- "):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 2 or cells[0].lower().startswith("payment method"):
            continue
        timing = cells[1]
        if "instant" in timing.lower():
            lo = hi = 0
        else:
            m = re.search(r"(\d+)\s*to\s*(\d+)\s*business", timing)
            if not m:
                continue
            lo, hi = int(m[1]), int(m[2])
        rows.append((cells[0].lower(), lo, hi, timing))
    return rows


def _row_for(method):
    s = re.sub(r"[_\-]", " ", (method or "").lower())
    cod = "cod" in s or "cash" in s
    find = lambda pred: next((r for r in _table() if pred(r[0])), None)
    if "upi" in s:
        return find(lambda n: "upi" in n)
    if "card" in s or "credit" in s or "debit" in s:
        return find(lambda n: "card" in n)
    if "net" in s and "bank" in s:
        return find(lambda n: "net banking" in n)
    if "wallet" in s:
        return find(lambda n: "wallet" in n and (("cash" in n) == cod))
    if "bank" in s or "neft" in s or "transfer" in s:
        return find(lambda n: "bank transfer" in n or "neft" in n)
    return None


def _fmt(d):
    return f"{d.day} {d.strftime('%B %Y')}"


def refund_window(payment_method, start_date):
    row = _row_for(payment_method)
    if not row:
        return {"error": f"payment method '{payment_method}' not found in REF-2; ask the customer how they paid"}
    try:
        start = date.fromisoformat(str(start_date)[:10])
    except Exception:
        return {"error": "start_date must be an ISO date like 2026-10-17"}
    name, lo, hi, timing = row
    if hi == 0:
        return {"method": name, "timeline": timing, "instant": True, "start": _fmt(start)}
    e, l = datetools.add_business_days(start, lo), datetools.add_business_days(start, hi)
    return {"method": name, "timeline": timing, "start": _fmt(start), "earliest": e.isoformat(), "latest": l.isoformat(),
            "answer": f"between {_fmt(e)} and {_fmt(l)}"}


def order_checks(order):
    """Refund-overdue checks for a verified order (ESC-1: more than 3 business days past the maximum timeline)."""
    out = []
    for pre, label in (("return", "return refund"), ("cancellation", "cancellation refund"), ("rto", "undelivered-parcel refund")):
        init, method = order.get(f"{pre}_refund_initiated_on"), order.get(f"{pre}_refund_method")
        if not init or not method or (pre == "return" and order.get("return_refund_completed_on")):
            continue
        w = refund_window(method, init)
        if "error" in w:
            continue
        latest = date.fromisoformat(w["latest"]) if "latest" in w else date.fromisoformat(str(init)[:10])
        past = datetools.business_days_between(latest, config.TODAY)
        out.append({"refund": label, "initiated_on": str(init)[:10], "method": w["method"], "timeline": w["timeline"],
                    "should_have_arrived_by": latest.isoformat(), "business_days_past_maximum": past,
                    "ESC-1 refund-overdue rule applies (create_ticket now and give the RRN)": past > 3,
                    "rrn": order.get(f"{pre}_rrn")})
    return out
