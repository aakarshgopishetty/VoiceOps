"""Deterministic date maths: business days = Mon-Sat excluding holidays.csv; return windows = calendar days."""
import csv
from datetime import date, timedelta

from . import config

_HOL = None


def holidays():
    global _HOL
    if _HOL is None:
        with open(config.DATA_DIR / "holidays.csv", encoding="utf-8") as f:
            _HOL = {date.fromisoformat(r["date"]) for r in csv.DictReader(f)}
    return _HOL


def is_business_day(d):
    return d.weekday() != 6 and d not in holidays()


def add_business_days(start, n):
    d = start
    while n > 0:
        d += timedelta(days=1)
        if is_business_day(d):
            n -= 1
    return d


def business_days_between(start, end):
    """Business days elapsed after `start` up to and including `end` (0 if end <= start)."""
    n, d = 0, start
    while d < end:
        d += timedelta(days=1)
        if is_business_day(d):
            n += 1
    return n


def date_math(op, a=None, b=None, n=None):
    a = date.fromisoformat(a) if a else config.TODAY
    if op == "add_business_days":
        r = add_business_days(a, int(n)); return {"result": r.isoformat(), "weekday": r.strftime("%A")}
    if op == "add_calendar_days":
        r = a + timedelta(days=int(n)); return {"result": r.isoformat(), "weekday": r.strftime("%A")}
    b = date.fromisoformat(b) if b else config.TODAY
    if op == "business_days_between":
        return {"business_days": business_days_between(a, b)}
    if op == "calendar_days_between":
        return {"calendar_days": (b - a).days}
    return {"error": f"unknown op {op}"}
