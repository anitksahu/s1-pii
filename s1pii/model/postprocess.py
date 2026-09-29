"""Deterministic validator channel and cross-mention propagation (ablations A3 and A4)."""
from __future__ import annotations

import re
from dataclasses import replace

from ..schema import Doc, Span, EMAIL, PHONE, ACCOUNT_NUMBER, PERSON

_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,4})?\b")
_SSN = re.compile(r"(?<!\d)(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}(?!\d)")
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_ABA = re.compile(r"(?<!\d)\d{9}(?!\d)")
_ROUTING_CUE = re.compile(r"routing|aba|transit", re.I)
_HONORIFIC = re.compile(r"(?:\b(?:mr|mrs|ms|miss|dr|prof|sir|madam)\.?\s+)$", re.I)


def luhn_ok(digits: str) -> bool:
    s, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        s += d
        alt = not alt
    return s % 10 == 0


def iban_ok(s: str) -> bool:
    s = s.replace(" ", "")
    if not 15 <= len(s) <= 34:
        return False
    r = s[4:] + s[:4]
    num = "".join(str(int(c, 36)) for c in r)
    return int(num) % 97 == 1


def aba_ok(d: str) -> bool:
    w = [3, 7, 1] * 3
    return sum(int(c) * k for c, k in zip(d, w)) % 10 == 0


def validator_spans(doc: Doc, source: str = "validator") -> list[Span]:
    t, out = doc.text, []
    def add(a, b, lab, raw):
        out.append(Span(doc.doc_id, a, b, lab, label_raw=raw, score=1.0, source=source, surface=t[a:b]))
    for m in _CARD.finditer(t):
        digits = re.sub(r"\D", "", m.group())
        if 13 <= len(digits) <= 19 and luhn_ok(digits):
            add(m.start(), m.end(), ACCOUNT_NUMBER, "card")
    for m in _IBAN.finditer(t):
        if iban_ok(m.group()):
            add(m.start(), m.end(), ACCOUNT_NUMBER, "iban")
    for m in _SSN.finditer(t):
        add(m.start(), m.end(), ACCOUNT_NUMBER, "ssn")
    for m in _EMAIL.finditer(t):
        add(m.start(), m.end(), EMAIL, "email")
    for m in _ABA.finditer(t):
        if aba_ok(m.group()) and _ROUTING_CUE.search(t[max(0, m.start() - 40):m.start()]):
            add(m.start(), m.end(), ACCOUNT_NUMBER, "aba")
    try:
        import phonenumbers
        for m in phonenumbers.PhoneNumberMatcher(t, "US", leniency=phonenumbers.Leniency.VALID):
            add(m.start, m.end, PHONE, "phone")
    except ImportError:
        pass
    return out


def _name_cue(text: str, a: int, surface: str) -> bool:
    return (" " in surface.strip()) or bool(_HONORIFIC.search(text[max(0, a - 8):a]))


def propagate(doc: Doc, preds: list[Span], min_score: float = 0.5, min_len: int = 4) -> list[Span]:
    """Copy confident predictions to other exact, case-sensitive, word-bounded occurrences in the
    same document. PERSON mentions propagate only to capitalized occurrences that are
    multi-word or follow an honorific. Propagated spans keep the source score and are marked
    ``label_raw='propagated'`` so their precision can be measured separately (C2)."""
    covered = [(p.start, p.end) for p in preds]
    new = []
    for p in sorted(preds, key=lambda s: -s.score):
        if p.score < min_score:
            continue
        surf = doc.text[p.start:p.end]
        if sum(c.isalnum() for c in surf) < min_len:
            continue
        for m in re.finditer(r"(?<!\w)" + re.escape(surf) + r"(?!\w)", doc.text):
            a, b = m.start(), m.end()
            if any(x < b and y > a for x, y in covered):
                continue
            if p.label_canonical == PERSON and not (surf[:1].isupper() and _name_cue(doc.text, a, surf)):
                continue
            new.append(replace(p, start=a, end=b, label_raw="propagated", source=p.source + ":prop", surface=surf))
            covered.append((a, b))
    return preds + new
