"""Seeded synthetic support conversations (chat and ASR-style) with exact character spans.

Covers what the public training sets lack: multi-turn dialogue, spelled-out digits,
filler words, identifiers split across turns, and PII-free decoy conversations. Every
entity is inserted through ``Builder.pii`` so its offsets are exact by construction.
"""
from __future__ import annotations

import random
from datetime import date as _date

from ..schema import (Doc, Span, PERSON, ADDRESS, EMAIL, PHONE, URL, DATE, ACCOUNT_NUMBER,
                      SECRET, OTHER_PII, validate_doc)

SYNTH_VERSION = "synth-v0.2"   # v0.2: dates of birth from a fixed range (v0.1 was relative to the run date)
_DIG = "zero one two three four five six seven eight nine".split()


class Builder:
    def __init__(self, doc_id: str):
        self.doc_id, self.parts, self.spans, self.pos = doc_id, [], [], 0

    def text(self, s: str) -> "Builder":
        self.parts.append(s); self.pos += len(s); return self

    def pii(self, s: str, label: str) -> "Builder":
        self.spans.append(Span(self.doc_id, self.pos, self.pos + len(s), label, label_raw=label.lower(),
                               source="synth", surface=s))
        return self.text(s)

    def doc(self, cluster: str) -> Doc:
        return validate_doc(Doc(self.doc_id, "".join(self.parts), tuple(self.spans), "synthetic_conv", "train",
                                cluster_id=cluster, meta={"synth_version": SYNTH_VERSION}))


def spell(digits: str, rng: random.Random) -> str:
    words = [_DIG[int(c)] if c.isdigit() else c for c in digits if c.isalnum()]
    if rng.random() < 0.3:
        words.insert(rng.randrange(1, max(2, len(words))), "uh")
    return " ".join(words)


def conversation(i: int, fk, rng: random.Random) -> Doc:
    b = Builder(f"synth_{i:07d}")
    asr = rng.random() < 0.35
    first, last = fk.first_name(), fk.last_name()
    name = f"{first} {last}"
    kind = rng.choice(["card", "account", "address", "login", "decoy", "dob"])
    b.text("User: " + rng.choice(["Hi, ", "Hello, ", "Hey there, ", ""]))
    if kind == "decoy":
        b.text(rng.choice(["what are your branch hours on Saturday?", "how do I enable dark mode in the app?",
                           "is the mobile deposit limit different for business accounts?"]) + "\n")
        b.text("Agent: " + rng.choice(["Our branches open at 9am.", "Go to Settings, then Display.",
                                       "Business limits are listed on the fees page."]) + "\n")
        return b.doc(f"synth:{kind}")
    b.text("my name is ").pii(name, PERSON).text(".\n")
    b.text("Agent: Thanks ").pii(first, PERSON).text(". How can I help today?\n")
    if kind == "card":
        card = fk.credit_card_number()
        b.text("User: my card ending ")
        last4 = card[-4:]
        b.pii(spell(last4, rng) if asr else last4, ACCOUNT_NUMBER).text(" was charged twice.\n")
        b.text("Agent: Please confirm the full card number.\nUser: it's ")
        if asr and len(card) >= 12:
            b.pii(spell(card[:8], rng), ACCOUNT_NUMBER).text("\nAgent: go on\nUser: ")
            b.pii(spell(card[8:], rng), ACCOUNT_NUMBER).text("\n")
        else:
            b.pii(card, ACCOUNT_NUMBER).text("\n")
        b.text("Agent: And the security code?\nUser: ").pii(fk.credit_card_security_code(), SECRET).text("\n")
    elif kind == "account":
        acct = fk.bban()
        b.text("User: account number ").pii(spell(acct, rng) if asr else acct, ACCOUNT_NUMBER)
        b.text(", and you can reach me at ").pii(fk.phone_number(), PHONE).text(".\n")
        b.text("Agent: I'll email the statement to ").pii(fk.email(), EMAIL).text(", is that right?\n")
    elif kind == "address":
        b.text("User: I moved to ").pii(fk.street_address(), ADDRESS).text(" last month.\n")
        b.text("Agent: Updated. Your profile page is ").pii(fk.url() + fk.user_name(), URL).text("\n")
    elif kind == "login":
        b.text("User: I can't log in, my username is ").pii(fk.user_name(), OTHER_PII)
        b.text(" and I tried ").pii(fk.password(length=10), SECRET).text(".\n")
        b.text("Agent: I see logins from ").pii(fk.ipv4(), OTHER_PII).text(".\n")
    elif kind == "dob":
        dob = fk.date_between_dates(_date(1940, 1, 1), _date(2006, 12, 31)).strftime(rng.choice(["%B %d, %Y", "%m/%d/%Y", "%d %b %Y"]))
        b.text("Agent: To verify, what's your date of birth?\nUser: ").pii(dob, DATE).text("\n")
        b.text("Agent: Thank you, ").pii(first, PERSON).text(".\n")
    return b.doc(f"synth:{kind}:{'asr' if asr else 'chat'}")


def generate(n: int, seed: int = 0) -> list[Doc]:
    from faker import Faker
    fk = Faker("en_US")
    fk.seed_instance(seed)
    rng = random.Random(seed)
    return [conversation(i, fk, rng) for i in range(n)]
