"""v2 label sets: labels are text supplied at inference, organised in a type hierarchy.

A ``Label`` is a name, an optional description and a ``sensitive`` flag. A ``LabelSet``
(the L of the plan) is what a caller passes at inference; its hash enters every cache key.

The hierarchy (``TREE``) serves two purposes:
* training: a label is never used as a negative for a span whose gold label is an
  ancestor, descendant or synonym of it (``compatible``), so umbrella labels such as
  ``unique_id`` or ``other_pii`` never teach the model that an SSN is *not* an identifier;
* inference: ``hier_decide`` backs off from a leaf to the deepest ancestor whose summed
  probability clears ``tau`` (hierarchical typed decision with type-level abstention).
Labels outside the tree (arbitrary user labels) hang directly under ``pii``.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .. import taxonomy as tx
from ..schema import IGNORE, NOT_PII, OTHER_PII

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"

# ------------------------------------------------------------------ hierarchy
# node -> parent. Leaves carry the fine distinctions the typing head must learn.
TREE: dict[str, str | None] = {
    "pii": None,
    "person_name": "pii", "first_name": "person_name", "last_name": "person_name",
    "contact": "pii", "email": "contact", "phone": "contact", "fax": "phone", "url": "contact",
    "location": "pii", "street_address": "location", "city": "location", "state": "location",
    "county": "location", "country": "location", "postcode": "location", "coordinate": "location",
    "date": "pii", "date_of_birth": "date", "time": "date",
    "identifier": "pii",
    "government_id": "identifier", "ssn": "government_id", "national_id": "government_id",
    "tax_id": "government_id", "passport_number": "government_id",
    "driver_license_number": "government_id", "certificate_license_number": "government_id",
    "financial_id": "identifier", "bank_account": "financial_id", "iban": "bank_account",
    "bban": "bank_account", "payment_card": "financial_id", "routing_number": "financial_id",
    "swift_bic": "financial_id",
    "customer_id": "identifier", "employee_id": "identifier", "medical_record_number": "identifier",
    "health_plan_beneficiary_number": "identifier", "vehicle_identifier": "identifier",
    "license_plate": "identifier", "device_identifier": "identifier",
    "secret": "pii", "password": "secret", "pin": "secret", "card_security_code": "secret",
    "api_key": "secret", "http_cookie": "secret",
    "online_id": "pii", "user_name": "online_id", "ip_address": "online_id", "ipv4": "ip_address",
    "ipv6": "ip_address", "mac_address": "online_id",
    "organization": "pii",
    "attribute": "pii", "occupation": "attribute", "education_level": "attribute",
    "employment_status": "attribute", "blood_type": "attribute", "race_ethnicity": "attribute",
    "religious_belief": "attribute", "sexuality": "attribute", "political_view": "attribute",
    "gender": "attribute", "age": "attribute", "language": "attribute",
    "biometric_identifier": "attribute", "quantity": "attribute", "demographic": "attribute",
}

# raw label (lowercased) -> (tree node, description, paraphrases)
NATIVE: dict[str, tuple[str, str, tuple[str, ...]]] = {
    # people
    "name": ("person_name", "full or partial name of a person", ("person name", "individual's name")),
    "person": ("person_name", "name of a specific person", ("personal name", "someone's name")),
    "private_person": ("person_name", "name of a private individual", ("private individual", "person's name")),
    "first_name": ("first_name", "given name of a person", ("given name", "forename")),
    "last_name": ("last_name", "family name of a person", ("surname", "family name")),
    # contact
    "email": ("email", "email address", ("e-mail address", "electronic mail address")),
    "private_email": ("email", "email address of a private individual", ("personal email", "email")),
    "phone_number": ("phone", "telephone number", ("phone", "contact number")),
    "phone": ("phone", "telephone number", ("phone number", "mobile number")),
    "phone_num": ("phone", "telephone number", ("phone number", "cell number")),
    "private_phone": ("phone", "telephone number of a private individual", ("personal phone", "phone number")),
    "fax_number": ("fax", "fax number", ("facsimile number", "fax")),
    "url": ("url", "web address or link", ("website", "link")),
    "private_url": ("url", "web address tied to a private individual", ("personal website", "profile link")),
    # location
    "address": ("location", "postal address or part of one", ("mailing address", "residential address")),
    "private_address": ("location", "address of a private individual", ("home address", "residence")),
    "loc": ("location", "location such as an address, city or country", ("place", "geographic location")),
    "street_address": ("street_address", "street name and house number", ("street", "house address")),
    "city": ("city", "name of a city or town", ("town", "municipality")),
    "state": ("state", "state or province", ("province", "region")),
    "county": ("county", "county or district", ("district", "administrative area")),
    "country": ("country", "name of a country", ("nation", "country name")),
    "postcode": ("postcode", "postal or zip code", ("zip code", "postal code")),
    "coordinate": ("coordinate", "geographic coordinates", ("latitude and longitude", "gps position")),
    "local_latlng": ("coordinate", "latitude and longitude pair", ("gps coordinates", "lat long")),
    # dates
    "date": ("date", "calendar date", ("day", "date value")),
    "private_date": ("date", "date tied to a private individual", ("personal date", "date")),
    "datetime": ("date", "date or time expression", ("time expression", "date")),
    "date_time": ("date", "date together with a time", ("timestamp", "date and time")),
    "time": ("time", "time of day", ("clock time", "hour")),
    "date_of_birth": ("date_of_birth", "date on which a person was born", ("birth date", "birthday")),
    # identifiers
    "unique_id": ("identifier", "unique identifier assigned to a person or record", ("id", "identifier")),
    "id_num": ("identifier", "identification number", ("id number", "reference number")),
    "code": ("identifier", "code or number identifying a person, case or record", ("identifier code", "reference code")),
    "ssn": ("ssn", "US social security number", ("social security number", "SSN")),
    "national_id": ("national_id", "national identity card number", ("national identity number", "citizen id")),
    "tax_id": ("tax_id", "tax identification number", ("taxpayer id", "TIN")),
    "passport_number": ("passport_number", "passport number", ("passport id", "travel document number")),
    "driver_license_number": ("driver_license_number", "driver's license number", ("driving licence number", "DL number")),
    "certificate_license_number": ("certificate_license_number", "number of a professional license or certificate", ("license number", "certificate number")),
    "account_number": ("bank_account", "bank or financial account number", ("account no", "bank account")),
    "iban": ("iban", "international bank account number", ("IBAN", "international account number")),
    "bban": ("bban", "basic bank account number", ("BBAN", "domestic account number")),
    "credit_card_number": ("payment_card", "credit or debit card number", ("card number", "payment card")),
    "credit_debit_card": ("payment_card", "credit or debit card number", ("card number", "credit card")),
    "bank_routing_number": ("routing_number", "bank routing number", ("ABA number", "routing code")),
    "routing_number": ("routing_number", "bank routing number", ("ABA routing number", "transit number")),
    "swift_bic": ("swift_bic", "SWIFT or BIC bank code", ("BIC", "SWIFT code")),
    "swift_bic_code": ("swift_bic", "SWIFT or BIC bank code", ("bank identifier code", "SWIFT")),
    "customer_id": ("customer_id", "customer identifier", ("client id", "customer number")),
    "employee_id": ("employee_id", "employee identifier", ("staff id", "employee number")),
    "medical_record_number": ("medical_record_number", "medical record number", ("MRN", "patient record number")),
    "health_plan_beneficiary_number": ("health_plan_beneficiary_number", "health insurance member number", ("insurance member id", "beneficiary number")),
    "vehicle_identifier": ("vehicle_identifier", "vehicle identification number", ("VIN", "chassis number")),
    "license_plate": ("license_plate", "vehicle license plate", ("number plate", "registration plate")),
    "device_identifier": ("device_identifier", "identifier of a device such as IMEI or serial number", ("device id", "serial number")),
    # secrets
    "secret": ("secret", "password, key or other secret credential", ("credential", "secret value")),
    "password": ("password", "account password", ("passcode", "login password")),
    "pin": ("pin", "personal identification number", ("PIN code", "pin number")),
    "account_pin": ("pin", "PIN of an account", ("account PIN", "access pin")),
    "cvv": ("card_security_code", "card verification value", ("CVV", "security code")),
    "credit_card_security_code": ("card_security_code", "card security code", ("CVC", "card verification code")),
    "api_key": ("api_key", "API key or access token", ("access token", "secret key")),
    "http_cookie": ("http_cookie", "HTTP cookie value", ("session cookie", "cookie")),
    # online identifiers
    "user_name": ("user_name", "account user name", ("username", "login name")),
    "username": ("user_name", "account user name", ("user handle", "login")),
    "ipv4": ("ipv4", "IPv4 address", ("IP address", "internet protocol address")),
    "ipv6": ("ipv6", "IPv6 address", ("IPv6", "internet protocol v6 address")),
    "mac_address": ("mac_address", "MAC hardware address", ("hardware address", "MAC")),
    # organisations and attributes
    "org": ("organization", "name of an organization", ("organisation", "institution")),
    "company": ("organization", "company name", ("business", "employer")),
    "company_name": ("organization", "name of a company", ("business name", "firm")),
    "occupation": ("occupation", "job title or occupation", ("profession", "job")),
    "education_level": ("education_level", "level of education", ("degree", "schooling")),
    "employment_status": ("employment_status", "employment status", ("job status", "work status")),
    "blood_type": ("blood_type", "blood group", ("blood group", "ABO type")),
    "race_ethnicity": ("race_ethnicity", "race or ethnicity", ("ethnic background", "ethnicity")),
    "religious_belief": ("religious_belief", "religion or belief", ("faith", "religion")),
    "sexuality": ("sexuality", "sexual orientation", ("orientation", "sexual identity")),
    "political_view": ("political_view", "political opinion or affiliation", ("political affiliation", "politics")),
    "gender": ("gender", "gender of a person", ("sex", "gender identity")),
    "age": ("age", "age of a person", ("years old", "person's age")),
    "language": ("language", "language a person speaks", ("spoken language", "mother tongue")),
    "biometric_identifier": ("biometric_identifier", "biometric data such as fingerprint or face template", ("biometric", "fingerprint id")),
    "dem": ("demographic", "demographic attribute such as nationality, age or profession", ("demographic", "personal attribute")),
    "quantity": ("quantity", "quantity that could identify a person", ("amount", "number")),
    "misc": ("pii", "other identifying information", ("miscellaneous identifier", "other personal detail")),
    "other_pii": ("pii", "other personally identifying information", ("personal information", "other identifier")),
}

# canonical fallbacks so arbitrary labels still produce admissible prediction types
_NODE_CANONICAL = {
    "person_name": "PERSON", "email": "EMAIL", "phone": "PHONE", "url": "URL", "location": "ADDRESS",
    "date": "DATE", "identifier": "ACCOUNT_NUMBER", "secret": "SECRET",
}


def node_of(name: str) -> str:
    return NATIVE.get(name.lower(), ("pii",))[0]


def ancestors(node: str) -> list[str]:
    out = []
    while node is not None:
        out.append(node)
        node = TREE.get(node)
    return out


def compatible(a: str, b: str) -> bool:
    """True if one label's node is an ancestor of (or equal to) the other's."""
    na, nb = node_of(a), node_of(b)
    return na in ancestors(nb) or nb in ancestors(na)


def canonical_of(name: str) -> str:
    for n in ancestors(node_of(name)):
        if n in _NODE_CANONICAL:
            return _NODE_CANONICAL[n]
    return OTHER_PII


@dataclass(frozen=True)
class Label:
    name: str
    description: str = ""
    sensitive: bool = True
    canonical: str = ""          # admissible prediction type used for scoring
    node: str = ""               # hierarchy node; "" = derived from NATIVE or "pii"

    def text(self, with_description: bool = True) -> str:
        n = self.name.replace("_", " ")
        return f"{n}: {self.description}" if with_description and self.description else n

    def resolved_node(self) -> str:
        return self.node or node_of(self.name)

    def resolved_canonical(self) -> str:
        return self.canonical or canonical_of(self.name)


@dataclass
class LabelSet:
    name: str
    labels: list[Label] = field(default_factory=list)
    descriptions: bool = True

    def texts(self) -> list[str]:
        return [l.text(self.descriptions) for l in self.labels]

    def hash(self) -> str:
        return hashlib.sha256(json.dumps([self.descriptions, [asdict(l) for l in self.labels]],
                                         sort_keys=True).encode()).hexdigest()[:16]

    def sensitive_mask(self) -> list[bool]:
        return [l.sensitive for l in self.labels]

    def to_yaml(self, path: Path) -> None:
        import yaml
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump({"name": self.name, "descriptions": self.descriptions,
                                        "labels": [asdict(l) for l in self.labels]}, sort_keys=False))

    @classmethod
    def from_yaml(cls, path: Path) -> "LabelSet":
        import yaml
        d = yaml.safe_load(Path(path).read_text())
        return cls(d["name"], [Label(**x) for x in d["labels"]], d.get("descriptions", True))

    @classmethod
    def from_names(cls, names: list[str], name: str = "adhoc", descriptions: bool = True) -> "LabelSet":
        """Build L from bare names or "name: description" strings (the inference interface)."""
        out = []
        for s in names:
            n, _, desc = s.partition(":")
            n = n.strip()
            out.append(Label(n, desc.strip() or NATIVE.get(n.lower(), ("", ""))[1]))
        return cls(name, out, descriptions)


def native_label(raw: str, sensitive: bool = True) -> Label:
    key = raw.lower()
    if key not in NATIVE:
        raise KeyError(f"no description for native label {raw!r}; add it to v2.labels.NATIVE")
    node, desc, _ = NATIVE[key]
    return Label(key, desc, sensitive, canonical_of(key), node)


def paraphrases(raw: str) -> tuple[str, ...]:
    return NATIVE[raw.lower()][2]


# ------------------------------------------------------------------ benchmark label sets
# Mechanical: every raw label of the benchmark that the taxonomy maps to a PII type becomes a
# sensitive label. TAB keeps the v1 DIRECT tier: all entity types are PII when DIRECT.
BENCHMARK_MAPS = {"pii_trace": "pii_trace", "nemotron": "nemotron", "spy_legal": "spy",
                  "spy_medical": "spy", "gretel": "gretel"}


def benchmark_label_set(dataset: str) -> LabelSet:
    if dataset.startswith("tab"):
        raws = sorted(tx.TAB_ENTITY)
        labs = [Label(r.lower(), NATIVE[r.lower()][1], True, tx.TAB_ENTITY[r]) for r in raws]
        return LabelSet(dataset, labs)
    table = tx.MAPS[BENCHMARK_MAPS[dataset]]
    seen, labs = set(), []
    for raw, canon in sorted(table.items(), key=lambda kv: kv[0].lower()):
        k = raw.lower()
        if canon in (IGNORE, NOT_PII) or k in seen:
            continue
        seen.add(k)
        labs.append(Label(k, NATIVE[k][1], True, canon, NATIVE[k][0]))
    return LabelSet(dataset, labs)


def canonical_label_set() -> LabelSet:
    """Primary C0' query set: exactly the canonical label strings of ``configs/labels.yaml``
    that every baseline received, with their descriptions (same information on both sides)."""
    import yaml
    cfg = yaml.safe_load((CONFIG_DIR / "labels.yaml").read_text())
    node = {"PERSON": "person_name", "ADDRESS": "location", "EMAIL": "email", "PHONE": "phone", "URL": "url",
            "DATE": "date", "ACCOUNT_NUMBER": "identifier", "SECRET": "secret", "OTHER_PII": "pii"}
    return LabelSet("canonical", [Label(k.replace(" ", "_"), v["description"], True, v["type"], node[v["type"]])
                                  for k, v in cfg.items()])


def write_benchmark_label_sets(datasets=("pii_trace", "tab_direct", "spy_legal", "spy_medical", "nemotron"),
                               out: Path | None = None) -> dict[str, str]:
    out = out or CONFIG_DIR / "benchmark_labels"
    hashes = {}
    for ds in datasets:
        ls = benchmark_label_set(ds)
        ls.to_yaml(out / f"{ds}.yaml")
        hashes[ds] = ls.hash()
    return hashes


def load_benchmark_label_set(dataset: str) -> LabelSet:
    p = CONFIG_DIR / "benchmark_labels" / f"{dataset}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: run `python -m s1pii.v2.labels freeze`")
    return LabelSet.from_yaml(p)


# ------------------------------------------------------------------ held-out labels (C3)

def c3_label_set(descriptions: bool = True) -> LabelSet:
    """C3 query set: every Nemotron raw label (PII and quasi-identifier), name + description."""
    labs = sorted({r.lower() for r in tx.NEMOTRON})
    return LabelSet("nemotron_all", [native_label(r) for r in labs], descriptions)


def heldout_path() -> Path:
    return CONFIG_DIR / "heldout_labels.yaml"


def load_heldout() -> dict:
    import yaml
    p = heldout_path()
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: run `python -m s1pii.v2.labels heldout` before training")
    return yaml.safe_load(p.read_text())


def heldout_nodes(h: dict | None = None) -> set[str]:
    return set((h or load_heldout())["nodes"])


def is_heldout(raw: str, nodes: set[str]) -> bool:
    """A raw label is held out if its node, or any ancestor up to (excluding) a non-leaf
    umbrella, is held out. Umbrella labels (node with children) are never held out."""
    return node_of(raw) in nodes


def select_heldout(span_counts: dict[str, int], embed, *, k: int = 10, min_spans: int = 50,
                   vetoes: dict[str, str] | None = None, encoder_id: str = "") -> dict:
    """Frozen rule: rank Nemotron leaf labels by the maximum cosine similarity of any of their
    texts (name, name+description, paraphrases) to any text of every *other* node's training
    labels; keep the ``k`` least similar with >= ``min_spans`` spans in the Nemotron
    *calibration* split (test plays no part in choosing the classes). Manual review may
    only veto (reason logged); a veto takes the next label in rank order. ``embed`` maps a
    list of strings to an (n, d) array."""
    import numpy as np
    vetoes = vetoes or {}
    leaves = {n for n in TREE if n not in set(TREE.values())}
    nem = sorted({r.lower() for r in tx.NEMOTRON})
    cands = [r for r in nem if node_of(r) in leaves]
    def texts(r):
        node, desc, par = NATIVE[r]
        return [r.replace("_", " "), f"{r.replace('_', ' ')}: {desc}", *par]
    pool = sorted(NATIVE)
    vec = {}
    allt = sorted({t for r in pool for t in texts(r)})
    E = np.asarray(embed(allt), dtype=np.float64)
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
    vec = dict(zip(allt, E))
    rows = []
    for r in cands:
        mine = np.stack([vec[t] for t in texts(r)])
        others = [o for o in pool if node_of(o) != node_of(r)]
        oth = np.stack([vec[t] for o in others for t in texts(o)])
        rows.append((float((mine @ oth.T).max()), r))
    rows.sort()
    chosen, log = [], []
    for sim, r in rows:
        n = span_counts.get(r, 0)
        entry = {"label": r, "node": node_of(r), "max_sim": round(sim, 4), "calib_spans": n}
        if n < min_spans:
            entry["skipped"] = f"< {min_spans} calibration spans"
        elif r in vetoes:
            entry["skipped"] = f"veto: {vetoes[r]}"
        elif len(chosen) < k:
            chosen.append(r); entry["chosen"] = True
        log.append(entry)
    nodes = sorted({node_of(r) for r in chosen})
    synonyms = sorted(x for x in NATIVE if node_of(x) in nodes)
    return {"rule": f"k={k} least max-cosine-similar Nemotron leaf labels with >= {min_spans} calibration spans",
            "encoder": encoder_id, "labels": chosen, "nodes": nodes, "synonyms_all_sources": synonyms,
            "ranking": log}


def apply_vetoes(h: dict, vetoes: dict[str, str], k: int = 10) -> dict:
    """Re-apply the frozen rule to a stored selection with manual vetoes: walk the stored
    ranking in order, skip vetoed labels (reason logged), take the first ``k`` eligible."""
    ranking = [dict(r) for r in h["ranking"]]
    chosen = []
    for r in ranking:
        r.pop("chosen", None)
        if r["label"] in vetoes and "skipped" not in r:
            r["skipped"] = f"veto: {vetoes[r['label']]}"
        if "skipped" not in r and len(chosen) < k:
            chosen.append(r["label"]); r["chosen"] = True
    nodes = sorted({node_of(r) for r in chosen})
    return {**h, "labels": chosen, "nodes": nodes, "synonyms_all_sources": sorted(x for x in NATIVE if node_of(x) in nodes),
            "vetoes": {**h.get("vetoes", {}), **vetoes}, "ranking": ranking}


def main(argv=None) -> None:
    import argparse
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("freeze")
    v = sub.add_parser("veto")
    v.add_argument("path", type=Path)
    v.add_argument("--veto", action="append", required=True, help="label=reason")
    h = sub.add_parser("heldout")
    h.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    h.add_argument("--veto", action="append", default=[], help="label=reason")
    a = ap.parse_args(argv)
    if a.cmd == "freeze":
        print(json.dumps(write_benchmark_label_sets(), indent=2))
        return
    if a.cmd == "veto":
        import yaml
        h = yaml.safe_load(a.path.read_text())
        out = apply_vetoes(h, dict(x.split("=", 1) for x in a.veto))
        a.path.write_text(yaml.safe_dump(out, sort_keys=False))
        print(json.dumps({"labels": out["labels"], "vetoes": out["vetoes"]}, indent=2))
        return
    import yaml
    import torch
    from huggingface_hub import HfApi
    from transformers import AutoModel, AutoTokenizer
    from .. import bench
    from ..schema import read_jsonl
    rev = HfApi().model_info(a.encoder).sha
    tok = AutoTokenizer.from_pretrained(a.encoder, revision=rev)
    enc = AutoModel.from_pretrained(a.encoder, revision=rev).eval()

    @torch.no_grad()
    def embed(xs):
        """sentence-transformers' recipe for this model: mean pooling over tokens, then L2."""
        out = []
        for i in range(0, len(xs), 64):
            e = tok(xs[i:i + 64], padding=True, truncation=True, max_length=128, return_tensors="pt")
            h = enc(**e).last_hidden_state
            m = e["attention_mask"].unsqueeze(-1).float()
            out.append(torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1).numpy())
        return __import__("numpy").concatenate(out)
    counts: dict[str, int] = {}
    for d in read_jsonl(bench.split_paths("nemotron")["calib"]):
        for s in d.spans:
            counts[s.label_raw.lower()] = counts.get(s.label_raw.lower(), 0) + 1
    res = select_heldout(counts, embed, encoder_id=f"{a.encoder}@{rev}",
                         vetoes=dict(v.split("=", 1) for v in a.veto))
    p = heldout_path()
    if p.exists():
        raise FileExistsError(f"{p} exists and is frozen; delete it explicitly to reselect")
    p.write_text(yaml.safe_dump(res, sort_keys=False))
    print(json.dumps({k: res[k] for k in ("labels", "nodes", "encoder")}, indent=2))


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])


def all_texts() -> list[str]:
    """Every label text any v2 stage can query: native labels as name, name + description
    and paraphrases, plus the frozen benchmark sets (names and name + description)."""
    t = set()
    for n in NATIVE:
        lab = native_label(n)
        t |= {lab.text(False), lab.text(True), *paraphrases(n)}
    for ds in ("pii_trace", "tab_direct", "spy_legal", "spy_medical", "nemotron"):
        ls = benchmark_label_set(ds)
        t |= set(ls.texts()) | set(LabelSet(ds, ls.labels, False).texts())
    c = canonical_label_set()
    t |= set(c.texts()) | set(LabelSet("c", c.labels, False).texts())
    return sorted(t)
