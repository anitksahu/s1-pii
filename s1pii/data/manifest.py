"""Licence gate. Fails closed: unknown datasets and unlisted purposes are refused."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

MANIFEST_PATH = Path(__file__).with_name("manifest.yaml")


class LicenceGateError(PermissionError):
    pass


@lru_cache(maxsize=1)
def manifest() -> dict:
    return yaml.safe_load(MANIFEST_PATH.read_text())


def entry(name: str) -> dict:
    m = manifest()
    if name not in m:
        raise LicenceGateError(f"{name}: not in data/manifest.yaml")
    return m[name]


def require_allowed(name: str, purpose: str) -> dict:
    e = entry(name)
    if purpose == "research":
        purpose_ok = True
    else:
        purpose_ok = purpose in e.get("purposes", [])
    if not purpose_ok:
        raise LicenceGateError(f"{name}: purpose {purpose!r} not allowed (allowed {e.get('purposes')})")
    if e.get("research_only") and os.environ.get("S1PII_ALLOW_RESEARCH") != "1":
        raise LicenceGateError(f"{name}: research-only; set S1PII_ALLOW_RESEARCH=1 to use it")
    return e


def headline_datasets() -> list[str]:
    return [k for k, v in manifest().items() if v.get("role") == "headline"]
