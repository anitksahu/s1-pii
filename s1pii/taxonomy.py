"""Per-dataset raw-label maps onto the canonical taxonomy.

Every raw label maps to a canonical PII type, ``IGNORE`` (excluded from both leak and
over-redaction, used for quasi-identifiers and categories outside the core set that some
systems legitimately mask) or ``NOT_PII``. Mapping is fail-closed: an unseen raw label
raises ``UnmappedLabelError``. Run ``python -m s1pii.data.census`` (notebook 01) to list
unmapped labels, then extend the map here in a reviewed commit.

The headline metric is type-agnostic, so what matters most is the PII / IGNORE / NOT_PII
decision. Types matter only for secondary per-type reporting.
"""
from __future__ import annotations

from .schema import (
    PERSON, ADDRESS, EMAIL, PHONE, URL, DATE, ACCOUNT_NUMBER, SECRET, OTHER_PII,
    IGNORE, NOT_PII,
)

MAP_VERSION = "taxonomy-v0.2"   # v0.2: labels found unmapped by the Colab census (Nemotron test, Gretel)


class UnmappedLabelError(KeyError):
    def __init__(self, dataset: str, labels):
        self.dataset = dataset
        self.labels = sorted(set(labels))
        super().__init__(f"{dataset}: unmapped raw labels {self.labels}; extend s1pii/taxonomy.py")


# TAB: label is (entity_type, identifier_type). Headline tier: DIRECT = PII,
# QUASI = IGNORE, NO_MASK = NOT_PII. The quasi tier counts QUASI as PII too.
TAB_ENTITY = {
    "PERSON": PERSON, "CODE": OTHER_PII, "LOC": ADDRESS, "ORG": OTHER_PII,
    "DATETIME": DATE, "DEM": OTHER_PII, "QUANTITY": OTHER_PII, "MISC": OTHER_PII,
}

SPY = {  # tags as emitted by SPY.py (_ENT_TAGS), plus the lowercase card spelling
    "NAME": PERSON, "EMAIL": EMAIL, "PHONE_NUM": PHONE, "ADDRESS": ADDRESS,
    "URL": URL, "USERNAME": OTHER_PII, "ID_NUM": ACCOUNT_NUMBER,
    "name": PERSON, "email": EMAIL, "phone_number": PHONE, "address": ADDRESS,
    "url": URL, "username": OTHER_PII, "id_num": ACCOUNT_NUMBER,
}

PII_TRACE = {
    "private_person": PERSON, "private_date": DATE, "private_url": URL,
    "private_address": ADDRESS, "account_number": ACCOUNT_NUMBER, "private_email": EMAIL,
    "private_phone": PHONE, "other_pii": OTHER_PII, "secret": SECRET,
}

NEMOTRON = {
    "first_name": PERSON, "last_name": PERSON, "date_of_birth": DATE,
    "street_address": ADDRESS, "email": EMAIL, "phone_number": PHONE, "url": URL,
    "ssn": ACCOUNT_NUMBER, "medical_record_number": ACCOUNT_NUMBER, "customer_id": ACCOUNT_NUMBER,
    "account_number": ACCOUNT_NUMBER, "health_plan_beneficiary_number": ACCOUNT_NUMBER,
    "certificate_license_number": ACCOUNT_NUMBER, "employee_id": ACCOUNT_NUMBER,
    "vehicle_identifier": ACCOUNT_NUMBER, "license_plate": ACCOUNT_NUMBER,
    "device_identifier": ACCOUNT_NUMBER, "bank_routing_number": ACCOUNT_NUMBER,
    "swift_bic": ACCOUNT_NUMBER,
    "cvv": SECRET, "pin": SECRET, "password": SECRET,
    "user_name": OTHER_PII, "ipv4": OTHER_PII, "mac_address": OTHER_PII,
    "biometric_identifier": OTHER_PII,
    "api_key": SECRET, "http_cookie": SECRET,
    "credit_debit_card": ACCOUNT_NUMBER, "national_id": ACCOUNT_NUMBER, "tax_id": ACCOUNT_NUMBER,
    "unique_id": ACCOUNT_NUMBER, "fax_number": PHONE, "ipv6": OTHER_PII,
    # quasi-identifiers and special-category attributes: excluded from headline metrics
    "city": IGNORE, "state": IGNORE, "county": IGNORE, "country": IGNORE, "postcode": IGNORE,
    "coordinate": IGNORE, "date": IGNORE, "time": IGNORE, "date_time": IGNORE,
    "company_name": IGNORE, "occupation": IGNORE, "education_level": IGNORE,
    "employment_status": IGNORE, "blood_type": IGNORE, "race_ethnicity": IGNORE,
    "religious_belief": IGNORE, "sexuality": IGNORE, "political_view": IGNORE,
    "gender": IGNORE, "age": IGNORE, "language": IGNORE,
}

GRETEL = {
    "name": PERSON, "first_name": PERSON, "last_name": PERSON,
    "street_address": ADDRESS, "address": ADDRESS,
    "email": EMAIL, "phone_number": PHONE, "url": URL,
    "account_number": ACCOUNT_NUMBER, "iban": ACCOUNT_NUMBER, "bban": ACCOUNT_NUMBER,
    "swift_bic_code": ACCOUNT_NUMBER, "swift_bic": ACCOUNT_NUMBER, "routing_number": ACCOUNT_NUMBER,
    "bank_routing_number": ACCOUNT_NUMBER, "credit_card_number": ACCOUNT_NUMBER,
    "customer_id": ACCOUNT_NUMBER, "employee_id": ACCOUNT_NUMBER, "ssn": ACCOUNT_NUMBER,
    "tax_id": ACCOUNT_NUMBER, "passport_number": ACCOUNT_NUMBER,
    "driver_license_number": ACCOUNT_NUMBER,
    "credit_card_security_code": SECRET, "password": SECRET, "api_key": SECRET, "pin": SECRET, "account_pin": SECRET,
    "user_name": OTHER_PII, "ipv4": OTHER_PII, "ipv6": OTHER_PII,
    "date_of_birth": DATE,
    "company": IGNORE, "date": IGNORE, "date_time": IGNORE, "time": IGNORE,
    "local_latlng": IGNORE, "city": IGNORE, "country": IGNORE, "postcode": IGNORE,
}

AI4PRIVACY = {
    "GIVENNAME": PERSON, "SURNAME": PERSON, "FIRSTNAME": PERSON, "LASTNAME": PERSON,
    "MIDDLENAME": PERSON, "PREFIX": IGNORE, "TITLE": IGNORE,
    "STREET": ADDRESS, "BUILDINGNUM": ADDRESS, "STREETADDRESS": ADDRESS, "SECONDARYADDRESS": ADDRESS,
    "ZIPCODE": IGNORE, "CITY": IGNORE, "STATE": IGNORE, "COUNTY": IGNORE,
    "EMAIL": EMAIL, "TELEPHONENUM": PHONE, "PHONENUMBER": PHONE, "URL": URL,
    "DATEOFBIRTH": DATE, "DOB": DATE, "DATE": IGNORE, "TIME": IGNORE, "AGE": IGNORE,
    "SEX": IGNORE, "GENDER": IGNORE,
    "SOCIALNUM": ACCOUNT_NUMBER, "SSN": ACCOUNT_NUMBER, "IDCARDNUM": ACCOUNT_NUMBER,
    "PASSPORTNUM": ACCOUNT_NUMBER, "DRIVERLICENSENUM": ACCOUNT_NUMBER, "TAXNUM": ACCOUNT_NUMBER,
    "CREDITCARDNUMBER": ACCOUNT_NUMBER, "ACCOUNTNUM": ACCOUNT_NUMBER, "ACCOUNTNUMBER": ACCOUNT_NUMBER,
    "IBAN": ACCOUNT_NUMBER, "PASSWORD": SECRET, "PIN": SECRET, "CREDITCARDCVV": SECRET,
    "USERNAME": OTHER_PII, "IP": OTHER_PII, "IPV4": OTHER_PII, "IPV6": OTHER_PII,
}

FRESH_REAL = {t: t for t in (PERSON, ADDRESS, EMAIL, PHONE, URL, DATE, ACCOUNT_NUMBER,
                             SECRET, OTHER_PII, IGNORE, NOT_PII)}

MAPS: dict[str, dict[str, str]] = {
    "spy": SPY, "pii_trace": PII_TRACE, "nemotron": NEMOTRON, "gretel": GRETEL,
    "ai4privacy": AI4PRIVACY, "fresh_real": FRESH_REAL,
}


# Prediction types admissible per benchmark. Predictions of other types are dropped for
# every system before scoring, so a correct prediction of a type the benchmark never
# annotates (e.g. a DATE on SPY) is not counted as over-redaction. OTHER_PII and
# ACCOUNT_NUMBER are both admissible wherever either is annotated, because systems
# legitimately confuse them.
_ALL = frozenset((PERSON, ADDRESS, EMAIL, PHONE, URL, DATE, ACCOUNT_NUMBER, SECRET, OTHER_PII))
ADMISSIBLE: dict[str, frozenset[str]] = {
    "tab_direct": _ALL, "tab_quasi": _ALL, "pii_trace": _ALL, "nemotron": _ALL, "gretel": _ALL,
    "ai4privacy": _ALL, "fresh_real": _ALL,
    "spy_medical": frozenset((PERSON, EMAIL, PHONE, ADDRESS, URL, ACCOUNT_NUMBER, OTHER_PII)),
    "spy_legal": frozenset((PERSON, EMAIL, PHONE, ADDRESS, URL, ACCOUNT_NUMBER, OTHER_PII)),
}


def admissible(dataset: str) -> frozenset[str]:
    if dataset not in ADMISSIBLE:
        raise KeyError(f"{dataset}: no admissible prediction types declared in taxonomy.ADMISSIBLE")
    return ADMISSIBLE[dataset]


def map_label(dataset: str, raw: str) -> str:
    table = MAPS[dataset]
    if raw not in table:
        raise UnmappedLabelError(dataset, [raw])
    return table[raw]


def map_tab(entity_type: str, identifier_type: str, tier: str = "direct") -> str:
    """TAB mapping. ``tier='direct'`` is the headline; ``tier='quasi'`` counts QUASI as PII."""
    if identifier_type == "NO_MASK":
        return NOT_PII
    if entity_type not in TAB_ENTITY:
        raise UnmappedLabelError("tab", [entity_type])
    if identifier_type == "DIRECT":
        return TAB_ENTITY[entity_type]
    if identifier_type == "QUASI":
        return TAB_ENTITY[entity_type] if tier == "quasi" else IGNORE
    raise UnmappedLabelError("tab", [identifier_type])
