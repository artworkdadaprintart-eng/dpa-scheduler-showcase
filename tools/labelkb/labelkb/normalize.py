"""Normalise drug names so the knowledge base has stable keys.

The cardinality problem this solves: brands are many-to-one against generics,
and composition strings are messy free text ("Each ml contains: Diclofenac
Sodium IP 25mg", "PANTOPRAZOLE (AS SODIUM SESQUIHYDRATE) 40 MG"). Lookups must
land on one key per molecule regardless of salt form, pharmacopoeia suffix,
casing, or ingredient order in a combination — otherwise the same drug
scatters into several profiles and the corpus looks thinner than it is.

Canonical key rules:
- lowercase base name, salt form stripped but *remembered*
- pharmacopoeia markers (IP/BP/USP) removed
- combination products: base names sorted alphabetically, joined with "+"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict, field

# Salt/ester/hydrate forms that attach to a base molecule. Stripped for the
# key, kept in the record: "as sodium" can matter to the label text itself.
_SALT_WORDS = (
    "sodium|potassium|calcium|magnesium|hydrochloride|hcl|dihydrochloride|"
    "sulphate|sulfate|maleate|tartrate|bitartrate|citrate|phosphate|"
    "diphosphate|acetate|besylate|besilate|mesylate|mesilate|tosylate|"
    "succinate|fumarate|oxalate|nitrate|stearate|palmitate|propionate|"
    "valerate|decanoate|enanthate|undecanoate|bromide|chloride|iodide|"
    "sesquihydrate|monohydrate|dihydrate|trihydrate|anhydrous|micronised|micronized"
)
_SALT_RE = re.compile(rf"\b(?:{_SALT_WORDS})\b", re.IGNORECASE)
_PHARMACOPOEIA_RE = re.compile(r"\b(?:ip|bp|usp|ph\.?\s*eur)\b\.?", re.IGNORECASE)
_PARENTHETICAL_RE = re.compile(r"\((?:as\s+)?([^)]*)\)", re.IGNORECASE)

_STRENGTH_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(mcg|µg|ug|mg|gm?|g|iu|i\.u\.|ml|%\s*w/[wv]|%)",
    re.IGNORECASE,
)

_FORM_WORDS = re.compile(
    r"\b(injection|injectable|tablets?|capsules?|syrup|suspension|oral\s+drops?|"
    r"drops?|cream|gel|ointment|lotion|solution|infusion|powder|sachet|spray)\b",
    re.IGNORECASE,
)

_LEADIN_RE = re.compile(
    r"^(?:each|every)\s*(?:ml|5\s*ml|tablet|capsule|gm?|sachet|vial|ampoule)?\s*"
    r"contains?\s*:?\s*",
    re.IGNORECASE,
)

# RapidOCR frequently drops inter-word spaces ("Eachmlcontains:DiclofenacSodiumIP25mg").
# Case survives, so word boundaries are recoverable: a lowercase->uppercase
# transition is a lost space, as is a letter->digit transition. digit->letter
# stays joined so "25mg" and "40 mg" both parse the same way downstream.
_LOWER_UPPER_RE = re.compile(r"(?<=[a-z])(?=[A-Z])")
_LETTER_DIGIT_RE = re.compile(r"(?<=[A-Za-z])(?=\d)")


def respace(text: str) -> str:
    """Restore the spaces OCR collapsed, using case and digit boundaries."""
    text = _LOWER_UPPER_RE.sub(" ", text)
    return _LETTER_DIGIT_RE.sub(" ", text)

_SPLIT_RE = re.compile(r"\s*(?:\+|&|,|\band\b)\s*", re.IGNORECASE)


@dataclass
class Ingredient:
    base: str
    salt: str | None = None
    strength_value: float | None = None
    strength_unit: str | None = None
    raw: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _clean_base(text: str) -> str:
    text = _PHARMACOPOEIA_RE.sub(" ", text)
    text = _SALT_RE.sub(" ", text)
    text = re.sub(r"[^a-z\- ]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip(" -")


def parse_ingredient(text: str) -> Ingredient | None:
    raw = text.strip()
    if not raw:
        return None

    salt = None
    # "(as sodium sesquihydrate)" carries the salt; pull it before stripping.
    for match in _PARENTHETICAL_RE.finditer(raw):
        inner_salt = _SALT_RE.search(match.group(1))
        if inner_salt:
            salt = inner_salt.group(0).lower()
    working = _PARENTHETICAL_RE.sub(" ", raw)

    strength_value = strength_unit = None
    strength = _STRENGTH_RE.search(working)
    if strength:
        strength_value = float(strength.group(1))
        strength_unit = strength.group(2).lower().replace("µg", "mcg").replace("ug", "mcg")
        working = working[: strength.start()] + " " + working[strength.end() :]

    if salt is None:
        inline = _SALT_RE.search(working)
        if inline:
            salt = inline.group(0).lower()

    working = _FORM_WORDS.sub(" ", working)
    base = _clean_base(working)
    if not base:
        return None
    return Ingredient(
        base=base,
        salt=salt,
        strength_value=strength_value,
        strength_unit=strength_unit,
        raw=raw,
    )


def parse_composition(text: str) -> list[Ingredient]:
    """One free-text composition line -> ingredients."""
    body = _LEADIN_RE.sub("", respace(text.strip()))
    ingredients = []
    for chunk in _SPLIT_RE.split(body):
        parsed = parse_ingredient(chunk)
        if parsed:
            ingredients.append(parsed)
    return ingredients


def generic_key(ingredients: list[Ingredient]) -> str:
    """Stable KB key: sorted base names joined with '+'."""
    bases = sorted({i.base for i in ingredients if i.base})
    return "+".join(bases)


@dataclass
class GenericQuery:
    key: str
    bases: list[str] = field(default_factory=list)
    strength_value: float | None = None
    strength_unit: str | None = None
    dosage_form: str | None = None
    raw: str = ""


def parse_query(text: str) -> GenericQuery:
    """Parse a request like 'pantoprazole 40mg injection' or
    'amoxicillin + clavulanic acid 625 tablet'."""
    raw = text.strip()
    working = raw

    form_match = _FORM_WORDS.search(working)
    dosage_form = None
    if form_match:
        dosage_form = form_match.group(1).lower().rstrip("s")
        if dosage_form == "capsule":
            dosage_form = "capsule"
        working = _FORM_WORDS.sub(" ", working)

    strength_value = strength_unit = None
    strength = _STRENGTH_RE.search(working)
    if strength:
        strength_value = float(strength.group(1))
        strength_unit = strength.group(2).lower()
        working = working[: strength.start()] + " " + working[strength.end() :]

    bases = sorted(
        {base for chunk in _SPLIT_RE.split(working) if (base := _clean_base(chunk))}
    )
    return GenericQuery(
        key="+".join(bases),
        bases=bases,
        strength_value=strength_value,
        strength_unit=strength_unit,
        dosage_form=dosage_form,
        raw=raw,
    )
