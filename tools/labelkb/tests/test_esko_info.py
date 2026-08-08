"""Tests for the Esko sidecar scraper.

The fixture is a *reconstruction*: the string table and framing are exactly
those observed in the real sidecar for job 88116 (Smayan Healthcare, JARODOL
1ML, 30x19), rebuilt with the documented encoder rather than checked in as an
opaque 5 KB blob. That keeps the test readable and lets it assert the framing
itself, which is the part that was subtly wrong first time round.

To validate against genuine sidecars, run ``labelkb info <path>`` on a real
corpus — see the README.
"""

from __future__ import annotations

from labelkb.corpus.esko_info import (
    EskoInfo,
    encode_array_header,
    encode_string,
    group_by_key,
    iter_strings,
    parse_info,
)

ARTWORK_PATH = (
    "G:/My Drive/Artworks/Smayan Healthcare/2026/AUGUST 2026/"
    "88116 - JARODOL 1ML 30X19/88116 - JARODOL 1ML 30X19.pdf"
)
DIELINE_PATH = "G$/My Drive/Keylines/Label 30 x 19.ARD"

# 152 characters -> 304 bytes. Anything above 255 bytes is what the original
# "0F 00 00 00 + uint16-LE" misreading silently dropped, so it earns its place.
LONG_REF = (
    "iOriginalRef=file%3A%2F%2Fguru%2FG%2524%2FMy%2520Drive%2FKeylines"
    "%2FLabel%252030%2520x%252019.ARD&iMIMEType=application%2Fard"
    "&iExtType=&iPageNr=&iCount="
)

# Key/value structure as it appears in the real file. Multi-valued keys carry
# an array header; scalars do not. `PDF_TFactory` is here on purpose: it is an
# undotted key, and an earlier parser mistook it for another ink.
FIELDS: list[tuple[str, list[str], bool]] = [
    ("Creator", ["Adobe Illustrator 30.1 (Windows)"], False),
    ("XMP.Layer.Name", ["Layer 1", "Label 30 x 19.ARD"], True),
    ("guru", [DIELINE_PATH], False),
    ("XMP.CADFiles", [LONG_REF], False),
    ("Ink.OriginalNames", ["Cyan", "Magenta", "Yellow", "Black", "Cut", "Outside Bleed"], True),
    ("Ink.Types", ["process", "process", "process", "process", "", ""], True),
    ("Ink.Names", ["cyan", "magenta", "yellow", "black", "Cut", "Outside Bleed"], True),
    ("PDF_TFactory", ["PDF"], False),
    ("file", [ARTWORK_PATH], False),
]

SEQUENCE = [s for key, values, _ in FIELDS for s in (key, *values)]


def _record(text: str) -> bytes:
    """One string record, with the header bytes Esko puts in front of it."""
    return b"\x41\x07\x00\x00\x00" + bytes([min(len(text), 255)]) + encode_string(text)


def build_sidecar(fields: list[tuple[str, list[str], bool]]) -> bytes:
    out = bytearray(b"\x45\x07\x00\x00\x00\x03")
    for key, values, is_array in fields:
        out += _record(key)
        if is_array:
            out += encode_array_header(len(values))
        for value in values:
            out += b"\x40\x05\x12" + _record(value)
        out += b"\x40\x05\x12"
    return bytes(out)


FIXTURE = build_sidecar(FIELDS)


def test_framing_round_trips():
    assert iter_strings(FIXTURE) == SEQUENCE


def test_long_strings_survive():
    """Regression: strings of 256+ bytes must not be skipped."""
    assert len(LONG_REF.encode("utf-16-le")) > 255
    assert LONG_REF in iter_strings(FIXTURE)


def test_empty_values_are_positional():
    """Blank ink types keep their slot, so parallel arrays stay aligned."""
    grouped = group_by_key(FIXTURE)
    assert grouped["Ink.Types"] == ["process"] * 4 + ["", ""]
    assert len(grouped["Ink.Types"]) == len(grouped["Ink.Names"])


def test_undotted_key_does_not_join_the_previous_run():
    """Regression: `PDF_TFactory` follows the inks and must not become one."""
    grouped = group_by_key(FIXTURE)
    assert grouped["Ink.Names"] == ["cyan", "magenta", "yellow", "black", "Cut", "Outside Bleed"]
    assert grouped["PDF_TFactory"] == ["PDF"]


def test_parses_the_real_values():
    info = parse_info(FIXTURE)
    assert info.creator == "Adobe Illustrator 30.1 (Windows)"
    assert info.dieline_ard == "Label 30 x 19.ARD"
    assert info.dieline_path == DIELINE_PATH
    assert info.source_path == ARTWORK_PATH
    assert info.ink_names == ["cyan", "magenta", "yellow", "black", "Cut", "Outside Bleed"]
    assert info.ink_original_names == ["Cyan", "Magenta", "Yellow", "Black", "Cut", "Outside Bleed"]
    assert info.layers == ["Layer 1", "Label 30 x 19.ARD"]


def test_colour_count_excludes_technical_inks():
    """Cut and Outside Bleed are dieline furniture, not printed colours."""
    assert parse_info(FIXTURE).colour_count == 4


def test_garbage_input_is_survivable():
    """A truncated or non-Esko file must return empty, not raise."""
    assert parse_info(b"").strings == []
    assert isinstance(parse_info(b"\x0f\xff\xff\xff\xff not esko"), EskoInfo)
