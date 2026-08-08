"""Read Esko's ``.metadata/.<name>.pdf.info`` sidecars.

Esko Automation Engine drops a small binary sidecar next to every processed
PDF. It is undocumented, but it carries exactly the technical data the corpus
otherwise hides: which dieline (.ARD) the artwork was built on, the separation
list, the ink types, the originating application, and the artwork's path on the
studio machine. All of it costs ~5 KB to read, versus rendering and looking.

Format, as verified against a real sidecar (job 88116)
-------------------------------------------------------
The container is a tagged stream. Strings are framed as::

    0F  <uint32 big-endian byte_length>  <UTF-16BE bytes>

Everything is big-endian — length AND payload. The trap, which produced two
wrong parsers in a row before a real file settled it: for ASCII text under 256
bytes, this framing is byte-for-byte identical to "tag ``0F 00 00 00`` +
uint16-LE length + UTF-16LE payload" (``0F 00 00 00 18 00 46 00 54…`` parses
cleanly both ways). The interpretations only diverge on strings of 256+ bytes
and on payload alignment — which is why the misreading passed every synthetic
test and then decoded nothing from a genuine file. If a future Esko version
really does emit the little-endian variant, the parser falls back to it when
the big-endian pass recovers nothing.

Strings arrive in key/value order: a key (``Ink.Names``, ``XMP.Layer.Name``,
``Creator``) followed by its value or values. Multi-valued keys are preceded by
an array header that encodes a half-open range::

    46 03 00 00 00 <lo> 03 00 00 00 <hi> 09 01 09 01     -> hi - lo elements

That count is what delimits a run. Guessing the boundary instead — "stop at the
next string that looks like an identifier" — fails on real data, because plenty
of *values* look like identifiers (``cyan``, ``process``, ``PDF``) and plenty of
*keys* are undotted (``PDF_TFactory``, ``guru``). Using the declared count keeps
``Ink.Names`` at six entries instead of swallowing whatever follows.

We deliberately do not parse the full container: the tag grammar varies between
Esko versions. We recover the string table and the array counts, both of which
have been stable across the corpus. Anything unrecognised stays in ``strings``.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

# The single byte that introduces a string record's big-endian length.
_STRING_TAG = b"\x0f"

# No legitimate field in these sidecars is anywhere near this long; the cap
# stops a coincidental tag match in binary noise from allocating wildly.
_MAX_STRING_BYTES = 4096

# Keys are dotted identifiers with no whitespace (``XMP.Doc.Creator``). Values
# that happen to contain dots ("Adobe Illustrator 30.1 (Windows)",
# "Label 30 x 19.ARD") always contain spaces too, so they never collide.
_KEY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)+$")

# Array header: 46 03 00 00 00 <lo> 03 00 00 00 <hi> 09 01 09 01, giving the
# half-open range [lo, hi) — so hi - lo elements follow.
_ARRAY_HEADER_RE = re.compile(
    rb"\x46\x03\x00\x00\x00(.)\x03\x00\x00\x00(.)\x09\x01\x09\x01", re.S
)

# Bare single-word keys that carry a value but have no dot.
_BARE_KEYS = frozenset(
    {"Creator", "Acquire", "guru", "file", "version", "full", "PDF_TFactory"}
)

_ARTWORK_SUFFIXES = (".pdf", ".ai", ".indd", ".psd", ".cdr")


def _is_key(text: str) -> bool:
    return bool(_KEY_RE.match(text)) or text in _BARE_KEYS


@dataclass(frozen=True)
class StringRecord:
    """A decoded string plus where it sat, so array headers can be located."""

    text: str
    start: int  # offset of the 0x0F marker
    end: int  # offset just past the payload


def _scan_records(data: bytes, encoding: str) -> list[StringRecord]:
    out: list[StringRecord] = []
    pos = 0
    end = len(data)
    while True:
        idx = data.find(_STRING_TAG, pos)
        if idx == -1:
            break
        start = idx + 1
        if start + 4 > end:
            break
        (nbytes,) = struct.unpack_from(">I", data, start)
        start += 4
        # UTF-16 payloads are always an even number of bytes. A single 0x0f is
        # a weak anchor, so every candidate must survive these checks before we
        # accept it as a string rather than coincidental binary noise.
        if nbytes % 2 or nbytes > _MAX_STRING_BYTES or start + nbytes > end:
            pos = idx + 1
            continue
        try:
            text = data[start : start + nbytes].decode(encoding)
        except UnicodeDecodeError:
            pos = idx + 1
            continue
        if any(ch < " " and ch != "\t" for ch in text):
            pos = idx + 1
            continue
        out.append(StringRecord(text, idx, start + nbytes))
        pos = start + nbytes
    return out


def iter_string_records(data: bytes) -> list[StringRecord]:
    """Recover the ordered string table, with byte offsets.

    Big-endian UTF-16 first — verified against a real sidecar (job 88116,
    which the earlier little-endian reading decoded as zero strings). If the
    big-endian pass recovers nothing from a non-trivial file, the little-endian
    variant is tried, in case another Esko version writes it: the two framings
    are byte-identical on short ASCII strings, so the doubt is real and the
    fallback costs nothing.

    Empty strings are preserved: they are meaningful positional values (an ink
    with no assigned type, for instance), so dropping them would silently
    misalign parallel arrays like ``Ink.Names`` and ``Ink.Types``.
    """
    records = _scan_records(data, "utf-16-be")
    if not records and len(data) > 64:
        records = _scan_records(data, "utf-16-le")
    return records


def iter_strings(data: bytes) -> list[str]:
    """The ordered string table, without offsets."""
    return [r.text for r in iter_string_records(data)]


def encode_string(text: str) -> bytes:
    """Frame a string the way Esko does. Used by the round-trip test."""
    payload = text.encode("utf-16-be")
    return _STRING_TAG + struct.pack(">I", len(payload)) + payload


def _declared_count(data: bytes, gap_start: int, gap_end: int) -> int | None:
    """Element count from an array header sitting between a key and its values."""
    match = _ARRAY_HEADER_RE.search(data, gap_start, gap_end)
    if not match:
        return None
    lo, hi = match.group(1)[0], match.group(2)[0]
    return hi - lo if hi >= lo else None


def group_by_key(data: bytes) -> dict[str, list[str]]:
    """Map each key to its values, using the container's declared array counts.

    Keys seen more than once (Esko writes both ``Creator`` and
    ``XMP.Doc.Creator``) each keep their own entry; callers pick.
    """
    records = iter_string_records(data)
    grouped: dict[str, list[str]] = {}
    i, n = 0, len(records)

    while i < n:
        record = records[i]
        if not _is_key(record.text):
            i += 1
            continue

        gap_end = records[i + 1].start if i + 1 < n else len(data)
        count = _declared_count(data, record.end, gap_end)
        if count is None:
            count = 1  # scalar

        values: list[str] = []
        for candidate in records[i + 1 : i + 1 + count]:
            # A dotted key can never be a value, so if the declared count
            # overruns we stop rather than absorb the next field.
            if _KEY_RE.match(candidate.text):
                break
            values.append(candidate.text)

        grouped.setdefault(record.text, []).extend(values)
        i += 1 + len(values)

    return grouped


def encode_array_header(count: int, lo: int = 0) -> bytes:
    """Frame an array header the way Esko does. Used by the round-trip test."""
    return (
        b"\x46\x03\x00\x00\x00"
        + bytes([lo])
        + b"\x03\x00\x00\x00"
        + bytes([lo + count])
        + b"\x09\x01\x09\x01"
    )


@dataclass
class EskoInfo:
    """What a sidecar tells us about one artwork."""

    creator: str | None = None
    source_path: str | None = None
    dieline_ard: str | None = None
    dieline_path: str | None = None
    ink_names: list[str] = field(default_factory=list)
    ink_original_names: list[str] = field(default_factory=list)
    ink_types: list[str] = field(default_factory=list)
    layers: list[str] = field(default_factory=list)
    strings: list[str] = field(default_factory=list)

    @property
    def colour_count(self) -> int:
        """Separations excluding technical inks (dieline, bleed marks)."""
        technical = {"cut", "outside bleed", "crease", "kiss cut", "varnish free"}
        return sum(1 for n in self.ink_names if n.strip().lower() not in technical)

    def to_dict(self) -> dict:
        return {
            "creator": self.creator,
            "source_path": self.source_path,
            "dieline_ard": self.dieline_ard,
            "dieline_path": self.dieline_path,
            "ink_names": self.ink_names,
            "ink_original_names": self.ink_original_names,
            "ink_types": self.ink_types,
            "layers": self.layers,
            "colour_count": self.colour_count,
        }


def _first_nonempty(grouped: dict[str, list[str]], *keys: str) -> str | None:
    for key in keys:
        for value in grouped.get(key, []):
            if value.strip():
                return value
    return None


def parse_info(data: bytes) -> EskoInfo:
    """Parse sidecar bytes into an :class:`EskoInfo`."""
    strings = iter_strings(data)
    grouped = group_by_key(data)

    info = EskoInfo(strings=strings)
    info.creator = _first_nonempty(grouped, "Creator", "XMP.Doc.Creator")
    info.ink_names = list(grouped.get("Ink.Names", []))
    info.ink_original_names = list(grouped.get("Ink.OriginalNames", []))
    info.ink_types = list(grouped.get("Ink.Types", []))
    info.layers = [v for v in grouped.get("XMP.Layer.Name", []) if v.strip()]

    # The dieline is referenced both as a full path (under the ``guru`` resource
    # key) and as a layer name. Prefer the path; fall back to any .ARD mention.
    for text in strings:
        if text.lower().endswith(".ard"):
            if "/" in text or "\\" in text:
                info.dieline_path = text
                info.dieline_ard = Path(text.replace("\\", "/")).name
                break
            info.dieline_ard = info.dieline_ard or text

    # The artwork's own path on the studio machine — also the cheapest evidence
    # of where the corpus is mounted (e.g. "G:/My Drive/Artworks/...").
    for text in strings:
        lowered = text.lower()
        if lowered.endswith(_ARTWORK_SUFFIXES) and ("/" in text or "\\" in text):
            info.source_path = text
            break

    return info


def parse_info_file(path: str | Path) -> EskoInfo:
    return parse_info(Path(path).read_bytes())


def find_sidecar(artwork_pdf: Path) -> Path | None:
    """Locate the sidecar for an artwork PDF.

    Esko writes ``<dir>/.metadata/.<filename>.info`` — a leading dot on the
    filename, inside a hidden directory.
    """
    candidate = artwork_pdf.parent / ".metadata" / f".{artwork_pdf.name}.info"
    return candidate if candidate.exists() else None
