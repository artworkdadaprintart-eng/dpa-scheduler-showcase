"""Read the job ticket off an 'Artwork for approval' proof PDF.

The production artwork is entirely outlines, but the proof wrapper around it is
live text — so this is the one place in the corpus where hard job data can be
had for free: customer, die size, repeat length, colour count, substrate, die
number and date. It costs a text extraction, no rendering and no vision.

Why geometry rather than string order
-------------------------------------
Extracting the ticket as a flat string interleaves labels and values, because
the form is a multi-column grid and reading order runs across it::

    Artwork No: Job Name:
    88116 - JARODOL 1ML 30X19 Customer Name:
    Smayan Healthcare Size:

Parsing that by position-in-string is guesswork. Instead we take words with
their bounding boxes and, for each label, look for its value to the right on
the same line, then directly beneath it — whichever is closer. That reads both
common ticket layouts without knowing in advance which one a given customer's
template uses.

Field names vary between templates, so an unmatched label is simply absent from
the result rather than an error. Run ``labelkb proof <pdf>`` against a real
proof to see exactly what was matched.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict, field
from pathlib import Path

import pymupdf

# Canonical field -> the label words that introduce it, longest first so
# "No of Colors" wins over the "No" in "Artwork No".
_ANCHORS: list[tuple[str, tuple[str, ...]]] = [
    ("no_of_colors", ("no", "of", "colors")),
    ("no_of_colours", ("no", "of", "colours")),
    ("new_flexo_plates", ("new", "flexo", "plates")),
    ("no_of_labels_per_roll", ("no", "of", "labels", "per", "roll")),
    ("customer_name", ("customer", "name")),
    ("artwork_no", ("artwork", "no")),
    ("file_details", ("file", "details")),
    ("winding_direction", ("winding", "direction")),
    ("job_name", ("job", "name")),
    ("new_die", ("new", "die")),
    ("roll_dia", ("roll", "dia")),
    ("substrate", ("substrate",)),
    ("remarks", ("remarks",)),
    ("repeat", ("repeat",)),
    ("dated", ("dated",)),
    ("size", ("size",)),
]

_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[xX]\s*(\d+(?:\.\d+)?)")


def _norm(word: str) -> str:
    return word.strip().rstrip(":").lower()


@dataclass
class Word:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def height(self) -> float:
        return self.y1 - self.y0


@dataclass
class Anchor:
    field_name: str
    x0: float
    y0: float
    x1: float
    y1: float
    indices: set[int]


@dataclass
class ProofTicket:
    fields: dict[str, str] = field(default_factory=dict)
    die_w_mm: float | None = None
    die_h_mm: float | None = None
    repeat_mm: float | None = None
    colours: list[str] = field(default_factory=list)
    source: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _extract_words(page: pymupdf.Page) -> list[Word]:
    return [Word(w[0], w[1], w[2], w[3], w[4]) for w in page.get_text("words") if w[4].strip()]


def _find_anchors(words: list[Word]) -> list[Anchor]:
    """Locate label phrases. A label always ends in a colon on these tickets,
    which keeps the word 'size' inside a value from being mistaken for one."""
    anchors: list[Anchor] = []
    claimed: set[int] = set()

    for field_name, phrase in _ANCHORS:
        n = len(phrase)
        for i in range(len(words) - n + 1):
            span = range(i, i + n)
            if claimed.intersection(span):
                continue
            candidate = words[i : i + n]
            if tuple(_norm(w.text) for w in candidate) != phrase:
                continue
            if not candidate[-1].text.rstrip().endswith(":"):
                continue
            # Words of one label sit on one line.
            if max(w.y0 for w in candidate) - min(w.y0 for w in candidate) > 3:
                continue
            anchors.append(
                Anchor(
                    field_name=field_name,
                    x0=min(w.x0 for w in candidate),
                    y0=min(w.y0 for w in candidate),
                    x1=max(w.x1 for w in candidate),
                    y1=max(w.y1 for w in candidate),
                    indices=set(span),
                )
            )
            claimed.update(span)
    return anchors


def _value_for(anchor: Anchor, words: list[Word], label_indices: set[int]) -> str:
    """Collect the value belonging to one label: to its right, else below it."""
    line_h = max(anchor.y1 - anchor.y0, 1.0)

    def free(i: int) -> bool:
        return i not in label_indices

    # To the right, on the same line.
    right = [
        (i, w)
        for i, w in enumerate(words)
        if free(i) and w.x0 >= anchor.x1 - 1 and abs(w.cy - (anchor.y0 + anchor.y1) / 2) < line_h * 0.6
    ]
    right.sort(key=lambda pair: pair[1].x0)

    # Directly beneath, within roughly two lines.
    below = [
        (i, w)
        for i, w in enumerate(words)
        if free(i)
        and w.y0 >= anchor.y1 - 1
        and w.y0 - anchor.y1 < line_h * 2.2
        and w.x1 > anchor.x0 - line_h
        and w.x0 < anchor.x1 + line_h * 6
    ]
    below.sort(key=lambda pair: (pair[1].y0, pair[1].x0))

    chosen = right or below
    if not chosen:
        return ""

    # Keep only words contiguous with the first: a gap means the next cell.
    picked = [chosen[0][1]]
    for _, word in chosen[1:]:
        previous = picked[-1]
        same_line = abs(word.cy - previous.cy) < line_h * 0.6
        if same_line and word.x0 - previous.x1 > line_h * 4:
            break
        if not same_line:
            break
        picked.append(word)

    return " ".join(w.text for w in picked).strip().strip(":").strip()


def parse_proof(pdf_path: str | Path) -> ProofTicket:
    """Extract the job ticket from a proof PDF."""
    path = Path(pdf_path)
    ticket = ProofTicket(source=str(path))

    with pymupdf.open(path) as doc:
        if doc.page_count == 0:
            return ticket
        words = _extract_words(doc[0])

    if not words:
        return ticket

    anchors = _find_anchors(words)
    label_indices = {i for a in anchors for i in a.indices}

    for anchor in anchors:
        value = _value_for(anchor, words, label_indices)
        if value:
            ticket.fields[anchor.field_name] = value

    _derive(ticket)
    return ticket


def _derive(ticket: ProofTicket) -> None:
    """Turn raw ticket strings into typed values."""
    if size := ticket.fields.get("size"):
        if match := _SIZE_RE.search(size):
            ticket.die_w_mm = float(match.group(1))
            ticket.die_h_mm = float(match.group(2))

    if repeat := ticket.fields.get("repeat"):
        if match := re.search(r"(\d+(?:\.\d+)?)", repeat):
            ticket.repeat_mm = float(match.group(1))

    colours = ticket.fields.get("no_of_colors") or ticket.fields.get("no_of_colours")
    if colours:
        # "CMYK+V" -> the four process inks plus a varnish station.
        tokens = [t for t in re.split(r"[+,/\s]+", colours.strip()) if t]
        expanded: list[str] = []
        for token in tokens:
            if token.upper() == "CMYK":
                expanded += ["cyan", "magenta", "yellow", "black"]
            else:
                expanded.append(token.lower())
        ticket.colours = expanded
