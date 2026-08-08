"""Discover job folders in the artwork corpus and resolve their files.

The corpus is organised as::

    Artworks/<Customer>/<Year>/<MONTH YEAR>/<jobno> - <BRAND> <pack> <WxH>/

but only loosely. Five years of manual filing means stray depths, `New folder
(6)`, loose `.cdr` files at the root, and shortcuts. So rather than assuming a
fixed four-level shape, we walk and recognise **job folders by their name** —
a leading job number and a trailing die size. Everything else is ignored.

Ganged imposition files (`Mix Job No 87306, 87855 … UNIT - 1-export.pdf`) are
press layouts holding several labels at once, not single artworks. They are
excluded: feeding one to the extractor would produce a record describing a
sheet rather than a label.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Iterator

from .esko_info import find_sidecar

# "88116 - JARODOL 1ML 30X19" — job number, separator, then the description.
_JOB_PREFIX_RE = re.compile(r"^\s*(\d{4,7})\s*[-–]\s*(.+)$")

# Die size in millimetres, e.g. "30X19", "92 x 45", "89X40.5". The last such
# match in the name wins: brand names occasionally contain digits.
_DIE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[Xx]\s*(\d+(?:\.\d+)?)")

# Pack size / strength, e.g. "100ML", "2ml", "40MG", "1GM".
_PACK_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*(ML|MG|GM|MCG|IU|KG|L|G)$", re.IGNORECASE)

# Tokens that mark the end of the brand name.
_FORM_WORDS = {
    "SYRUP": "syrup", "SYP": "syrup", "SUSP": "suspension", "SUSPENSION": "suspension",
    "INJ": "injection", "INJECTION": "injection", "DROP": "drops", "DROPS": "drops",
    "TAB": "tablet", "TABLET": "tablet", "TABLETS": "tablet", "CAP": "capsule",
    "CAPSULE": "capsule", "CREAM": "cream", "GEL": "gel", "OINT": "ointment",
    "OINTMENT": "ointment", "LOTION": "lotion", "SOLUTION": "solution",
    "POWDER": "powder", "SACHET": "sachet", "SPRAY": "spray", "SHAMPOO": "shampoo",
    "SOAP": "soap", "KIT": "kit", "INFUSION": "infusion",
}

# Words that carry job history, not product identity.
_NOISE_WORDS = {"REPEAT", "NEW", "REV", "REVISED", "UNIT", "COPY"}

# Names that mean "this is not a single label artwork".
_GANG_MARKERS = ("mix job no", "unit -", "-export", "-assets", "ups -")

_JUNK_DIRS = {".metadata", ".tmp", "__macosx"}

_SOURCE_SUFFIXES = {".cdr", ".ai", ".indd", ".psd"}


def is_gang_file(name: str) -> bool:
    """True for ganged press layouts rather than single-label artwork."""
    lowered = name.lower()
    return any(marker in lowered for marker in _GANG_MARKERS)


@dataclass
class ParsedName:
    job_no: str
    brand: str | None = None
    pack_size: str | None = None
    dosage_form: str | None = None
    die_w_mm: float | None = None
    die_h_mm: float | None = None
    is_repeat: bool = False
    raw: str = ""


def parse_job_folder(name: str) -> ParsedName | None:
    """Pull job number, brand, pack size and die size out of a folder name.

    Returns ``None`` if the name is not a job folder at all.
    """
    match = _JOB_PREFIX_RE.match(name)
    if not match:
        return None

    job_no, rest = match.group(1), match.group(2).strip()
    parsed = ParsedName(job_no=job_no, raw=name)

    # Die size is the last WxH in the name; everything before it describes the
    # product.
    die_matches = list(_DIE_RE.finditer(rest))
    if die_matches:
        last = die_matches[-1]
        parsed.die_w_mm = float(last.group(1))
        parsed.die_h_mm = float(last.group(2))
        rest = rest[: last.start()].strip()

    tokens = [t for t in re.split(r"\s+", rest) if t]
    brand_tokens: list[str] = []
    for token in tokens:
        bare = token.strip("().,-").upper()
        if _PACK_RE.match(bare):
            parsed.pack_size = token.strip("().,")
            break
        if bare in _FORM_WORDS:
            parsed.dosage_form = _FORM_WORDS[bare]
            break
        if bare in _NOISE_WORDS:
            break
        brand_tokens.append(token)

    # Trailing tokens still carry pack size and form even after the brand ends.
    for token in tokens[len(brand_tokens) :]:
        bare = token.strip("().,-").upper()
        if parsed.pack_size is None and _PACK_RE.match(bare):
            parsed.pack_size = token.strip("().,")
        if parsed.dosage_form is None and bare in _FORM_WORDS:
            parsed.dosage_form = _FORM_WORDS[bare]
        if bare == "REPEAT":
            parsed.is_repeat = True

    if brand_tokens:
        parsed.brand = " ".join(brand_tokens).strip("-– ")
    return parsed


@dataclass
class JobFolder:
    job_no: str
    customer: str
    path: str
    parsed: dict
    artwork_pdf: str | None = None
    proof_pdf: str | None = None
    source_files: list[str] = field(default_factory=list)
    info_sidecar: str | None = None
    artwork_size: int | None = None
    artwork_mtime: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        """Cheap change detector, so a re-run skips untouched jobs."""
        return f"{self.artwork_size}:{self.artwork_mtime}"


def resolve_files(job_dir: Path, job_no: str) -> dict:
    """Classify the files inside a job folder."""
    artwork: Path | None = None
    proof: Path | None = None
    sources: list[Path] = []

    for entry in sorted(job_dir.iterdir()):
        if entry.is_dir() or entry.name.startswith("."):
            continue
        suffix = entry.suffix.lower()
        if suffix in _SOURCE_SUFFIXES:
            sources.append(entry)
            continue
        if suffix != ".pdf" or is_gang_file(entry.name):
            continue
        if "approval" in entry.stem.lower():
            # Prefer a proof that names this job over a stray one.
            if proof is None or entry.stem.startswith(job_no):
                proof = entry
        else:
            if artwork is None or entry.stem.strip() == job_dir.name.strip():
                artwork = entry

    return {
        "artwork_pdf": str(artwork) if artwork else None,
        "proof_pdf": str(proof) if proof else None,
        "source_files": [str(p) for p in sources],
        "info_sidecar": str(find_sidecar(artwork)) if artwork and find_sidecar(artwork) else None,
    }


def iter_job_folders(root: Path, max_depth: int = 5) -> Iterator[JobFolder]:
    """Yield every job folder under ``root``.

    Walking stops descending once a job folder is recognised — nothing useful
    nests below one, and it keeps `.metadata` out of the results.
    """
    root = Path(root)
    root_depth = len(root.parts)

    for dirpath, dirnames, _ in os.walk(root):
        current = Path(dirpath)
        depth = len(current.parts) - root_depth

        dirnames[:] = [d for d in dirnames if d.lower() not in _JUNK_DIRS]
        if depth >= max_depth:
            dirnames[:] = []

        parsed = parse_job_folder(current.name) if depth > 0 else None
        if parsed is None or parsed.die_w_mm is None:
            # Not a job folder (or too ambiguous to trust) — keep descending.
            continue
        if is_gang_file(current.name):
            dirnames[:] = []
            continue

        dirnames[:] = []  # a job folder is a leaf
        relative = current.relative_to(root)
        customer = relative.parts[0] if relative.parts else ""

        job = JobFolder(
            job_no=parsed.job_no,
            customer=customer,
            path=str(current),
            parsed=asdict(parsed),
        )
        try:
            job.__dict__.update(resolve_files(current, parsed.job_no))
        except OSError:
            continue

        if job.artwork_pdf:
            try:
                stat = Path(job.artwork_pdf).stat()
                job.artwork_size, job.artwork_mtime = stat.st_size, stat.st_mtime
            except OSError:
                pass

        yield job


def build_manifest(root: Path, out_path: Path) -> int:
    """Write one JSON object per job folder. Returns the count."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for job in iter_job_folders(root):
            handle.write(json.dumps(job.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count


def load_manifest(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
