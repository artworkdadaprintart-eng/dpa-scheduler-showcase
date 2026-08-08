"""Fuse every signal about one job into a single LabelRecord.

Inputs, in order of trustworthiness:

1. Deterministic facts — folder name, artwork page box (true mm), Esko sidecar
   (dieline, inks), proof ticket (substrate, repeat, customer). These are never
   overridden by vision.
2. Measurements — OCR boxes in mm, CV regions (red band, ruled boxes, barcode).
3. Judgements — role assignments (VLM or rules), regime detection.

The record keeps per-field confidence and the engine that produced each
judgement. Extraction never fails a job for being hard to read: it produces a
low-confidence record instead, because "this label needs human review" is
knowledge too — it is what feeds the review queue.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from . import EXTRACTOR_VERSION
from .config import Config
from .corpus.esko_info import parse_info_file
from .corpus.proof import parse_proof
from .render import page_geometry, render_page
from .vision import cv as cvmod
from .vision.ocr import OcrEngine, OcrResult
from .vision.vlm import Role, RoleResult, Transport, assign_roles


class Zone(BaseModel):
    role: str
    text: str
    bbox_norm: list[float]  # x0,y0,x1,y1 in 0..1 of die width/height
    height_mm: float
    confidence: float


class Prominence(BaseModel):
    brand_mm: Optional[float] = None
    generic_mm: Optional[float] = None
    ratio: Optional[float] = None  # generic/brand: Rule 96 cares about this


class Compliance(BaseModel):
    rx_status: str = "unknown"  # rx | otc | unknown
    statutory_warnings: list[str] = Field(default_factory=list)
    red_band: bool = False
    red_band_bbox_mm: Optional[list[float]] = None
    prominence: Prominence = Field(default_factory=Prominence)
    storage: Optional[str] = None
    licence_no: Optional[str] = None
    manufacturer: Optional[str] = None
    marketer: Optional[str] = None
    mrp_text: Optional[str] = None
    batch_block: Optional[str] = None
    has_barcode: bool = False


class PrintSpec(BaseModel):
    die_w_mm: Optional[float] = None
    die_h_mm: Optional[float] = None
    dieline_ard: Optional[str] = None
    substrate: Optional[str] = None
    inks: list[str] = Field(default_factory=list)
    colour_count: Optional[int] = None
    repeat_mm: Optional[float] = None
    creator: Optional[str] = None


class Provenance(BaseModel):
    artwork_pdf: Optional[str] = None
    proof_pdf: Optional[str] = None
    render_png: Optional[str] = None
    role_engine: str = "rules"
    role_disagreements: list[int] = Field(default_factory=list)
    ocr_mean_confidence: Optional[float] = None
    extractor_version: int = EXTRACTOR_VERSION
    notes: list[str] = Field(default_factory=list)


class LabelRecord(BaseModel):
    job_no: str
    customer: str
    brand: Optional[str] = None
    generic_name: Optional[str] = None
    composition_raw: list[str] = Field(default_factory=list)
    dosage_form: Optional[str] = None
    pack_size: Optional[str] = None
    regime: str = "unknown"  # allopathic | ayush | fssai | other | unknown
    regime_evidence: Optional[str] = None
    compliance: Compliance = Field(default_factory=Compliance)
    print_spec: PrintSpec = Field(default_factory=PrintSpec)
    zones: list[Zone] = Field(default_factory=list)
    provenance: Provenance = Field(default_factory=Provenance)


_RX_HINTS = re.compile(r"schedule\s*[hx]|prescription|registered\s*medical", re.I)
_AYUSH_HINTS = re.compile(r"ayurved|ayush|proprietary\s*medicine|herbal", re.I)
_FSSAI_HINTS = re.compile(r"fssai|nutraceutical|food\s*supplement|dietary|not\s*for\s*medicinal", re.I)
_PHARMA_HINTS = re.compile(r"\b(ip|bp|usp)\b|injection|tablets?|capsules?|mfg\.?\s*lic", re.I)


def detect_regime(full_text: str) -> tuple[str, str]:
    """Which rulebook governs this label, from its own text."""
    squashed = re.sub(r"\s+", " ", full_text)
    if _AYUSH_HINTS.search(squashed):
        return "ayush", _AYUSH_HINTS.search(squashed).group(0)
    if _FSSAI_HINTS.search(squashed):
        return "fssai", _FSSAI_HINTS.search(squashed).group(0)
    if _PHARMA_HINTS.search(squashed):
        return "allopathic", _PHARMA_HINTS.search(squashed).group(0)
    return "other", ""


def _texts_for(roles: RoleResult, ocr: OcrResult, role: Role) -> list[str]:
    return [ocr.lines[i].text for i in roles.lines_with(role)]


def _first(roles: RoleResult, ocr: OcrResult, role: Role) -> Optional[str]:
    texts = _texts_for(roles, ocr, role)
    return texts[0] if texts else None


def fuse(
    job: dict,
    ocr: OcrResult,
    roles: RoleResult,
    regions: dict,
    geometry,
    ticket=None,
    esko=None,
) -> LabelRecord:
    """Pure assembly: no I/O, fully unit-testable."""
    parsed = job.get("parsed") or {}
    record = LabelRecord(job_no=job["job_no"], customer=job.get("customer", ""))

    # Identity: folder name first, vision to fill gaps.
    record.brand = parsed.get("brand") or _first(roles, ocr, Role.BRAND)
    record.generic_name = _first(roles, ocr, Role.GENERIC_NAME)
    record.composition_raw = _texts_for(roles, ocr, Role.COMPOSITION)
    record.dosage_form = parsed.get("dosage_form")
    record.pack_size = parsed.get("pack_size")
    if not record.dosage_form:
        form = _first(roles, ocr, Role.DOSAGE_FORM)
        record.dosage_form = form.lower() if form else None

    # Regime.
    record.regime, record.regime_evidence = detect_regime(ocr.full_text)

    # Compliance.
    compliance = record.compliance
    compliance.statutory_warnings = _texts_for(roles, ocr, Role.STATUTORY_WARNING)
    compliance.storage = _first(roles, ocr, Role.STORAGE)
    compliance.licence_no = _first(roles, ocr, Role.LICENCE_NO)
    compliance.manufacturer = _first(roles, ocr, Role.MANUFACTURER)
    compliance.marketer = _first(roles, ocr, Role.MARKETER)
    compliance.mrp_text = _first(roles, ocr, Role.MRP)
    compliance.batch_block = _first(roles, ocr, Role.BATCH_MFG_EXP)

    warnings_text = " ".join(compliance.statutory_warnings)
    if _RX_HINTS.search(warnings_text) or roles.lines_with(Role.RX_SYMBOL):
        compliance.rx_status = "rx"

    scale = ocr.px_per_mm or 1.0
    red_bands = regions.get("red_bands") or []
    if red_bands:
        compliance.red_band = True
        compliance.red_band_bbox_mm = red_bands[0].bbox_mm(scale)
    compliance.has_barcode = bool(regions.get("barcodes"))

    # Prominence: the measured heart of Rule 96's "conspicuous" requirement.
    brand_lines = roles.lines_with(Role.BRAND)
    generic_lines = roles.lines_with(Role.GENERIC_NAME)
    if brand_lines:
        compliance.prominence.brand_mm = round(ocr.lines[brand_lines[0]].height_px / scale, 3)
    if generic_lines:
        compliance.prominence.generic_mm = round(
            ocr.lines[generic_lines[0]].height_px / scale, 3
        )
    if compliance.prominence.brand_mm and compliance.prominence.generic_mm:
        compliance.prominence.ratio = round(
            compliance.prominence.generic_mm / compliance.prominence.brand_mm, 3
        )

    # Print spec: deterministic sources only.
    spec = record.print_spec
    spec.die_w_mm = geometry.width_mm if geometry else parsed.get("die_w_mm")
    spec.die_h_mm = geometry.height_mm if geometry else parsed.get("die_h_mm")
    if esko:
        spec.dieline_ard = esko.dieline_ard
        spec.inks = esko.ink_names
        spec.colour_count = esko.colour_count
        spec.creator = esko.creator
    if ticket:
        spec.substrate = ticket.fields.get("substrate")
        spec.repeat_mm = ticket.repeat_mm
        if not spec.inks and ticket.colours:
            spec.inks = ticket.colours
            spec.colour_count = len([c for c in ticket.colours if c not in ("v",)])

    # Zone map: normalised against the die, heights in mm.
    die_w = spec.die_w_mm or 1.0
    die_h = spec.die_h_mm or 1.0
    assignments = {a.line_index: a for a in roles.assignments}
    for index, line in enumerate(ocr.lines):
        assignment = assignments.get(index)
        record.zones.append(
            Zone(
                role=(assignment.role.value if assignment else Role.OTHER.value),
                text=line.text,
                bbox_norm=[
                    round(line.x0 / scale / die_w, 4),
                    round(line.y0 / scale / die_h, 4),
                    round(line.x1 / scale / die_w, 4),
                    round(line.y1 / scale / die_h, 4),
                ],
                height_mm=round(line.height_px / scale, 3),
                confidence=(assignment.confidence if assignment else 0.0),
            )
        )

    # Provenance.
    record.provenance.artwork_pdf = job.get("artwork_pdf")
    record.provenance.proof_pdf = job.get("proof_pdf")
    record.provenance.role_engine = roles.engine
    record.provenance.role_disagreements = roles.disagreements
    if ocr.lines:
        record.provenance.ocr_mean_confidence = round(
            sum(l.confidence for l in ocr.lines) / len(ocr.lines), 4
        )
    return record


def extract_label(
    job: dict,
    config: Config | None = None,
    ocr_engine: OcrEngine | None = None,
    transport: Transport | None = None,
) -> LabelRecord:
    """Run the full pipeline for one job folder (I/O wrapper around fuse)."""
    config = config or Config.load()
    artwork = job.get("artwork_pdf")
    if not artwork:
        raise ValueError(f"job {job.get('job_no')}: no artwork PDF")

    geometry = page_geometry(artwork)
    render = render_page(
        artwork,
        config.renders_dir / f"{job['job_no']}.png",
        dpi=config.render_dpi,
    )
    ocr = (ocr_engine or OcrEngine()).read(render.path, px_per_mm=render.px_per_mm)
    regions = cvmod.analyse(render.path)
    roles = assign_roles(ocr, render.path, config=config, transport=transport)

    ticket = parse_proof(job["proof_pdf"]) if job.get("proof_pdf") else None
    esko = parse_info_file(job["info_sidecar"]) if job.get("info_sidecar") else None

    record = fuse(job, ocr, roles, regions, geometry, ticket=ticket, esko=esko)
    record.provenance.render_png = render.path

    # Cross-check: the die size the page claims versus the folder name.
    parsed = job.get("parsed") or {}
    if parsed.get("die_w_mm") and not geometry.matches(
        parsed["die_w_mm"], parsed["die_h_mm"], tolerance_mm=3.0
    ):
        record.provenance.notes.append(
            f"page box {geometry.width_mm:g}x{geometry.height_mm:g}mm does not match "
            f"folder die {parsed['die_w_mm']:g}x{parsed['die_h_mm']:g}mm"
        )
    return record


def run_batch(
    manifest_rows: list[dict],
    config: Config | None = None,
    limit: int | None = None,
    transport: Transport | None = None,
) -> dict:
    """Extract every job in the manifest, resumably.

    A job is skipped when its record already exists with the current extractor
    version — so re-runs after an interruption (or after adding jobs) only do
    new work. Failures are recorded and move on; one broken PDF must not stop
    a corpus run overnight.
    """
    config = config or Config.load()
    config.records_dir.mkdir(parents=True, exist_ok=True)
    engine = OcrEngine()

    done = skipped = failed = 0
    failures: list[dict] = []
    for row in manifest_rows:
        if limit is not None and done >= limit:
            break
        out = config.records_dir / f"{row['job_no']}.json"
        if out.exists():
            try:
                existing = json.loads(out.read_text(encoding="utf-8"))
                if existing.get("provenance", {}).get("extractor_version") == EXTRACTOR_VERSION:
                    skipped += 1
                    continue
            except (json.JSONDecodeError, OSError):
                pass  # unreadable record -> redo it
        try:
            record = extract_label(row, config=config, ocr_engine=engine, transport=transport)
        except Exception as exc:  # noqa: BLE001 - batch must survive anything
            failed += 1
            failures.append({"job_no": row.get("job_no"), "error": str(exc)})
            continue
        out.write_text(record.model_dump_json(indent=2), encoding="utf-8")
        done += 1

    return {"extracted": done, "skipped": skipped, "failed": failed, "failures": failures}
