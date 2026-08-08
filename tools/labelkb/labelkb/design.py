"""CREATE: draw a new label design from what the corpus taught.

This is the deliverable the whole tool exists for: given a generic name, an
actual design — correct die size, every mandatory element placed in its
learned zone at its learned size, statutory wording taken verbatim from
approved precedent — as an element list that renders to SVG and PDF at true
millimetre scale, ready to open in CorelDRAW or Illustrator.

The layout is deterministic, driven by the archetype (median zone map for this
dosage form at this die size) and the generic's own precedent labels. Nothing
is invented: every element carries its source — which jobs it was learned
from, or which statutory duty demands it — and anything the corpus cannot
settle (schedule status of an unseen molecule above all) becomes a CONFIRM
item on the design itself, never a silent guess.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Optional

from pydantic import BaseModel, Field

from .config import Config
from .kb import lookup_generic

# Fallback stacked layout (fractions of the die) for when no archetype exists
# yet. Deliberately plain: it is a starting grid, and it is labelled as such.
_DEFAULT_ZONES: dict[str, tuple[float, float, float, float, float]] = {
    # role: (x0, y0, x1, y1, height_mm)
    "brand": (0.04, 0.06, 0.60, 0.26, 4.0),
    "generic_name": (0.04, 0.30, 0.70, 0.40, 2.0),
    "composition": (0.04, 0.44, 0.70, 0.54, 1.4),
    "statutory_warning": (0.04, 0.58, 0.70, 0.72, 1.2),
    "storage": (0.04, 0.74, 0.70, 0.80, 1.2),
    "licence_no": (0.04, 0.82, 0.45, 0.88, 1.2),
    "mrp": (0.46, 0.82, 0.70, 0.88, 1.2),
    "batch_mfg_exp": (0.04, 0.90, 0.55, 0.96, 1.2),
    "marketer": (0.56, 0.90, 0.92, 0.96, 1.2),
}

RED_BAND_WIDTH_MM = 6.0
DISCLAIMER = (
    "DESIGN DRAFT - generated from approved precedent as a design aid. "
    "Not regulatory sign-off: a qualified person must verify and approve."
)


class Element(BaseModel):
    kind: str  # dieline | red_band | text | placeholder | barcode
    role: str
    bbox_mm: list[float]
    text: Optional[str] = None
    font_mm: Optional[float] = None
    source: str = ""  # where this element came from - jobs or rule
    confirm: bool = False


class Design(BaseModel):
    query: str
    generic_display: str
    dosage_form: Optional[str] = None
    die_w_mm: float
    die_h_mm: float
    status: str  # exact | partial | none
    archetype_key: Optional[str] = None
    archetype_n: int = 0
    precedent_jobs: list[str] = Field(default_factory=list)
    elements: list[Element] = Field(default_factory=list)
    confirm_items: list[str] = Field(default_factory=list)
    inks: list[str] = Field(default_factory=list)
    substrate: Optional[str] = None
    disclaimer: str = DISCLAIMER


def _load_json(path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _most_common(values) -> Optional[str]:
    values = [v for v in values if v]
    return Counter(values).most_common(1)[0][0] if values else None


def _pick_die(die_arg: Optional[str], products: list[dict]) -> tuple[float, float, str]:
    if die_arg:
        w, h = (float(v) for v in die_arg.lower().split("x"))
        return w, h, "requested"
    common = _most_common(p.get("die") for p in products)
    if common:
        w, h = (float(v) for v in common.split("x"))
        return w, h, f"most common among {len(products)} precedent label(s)"
    return 70.0, 30.0, "default (no precedent die size)"


def _zone_to_mm(bbox_norm, die_w, die_h) -> list[float]:
    return [
        round(bbox_norm[0] * die_w, 2),
        round(bbox_norm[1] * die_h, 2),
        round(bbox_norm[2] * die_w, 2),
        round(bbox_norm[3] * die_h, 2),
    ]


def generate_design(
    query_text: str,
    die: Optional[str] = None,
    config: Optional[Config] = None,
) -> Design:
    config = config or Config.load()
    lookup = lookup_generic(query_text, config)
    query = lookup["query"]
    status = lookup["status"]

    products: list[dict] = []
    if status == "exact":
        products = lookup["profile"]["products"]
    elif status == "partial":
        products = [p for entry in lookup["related"] for p in entry["products"]]
    elif status == "none":
        products = [p for entry in lookup.get("same_form", []) for p in entry["products"]]

    die_w, die_h, die_source = _pick_die(die, products)
    dosage_form = query.get("dosage_form") or _most_common(
        p.get("dosage_form") for p in products
    )

    design = Design(
        query=query_text,
        generic_display=query.get("raw") or query_text,
        dosage_form=dosage_form,
        die_w_mm=die_w,
        die_h_mm=die_h,
        status=status,
        precedent_jobs=[p["job_no"] for p in products][:10],
    )

    # ---- archetype ---------------------------------------------------------
    archetypes = _load_json(config.kb / "archetypes.json")
    archetype = None
    key = f"{dosage_form}|{round(die_w):g}x{round(die_h):g}" if dosage_form else None
    if key and key in archetypes:
        archetype = archetypes[key]
        design.archetype_key = key
    elif dosage_form:
        # Same form at another die: zones are normalised, so they transfer.
        for candidate_key, candidate in archetypes.items():
            if candidate_key.startswith(f"{dosage_form}|"):
                archetype = candidate
                design.archetype_key = candidate_key + " (rescaled)"
                break
    if archetype:
        design.archetype_n = archetype["n"]

    def zone_for(role: str) -> tuple[list[float], float, str]:
        if archetype and role in archetype["zones"]:
            zone = archetype["zones"][role]
            return (
                _zone_to_mm(zone["bbox_norm"], die_w, die_h),
                zone["height_mm"],
                f"learned from {zone['support']}/{archetype['n']} labels "
                f"({design.archetype_key})",
            )
        x0, y0, x1, y1, height = _DEFAULT_ZONES[role]
        return (
            _zone_to_mm([x0, y0, x1, y1], die_w, die_h),
            height,
            "default grid (no archetype yet)",
        )

    # ---- compliance decisions ---------------------------------------------
    rx_status = _most_common(p.get("rx_status") for p in products) or "unknown"
    if status != "exact":
        design.confirm_items.append(
            "Schedule status (H/H1/X/OTC) of this molecule is not in the corpus - "
            "CONFIRM against the current schedules before release."
        )
    if rx_status == "unknown" and status == "exact":
        design.confirm_items.append(
            "Precedent labels did not settle Rx/OTC status - CONFIRM."
        )

    warnings = [w for p in products for w in (p.get("statutory_warnings") or [])]
    warning_text = _most_common(warnings)
    red_band_n = sum(1 for p in products if p.get("red_band"))

    # ---- elements ----------------------------------------------------------
    elements = design.elements
    elements.append(
        Element(
            kind="dieline",
            role="dieline",
            bbox_mm=[0, 0, die_w, die_h],
            source=f"die {die_w:g}x{die_h:g} mm - {die_source}",
        )
    )

    if red_band_n and rx_status == "rx":
        elements.append(
            Element(
                kind="red_band",
                role="red_band",
                bbox_mm=[die_w - RED_BAND_WIDTH_MM, 0, die_w, die_h],
                source=f"red band on {red_band_n}/{len(products)} precedent Rx labels",
            )
        )

    brand_bbox, brand_height, brand_source = zone_for("brand")
    elements.append(
        Element(
            kind="placeholder",
            role="brand",
            bbox_mm=brand_bbox,
            text="BRAND NAME",
            font_mm=brand_height,
            source=brand_source + " - replace with the customer's brand",
        )
    )

    generic_bbox, generic_height, generic_source = zone_for("generic_name")
    ratio = (archetype or {}).get("prominence_ratio_median")
    if ratio:
        generic_height = round(brand_height * ratio, 2)
        generic_source += f"; sized at {ratio:g}x brand (median precedent ratio)"
    display = design.generic_display
    elements.append(
        Element(
            kind="text",
            role="generic_name",
            bbox_mm=generic_bbox,
            text=display,
            font_mm=generic_height,
            source=generic_source,
        )
    )

    composition_bbox, composition_height, composition_source = zone_for("composition")
    composition_text = _most_common(
        line for p in products for line in (p.get("composition_raw") or [])
    )
    elements.append(
        Element(
            kind="text" if composition_text else "placeholder",
            role="composition",
            bbox_mm=composition_bbox,
            text=composition_text or "Each ___ contains: ____________",
            font_mm=composition_height,
            source=(
                "wording pattern from precedent" if composition_text
                else "placeholder - supply the approved composition"
            ),
            confirm=not composition_text,
        )
    )

    warning_bbox, warning_height, warning_zone_source = zone_for("statutory_warning")
    if warning_text and rx_status == "rx":
        elements.append(
            Element(
                kind="text",
                role="statutory_warning",
                bbox_mm=warning_bbox,
                text=warning_text,
                font_mm=warning_height,
                source=(
                    f"verbatim from approved precedent "
                    f"({_most_common([p['job_no'] for p in products if warning_text in (p.get('statutory_warnings') or [])]) or 'precedent'}); "
                    + warning_zone_source
                ),
            )
        )
    else:
        elements.append(
            Element(
                kind="placeholder",
                role="statutory_warning",
                bbox_mm=warning_bbox,
                text="[statutory warning per schedule status]",
                font_mm=warning_height,
                source=warning_zone_source,
                confirm=True,
            )
        )
        design.confirm_items.append(
            "No precedent statutory wording available - CONFIRM the exact box text."
        )

    storage_text = _most_common(p.get("storage") for p in products)
    storage_bbox, storage_height, storage_source = zone_for("storage")
    elements.append(
        Element(
            kind="text" if storage_text else "placeholder",
            role="storage",
            bbox_mm=storage_bbox,
            text=storage_text or "Store in a cool dry place. Protect from light.",
            font_mm=storage_height,
            source=storage_source if storage_text else storage_source + " - generic wording, confirm",
            confirm=not storage_text,
        )
    )

    for role, placeholder in [
        ("licence_no", "Mfg. Lic. No. ____________"),
        ("mrp", "M.R.P. Rs. ______ incl. of all taxes"),
        ("batch_mfg_exp", "Batch No.:    Mfg. Dt.:    Exp. Dt.:"),
        ("marketer", "Marketed by: ____________"),
    ]:
        bbox, height, source = zone_for(role)
        elements.append(
            Element(
                kind="placeholder",
                role=role,
                bbox_mm=bbox,
                text=placeholder,
                font_mm=height,
                source=source,
            )
        )

    # Barcode reserve, clear of the red band.
    barcode_width = min(25.0, die_w * 0.3)
    barcode_x1 = die_w - (RED_BAND_WIDTH_MM + 2 if red_band_n else 2)
    elements.append(
        Element(
            kind="barcode",
            role="barcode",
            bbox_mm=[
                round(barcode_x1 - barcode_width, 2),
                round(die_h * 0.55, 2),
                round(barcode_x1, 2),
                round(die_h * 0.80, 2),
            ],
            text="[barcode]",
            source="reserved zone - supply GTIN",
        )
    )

    design.inks = list(
        _most_common(tuple(p.get("inks") or ()) for p in products) or
        ["cyan", "magenta", "yellow", "black"]
    )
    design.substrate = _most_common(p.get("substrate") for p in products)
    return design


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def to_svg(design: Design) -> str:
    """SVG at true millimetre scale - opens in Illustrator/Corel at die size."""
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{design.die_w_mm}mm" '
        f'height="{design.die_h_mm}mm" '
        f'viewBox="0 0 {design.die_w_mm} {design.die_h_mm}">',
        f"<!-- {_svg_escape(design.disclaimer)} -->",
        f"<!-- generated for: {_svg_escape(design.query)}; "
        f"precedent jobs: {', '.join(design.precedent_jobs) or 'none'} -->",
        f'<rect x="0" y="0" width="{design.die_w_mm}" height="{design.die_h_mm}" '
        f'fill="white"/>',
    ]
    for element in design.elements:
        x0, y0, x1, y1 = element.bbox_mm
        width, height = round(x1 - x0, 3), round(y1 - y0, 3)
        if element.kind == "dieline":
            parts.append(
                f'<rect x="0" y="0" width="{design.die_w_mm}" height="{design.die_h_mm}" '
                f'fill="none" stroke="#00a651" stroke-width="0.2">'
                f"<title>dieline (Cut)</title></rect>"
            )
            continue
        if element.kind == "red_band":
            parts.append(
                f'<rect x="{x0}" y="{y0}" width="{width}" height="{height}" '
                f'fill="#d92626"><title>{_svg_escape(element.source)}</title></rect>'
            )
            continue
        if element.kind == "barcode":
            parts.append(
                f'<rect x="{x0}" y="{y0}" width="{width}" height="{height}" '
                f'fill="none" stroke="#888" stroke-width="0.15" stroke-dasharray="1,1"/>'
                f'<text x="{x0 + width / 2}" y="{(y0 + y1) / 2}" font-size="1.6" '
                f'fill="#888" text-anchor="middle">[barcode]</text>'
            )
            continue

        colour = "#0066cc" if element.kind == "placeholder" else "#111"
        dash = ' stroke-dasharray="0.8,0.8"' if element.kind == "placeholder" else ""
        parts.append(
            f'<rect x="{x0}" y="{y0}" width="{width}" height="{height}" fill="none" '
            f'stroke="{colour}" stroke-width="0.1" opacity="0.55"{dash}>'
            f"<title>{_svg_escape(element.role)}: {_svg_escape(element.source)}</title></rect>"
        )
        font = element.font_mm or 1.5
        baseline = min(y0 + font * 1.1, y1 - 0.2)
        parts.append(
            f'<text x="{x0 + 0.4}" y="{round(baseline, 2)}" font-size="{font}" '
            f'font-family="Helvetica, Arial, sans-serif" fill="{colour}">'
            f"{_svg_escape(element.text or '')}</text>"
        )
    parts.append("</svg>")
    return "\n".join(parts)


def to_pdf(design: Design, out_path) -> None:
    """The same design as a PDF page at exact die size."""
    import pymupdf

    mm = 72 / 25.4
    doc = pymupdf.open()
    page = doc.new_page(width=design.die_w_mm * mm, height=design.die_h_mm * mm)

    for element in design.elements:
        x0, y0, x1, y1 = (v * mm for v in element.bbox_mm)
        rect = pymupdf.Rect(x0, y0, x1, y1)
        if element.kind == "dieline":
            page.draw_rect(rect, color=(0, 0.65, 0.32), width=0.6)
            continue
        if element.kind == "red_band":
            page.draw_rect(rect, color=(0.85, 0.15, 0.15), fill=(0.85, 0.15, 0.15))
            continue
        if element.kind == "barcode":
            page.draw_rect(rect, color=(0.5, 0.5, 0.5), width=0.4, dashes="[2 2] 0")
            continue
        colour = (0, 0.4, 0.8) if element.kind == "placeholder" else (0.07, 0.07, 0.07)
        page.draw_rect(rect, color=colour, width=0.3)
        font = (element.font_mm or 1.5) * mm
        page.insert_text(
            (x0 + 1, min(y0 + font * 1.05, y1 - 1)),
            element.text or "",
            fontsize=font,
            color=colour,
        )

    doc.set_metadata({"title": f"labelkb draft - {design.query}", "subject": DISCLAIMER})
    doc.save(out_path)
    doc.close()
