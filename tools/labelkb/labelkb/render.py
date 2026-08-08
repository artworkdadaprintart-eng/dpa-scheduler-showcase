"""Rasterise artwork PDFs and recover their true physical size.

Two jobs, both foundational:

* **Geometry.** The page box gives the label's real dimensions in millimetres.
  Everything downstream that matters for compliance — minimum type heights, the
  generic-versus-brand size ratio — is a measurement, and a measurement needs a
  scale. Get this wrong and every derived number is wrong with it, silently.

* **Rendering.** The artwork is outlines, so it has to become pixels before OCR
  or a vision model can read it. 600 dpi is the default for a reason: a 19 mm
  tall label lands around 450 px, and 1 mm statutory type still comes out near
  24 px, which OCR can hold. Halve the resolution and the fine print — the part
  that carries the legal obligations — is the first thing to dissolve.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import pymupdf

from .config import DEFAULT_RENDER_DPI, MAX_RENDER_LONG_EDGE_PX

MM_PER_INCH = 25.4
PT_PER_INCH = 72.0


def pt_to_mm(points: float) -> float:
    return points / PT_PER_INCH * MM_PER_INCH


@dataclass
class PageGeometry:
    """Physical size of an artwork page, and the boxes it declares."""

    page_count: int
    width_mm: float
    height_mm: float
    width_pt: float
    height_pt: float
    trim_w_mm: float | None = None
    trim_h_mm: float | None = None
    rotation: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    def matches(self, w_mm: float, h_mm: float, tolerance_mm: float = 1.5) -> bool:
        """Does this page match an expected die size, in either orientation?

        Real production pages are the die *plus bleed*: job 88116's page box
        measures 32x21 mm around its 30x19 die — 1 mm per side. So a match is
        either the die itself (folder names round to whole millimetres) or the
        die grown by a uniform bleed on all sides.
        """
        return self.bleed_mm(w_mm, h_mm, tolerance_mm) is not None

    def bleed_mm(
        self, w_mm: float, h_mm: float, tolerance_mm: float = 1.5, max_bleed_mm: float = 3.0
    ) -> float | None:
        """Per-side bleed implied by the expected die, or None if no match.

        0.0 means the page IS the die. A positive value means the page is the
        die grown uniformly — the same excess on both axes, within tolerance —
        by up to ``max_bleed_mm`` per side. Non-uniform excess is a genuine
        mismatch, not bleed.
        """
        boxes = [(self.width_mm, self.height_mm)]
        if self.trim_w_mm and self.trim_h_mm:
            # The TrimBox, when present, is the die itself - check it first.
            boxes.insert(0, (self.trim_w_mm, self.trim_h_mm))
        for page_w, page_h in boxes:
            for die_w, die_h in ((w_mm, h_mm), (h_mm, w_mm)):
                excess_w = page_w - die_w
                excess_h = page_h - die_h
                if abs(excess_w) <= tolerance_mm and abs(excess_h) <= tolerance_mm:
                    return 0.0
                uniform = abs(excess_w - excess_h) <= tolerance_mm
                if (
                    uniform
                    and 0 < excess_w <= 2 * max_bleed_mm + tolerance_mm
                    and 0 < excess_h <= 2 * max_bleed_mm + tolerance_mm
                ):
                    return round((excess_w + excess_h) / 4, 2)
        return None


def page_geometry(pdf_path: str | Path, page_index: int = 0) -> PageGeometry:
    with pymupdf.open(Path(pdf_path)) as doc:
        if doc.page_count == 0:
            raise ValueError(f"{pdf_path}: PDF has no pages")
        page = doc[page_index]
        rect = page.rect
        geometry = PageGeometry(
            page_count=doc.page_count,
            width_mm=round(pt_to_mm(rect.width), 3),
            height_mm=round(pt_to_mm(rect.height), 3),
            width_pt=round(rect.width, 3),
            height_pt=round(rect.height, 3),
            rotation=page.rotation,
        )
        try:
            trim = page.trimbox
            if trim and trim.width and trim.height:
                geometry.trim_w_mm = round(pt_to_mm(trim.width), 3)
                geometry.trim_h_mm = round(pt_to_mm(trim.height), 3)
        except (AttributeError, ValueError):
            pass
        return geometry


@dataclass
class Render:
    path: str
    dpi: float
    width_px: int
    height_px: int
    px_per_mm: float

    def to_dict(self) -> dict:
        return asdict(self)

    def px_to_mm(self, pixels: float) -> float:
        """Convert a measured pixel distance back to millimetres."""
        return pixels / self.px_per_mm


def effective_dpi(geometry: PageGeometry, requested_dpi: int) -> float:
    """Back off the requested dpi if it would produce an unreasonable bitmap."""
    long_edge_in = max(geometry.width_pt, geometry.height_pt) / PT_PER_INCH
    if long_edge_in <= 0:
        return float(requested_dpi)
    max_dpi = MAX_RENDER_LONG_EDGE_PX / long_edge_in
    return float(min(requested_dpi, max_dpi))


def render_page(
    pdf_path: str | Path,
    out_png: str | Path,
    dpi: int = DEFAULT_RENDER_DPI,
    page_index: int = 0,
) -> Render:
    """Render one page to PNG, returning the scale needed to measure it."""
    pdf_path, out_png = Path(pdf_path), Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    geometry = page_geometry(pdf_path, page_index)
    used_dpi = effective_dpi(geometry, dpi)

    with pymupdf.open(pdf_path) as doc:
        page = doc[page_index]
        zoom = used_dpi / PT_PER_INCH
        pixmap = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
        pixmap.save(out_png)

    return Render(
        path=str(out_png),
        dpi=round(used_dpi, 3),
        width_px=pixmap.width,
        height_px=pixmap.height,
        px_per_mm=round(used_dpi / MM_PER_INCH, 6),
    )


def has_extractable_text(pdf_path: str | Path, page_index: int = 0) -> bool:
    """Whether the page carries live text.

    Across this corpus the answer for production artwork is no — type is
    converted to curves — which is precisely why the vision pipeline exists. It
    is still worth asking per file: a customer-supplied PDF occasionally arrives
    with live text, and when it does, that text beats anything OCR can produce.
    """
    with pymupdf.open(Path(pdf_path)) as doc:
        if doc.page_count == 0:
            return False
        return bool(doc[page_index].get_text("text").strip())
