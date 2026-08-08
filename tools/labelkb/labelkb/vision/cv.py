"""Classical CV for the label furniture OCR and VLMs are bad at.

Three detectors, each targeting a structure with regulatory meaning:

* **Red band** — the vertical/horizontal red stripe that Rule 96/97 practice
  puts on Schedule H/H1 packs. It is a colour-and-shape fact, not a text fact:
  a saturated red region, elongated, touching a label edge. Exact in OpenCV,
  unreliable to ask of a language model.

* **Ruled boxes** — statutory warnings sit inside hairline rectangles. Finding
  the rectangles (and later, which OCR text they enclose) turns "warning text
  somewhere" into "warning text in its box", which is what approval looks for.

* **Barcode zones** — dense runs of parallel vertical strokes. Detected by the
  classic gradient trick: strong x-gradients minus weak y-gradients, closed
  into blobs.

Everything reports in render pixels; callers convert with the render's
px_per_mm exactly as the OCR layer does.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np


@dataclass
class Region:
    """A detected region in render pixels."""

    kind: str  # red_band | ruled_box | barcode
    x0: int
    y0: int
    x1: int
    y1: int
    score: float  # detector-specific: coverage/solidity, in [0, 1]
    edge_touching: bool = False

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def aspect(self) -> float:
        short = min(self.width, self.height)
        return (max(self.width, self.height) / short) if short else 0.0

    def bbox_mm(self, px_per_mm: float) -> list[float]:
        return [round(v / px_per_mm, 3) for v in (self.x0, self.y0, self.x1, self.y1)]

    def contains(self, x: float, y: float) -> bool:
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1

    def to_dict(self) -> dict:
        return asdict(self)


def load_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not read image: {path}")
    return image


def _touches_edge(x0: int, y0: int, x1: int, y1: int, shape, margin_frac: float = 0.02) -> bool:
    height, width = shape[:2]
    margin = max(2, int(min(height, width) * margin_frac))
    return x0 <= margin or y0 <= margin or x1 >= width - margin or y1 >= height - margin


def find_red_bands(
    image: np.ndarray,
    min_area_frac: float = 0.002,
    min_aspect: float = 3.0,
) -> list[Region]:
    """Saturated-red, elongated regions - the Schedule H band candidates.

    ``min_aspect`` keeps red brand panels and red hero graphics out: the
    statutory band is a stripe, not a block. Edge contact is reported rather
    than required, because bands are sometimes inset by a bleed-safe margin.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Red wraps around the hue axis, so it needs two ranges.
    lower = cv2.inRange(hsv, (0, 120, 70), (10, 255, 255))
    upper = cv2.inRange(hsv, (170, 120, 70), (180, 255, 255))
    mask = cv2.morphologyEx(
        lower | upper, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    )

    min_area = image.shape[0] * image.shape[1] * min_area_frac
    regions: list[Region] = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h < min_area or min(w, h) == 0:
            continue
        if max(w, h) / min(w, h) < min_aspect:
            continue
        coverage = cv2.contourArea(contour) / (w * h)
        regions.append(
            Region(
                kind="red_band",
                x0=x,
                y0=y,
                x1=x + w,
                y1=y + h,
                score=round(float(coverage), 4),
                edge_touching=_touches_edge(x, y, x + w, y + h, image.shape),
            )
        )
    regions.sort(key=lambda r: r.score, reverse=True)
    return regions


def find_ruled_boxes(
    image: np.ndarray,
    min_area_frac: float = 0.003,
    max_area_frac: float = 0.6,
) -> list[Region]:
    """Hairline rectangles - the frames statutory warnings sit inside.

    A ruled box is a *stroked* rectangle: its contour approximates four
    corners, and the region inside is mostly not ink. The solidity check
    (contour area versus bounding area) rejects filled blocks, which also
    produce four-corner contours.
    """
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Inverted adaptive threshold: ink -> white, regardless of label colour.
    ink = cv2.adaptiveThreshold(
        grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15
    )

    area = image.shape[0] * image.shape[1]
    regions: list[Region] = []
    contours, _ = cv2.findContours(ink, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        box_area = w * h
        if box_area < area * min_area_frac or box_area > area * max_area_frac:
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        # Stroke, not slab — judged on the grey image, NOT the adaptive
        # threshold: adaptive thresholding hollows out large filled regions
        # (only their edges register as ink), which made slabs indistinguishable
        # from frames. On grey the test is direct: a frame's interior is a
        # different brightness from its stroke; a slab's interior IS the stroke.
        stroke_w = max(3, int(min(w, h) * 0.08))
        interior = grey[y + stroke_w : y + h - stroke_w, x + stroke_w : x + w - stroke_w]
        if interior.size == 0:
            continue
        # The ring is bbox-minus-interior — strictly inside the bounding rect.
        # Drawing a rectangle outline as the mask instead spills onto the
        # background outside the shape, which lends a filled slab exactly the
        # stroke/interior contrast it does not have.
        crop = grey[y : y + h, x : x + w].astype(np.float64)
        total = crop.sum()
        interior_sum = interior.astype(np.float64).sum()
        ring_pixels = crop.size - interior.size
        if ring_pixels <= 0:
            continue
        ring_mean = (total - interior_sum) / ring_pixels
        interior_mean = float(interior.mean())
        contrast = abs(interior_mean - ring_mean)
        if contrast < 40:
            continue  # interior matches the stroke -> solid panel, not a frame
        regions.append(
            Region(
                kind="ruled_box",
                x0=x,
                y0=y,
                x1=x + w,
                y1=y + h,
                score=round(min(contrast / 255.0, 1.0), 4),
            )
        )
    # Nested duplicates (inner+outer edge of the same stroke) collapse to the outer.
    regions.sort(key=lambda r: (r.width * r.height), reverse=True)
    kept: list[Region] = []
    for region in regions:
        if any(
            k.contains(region.x0 + 3, region.y0 + 3) and k.contains(region.x1 - 3, region.y1 - 3)
            for k in kept
        ):
            continue
        kept.append(region)
    return kept


def _bar_uniformity(grey_crop: np.ndarray) -> float:
    """How much this crop looks like a 1D barcode, in [0, 1].

    The property that separates bars from everything else on a label: in a
    true 1D code, every inked column spans (nearly) the full region height,
    uniformly. Text fails this hard — glyph columns have wildly varying ink
    spans — which is exactly the false positive that sank the earlier
    edge-density heuristic (text lines scored 0.9; the real barcode scored 0).
    """
    ink = grey_crop < 128
    heights = ink.sum(axis=0)
    inked = heights[heights > 0]
    if inked.size < 10:
        return 0.0
    full = float(grey_crop.shape[0])
    span = float(np.median(inked)) / full  # bars reach ~1.0
    spread = float(np.std(inked)) / full  # bars ~0.0
    # Bars alternate: a healthy code has many ink runs across the width.
    transitions = int(np.count_nonzero(np.diff((heights > 0).astype(np.int8))))
    if transitions < 8:
        return 0.0
    return max(0.0, span - spread)


def _builtin_barcode_regions(image: np.ndarray) -> list[Region] | None:
    """OpenCV's own detector, when this build ships it. Returns None if absent
    or if it finds nothing (the heuristic then gets its turn — the built-in
    detector only reads clean, complete codes)."""
    detector_cls = getattr(getattr(cv2, "barcode", None), "BarcodeDetector", None)
    if detector_cls is None:
        return None
    try:
        found, _, _, corners = detector_cls().detectAndDecodeWithType(image)
    except cv2.error:  # pragma: no cover - detector quirk, fall through
        return None
    if not found or corners is None:
        return None
    regions = []
    for quad in corners:
        xs, ys = quad[:, 0], quad[:, 1]
        regions.append(
            Region(
                kind="barcode",
                x0=int(xs.min()),
                y0=int(ys.min()),
                x1=int(xs.max()),
                y1=int(ys.max()),
                score=1.0,
            )
        )
    return regions or None


def find_barcodes(
    image: np.ndarray,
    min_area_frac: float = 0.002,
    min_uniformity: float = 0.55,
) -> list[Region]:
    """Zones of parallel full-height bars - 1D barcode candidates."""
    builtin = _builtin_barcode_regions(image)
    if builtin:
        return builtin

    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ink = cv2.threshold(grey, 128, 255, cv2.THRESH_BINARY_INV)[1]
    # Bridge the gaps between bars so a code becomes one blob. The kernel must
    # span the widest legitimate gap in a 1D code (a few module widths at this
    # resolution) or the code fragments into tall-thin blobs that then fail the
    # wider-than-tall test. Fusing neighbouring text is harmless — the
    # uniformity filter rejects text regardless of how it is blobbed.
    closed = cv2.morphologyEx(
        ink, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    )

    min_area = image.shape[0] * image.shape[1] * min_area_frac
    regions: list[Region] = []
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h < min_area or h == 0 or w < h:
            continue  # 1D codes are wider than tall
        uniformity = _bar_uniformity(grey[y : y + h, x : x + w])
        if uniformity < min_uniformity:
            continue
        regions.append(
            Region(kind="barcode", x0=x, y0=y, x1=x + w, y1=y + h, score=round(uniformity, 4))
        )
    regions.sort(key=lambda r: r.score, reverse=True)
    return regions


def analyse(image_path: str | Path) -> dict[str, list[Region]]:
    """Run all three detectors over one render."""
    image = load_image(image_path)
    return {
        "red_bands": find_red_bands(image),
        "ruled_boxes": find_ruled_boxes(image),
        "barcodes": find_barcodes(image),
    }
