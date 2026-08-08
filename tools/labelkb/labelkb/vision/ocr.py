"""OCR over artwork renders, with every box convertible to millimetres.

Division of labour (the core design decision of the pipeline): OCR is the only
component trusted to *read* — it works on the full 600-dpi render where 1 mm
statutory type is ~24 px and legible. The VLM never reads small print; it only
assigns roles to the boxes found here. So whatever OCR misses is genuinely
missing, which is why every word keeps its confidence and nothing is silently
dropped.

Measurements: RapidOCR returns quadrilateral boxes in pixels. With the render's
``px_per_mm`` (known exactly, because we rendered the PDF ourselves) a box
height becomes a physical type height. Compliance arguments — minimum letter
heights, the generic-vs-brand prominence ratio — are made in millimetres, so
the conversion happens here, once, rather than ad hoc downstream.

A box height is not a cap height: it spans ascender to descender plus a little
padding. Ratios between boxes are unaffected (both sides scale the same way);
absolute minimum-height checks use ``CAP_HEIGHT_FACTOR`` as a stated, testable
approximation rather than a hidden fudge.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path

# Fraction of an OCR box height that is actual capital-letter height, the rest
# being ascender/descender room and detector padding. Empirically ~0.70 for the
# Latin text on these labels; kept as a named constant so a calibration pass on
# the golden set can tune one number instead of hunting magic values.
CAP_HEIGHT_FACTOR = 0.70

# Words on the same line have vertical centres within this fraction of the
# taller box's height.
_LINE_OVERLAP = 0.6


@dataclass
class TextBox:
    """One OCR detection, in render pixels."""

    text: str
    confidence: float
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def height_px(self) -> float:
        return self.y1 - self.y0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    def height_mm(self, px_per_mm: float) -> float:
        return self.height_px / px_per_mm

    def cap_height_mm(self, px_per_mm: float) -> float:
        return self.height_mm(px_per_mm) * CAP_HEIGHT_FACTOR

    def bbox_mm(self, px_per_mm: float) -> tuple[float, float, float, float]:
        return (
            self.x0 / px_per_mm,
            self.y0 / px_per_mm,
            self.x1 / px_per_mm,
            self.y1 / px_per_mm,
        )


@dataclass
class Line:
    """OCR boxes grouped into a reading line."""

    boxes: list[TextBox] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(b.text for b in sorted(self.boxes, key=lambda b: b.x0))

    @property
    def x0(self) -> float:
        return min(b.x0 for b in self.boxes)

    @property
    def y0(self) -> float:
        return min(b.y0 for b in self.boxes)

    @property
    def x1(self) -> float:
        return max(b.x1 for b in self.boxes)

    @property
    def y1(self) -> float:
        return max(b.y1 for b in self.boxes)

    @property
    def height_px(self) -> float:
        return self.y1 - self.y0

    @property
    def confidence(self) -> float:
        return min(b.confidence for b in self.boxes)


@dataclass
class OcrResult:
    boxes: list[TextBox]
    lines: list[Line]
    px_per_mm: float | None = None

    @property
    def full_text(self) -> str:
        return "\n".join(line.text for line in self.lines)

    def to_dict(self) -> dict:
        scale = self.px_per_mm

        def box_dict(box: TextBox) -> dict:
            data = asdict(box)
            if scale:
                data["height_mm"] = round(box.height_mm(scale), 3)
                data["cap_height_mm"] = round(box.cap_height_mm(scale), 3)
                data["bbox_mm"] = [round(v, 3) for v in box.bbox_mm(scale)]
            return data

        return {
            "px_per_mm": scale,
            "boxes": [box_dict(b) for b in self.boxes],
            "lines": [
                {
                    "text": line.text,
                    "bbox_px": [line.x0, line.y0, line.x1, line.y1],
                    "height_mm": round(line.height_px / scale, 3) if scale else None,
                    "confidence": round(line.confidence, 4),
                }
                for line in self.lines
            ],
        }


def group_lines(boxes: list[TextBox]) -> list[Line]:
    """Cluster boxes into reading lines by vertical-centre proximity."""
    lines: list[Line] = []
    for box in sorted(boxes, key=lambda b: (b.cy, b.x0)):
        for line in lines:
            reference = max(line.height_px, box.height_px, 1.0)
            if abs(box.cy - (line.y0 + line.y1) / 2) < reference * _LINE_OVERLAP:
                line.boxes.append(box)
                break
        else:
            lines.append(Line(boxes=[box]))
    lines.sort(key=lambda l: (l.y0, l.x0))
    return lines


class OcrEngine:
    """RapidOCR wrapper. Lazy: the ONNX models load on first use, not import,
    so the base install (no [ocr] extra) can still import this module."""

    def __init__(self) -> None:
        self._engine = None

    def _load(self):
        if self._engine is None:
            try:
                from rapidocr_onnxruntime import RapidOCR
            except ImportError as exc:  # pragma: no cover - environment specific
                raise RuntimeError(
                    "RapidOCR is not installed. Run: pip install -e '.[ocr]'"
                ) from exc
            self._engine = RapidOCR()
        return self._engine

    def read(self, image_path: str | Path, px_per_mm: float | None = None) -> OcrResult:
        engine = self._load()
        raw, _ = engine(str(image_path))

        boxes: list[TextBox] = []
        for quad, text, score in raw or []:
            xs = [point[0] for point in quad]
            ys = [point[1] for point in quad]
            text = (text or "").strip()
            if not text:
                continue
            boxes.append(
                TextBox(
                    text=text,
                    confidence=float(score),
                    x0=min(xs),
                    y0=min(ys),
                    x1=max(xs),
                    y1=max(ys),
                )
            )
        return OcrResult(boxes=boxes, lines=group_lines(boxes), px_per_mm=px_per_mm)
