"""Tests for the OCR layer.

Real OCR runs against a synthetic label rendered by our own renderer, so the
whole chain px->mm is exercised: known font sizes in -> measured millimetres
out. Tolerances are honest about what an OCR box is (glyph extents plus
padding, not cap height): ratios use wide bands, absolute heights use a band
derived from the font metrics.
"""

from __future__ import annotations

import re

import pymupdf
import pytest

from labelkb.render import render_page
from labelkb.vision.ocr import CAP_HEIGHT_FACTOR, OcrEngine, TextBox, group_lines

BRAND_PT = 14
GENERIC_PT = 6
FINE_PT = 4


@pytest.fixture(scope="module")
def label(tmp_path_factory):
    """A 60x40 mm label with text at known sizes, rendered at 300 dpi."""
    tmp = tmp_path_factory.mktemp("ocr")
    doc = pymupdf.open()
    page = doc.new_page(width=60 / 25.4 * 72, height=40 / 25.4 * 72)
    page.insert_text((10, 20), "JARODOL", fontsize=BRAND_PT)
    page.insert_text((10, 32), "Diclofenac Sodium Injection IP", fontsize=GENERIC_PT)
    page.insert_text((10, 44), "SCHEDULE H PRESCRIPTION DRUG", fontsize=FINE_PT)
    page.insert_text((10, 54), "Mfg. Lic. No. MNB/25/417", fontsize=FINE_PT)
    pdf = tmp / "label.pdf"
    doc.save(pdf)
    doc.close()

    render = render_page(pdf, tmp / "label.png", dpi=300)
    result = OcrEngine().read(render.path, px_per_mm=render.px_per_mm)
    return render, result


def _squash(text: str) -> str:
    """RapidOCR merges inter-word spaces unpredictably; compare without them."""
    return re.sub(r"\s+", "", text).lower()


class TestReading:
    def test_all_four_lines_found(self, label):
        _, result = label
        assert len(result.lines) == 4

    def test_brand_read_exactly(self, label):
        _, result = label
        assert result.lines[0].text == "JARODOL"

    def test_fine_print_read_at_one_and_a_half_mm(self, label):
        """The statutory line is ~1.4 mm tall at this size - it must survive."""
        _, result = label
        texts = [_squash(l.text) for l in result.lines]
        assert any("scheduleh" in t for t in texts)
        assert any("mnb/25/417" in t for t in texts)

    def test_confidence_is_carried(self, label):
        _, result = label
        assert all(l.confidence > 0.8 for l in result.lines)


class TestMeasurement:
    def test_brand_height_in_mm(self, label):
        """14 pt caps-only: cap height ~3.5 mm, nominal em 4.94 mm. The OCR box
        lands between the two; detector padding differs between the legacy and
        current rapidocr packages (3.98 vs 4.83 mm on this fixture), so the
        band covers both — anything inside it converts to sane compliance
        measurements via CAP_HEIGHT_FACTOR."""
        render, result = label
        brand = result.lines[0]
        measured = brand.height_px / render.px_per_mm
        assert 3.0 < measured < 5.0, f"measured {measured:.2f} mm"

    def test_prominence_ratio_survives_measurement(self, label):
        """Brand (14 pt) vs generic (6 pt): the ratio compliance turns on.
        Boxes measure glyph extents, so 2.33 in metal comes out lower in
        pixels - but it must stay clearly above 1 and roughly proportional."""
        render, result = label
        brand, generic = result.lines[0], result.lines[1]
        ratio = brand.height_px / generic.height_px
        assert 1.4 < ratio < 3.2, f"ratio {ratio:.2f}"

    def test_mm_conversion_round_trip(self, label):
        render, result = label
        box = result.boxes[0]
        assert box.height_mm(render.px_per_mm) == pytest.approx(
            box.height_px / render.px_per_mm
        )
        assert box.cap_height_mm(render.px_per_mm) == pytest.approx(
            box.height_mm(render.px_per_mm) * CAP_HEIGHT_FACTOR
        )

    def test_to_dict_includes_mm(self, label):
        _, result = label
        data = result.to_dict()
        assert data["boxes"][0]["height_mm"] > 0
        assert len(data["boxes"][0]["bbox_mm"]) == 4


class TestLineGrouping:
    def _box(self, text, x0, y0, x1, y1):
        return TextBox(text=text, confidence=1.0, x0=x0, y0=y0, x1=x1, y1=y1)

    def test_same_baseline_groups(self):
        lines = group_lines(
            [self._box("Mfg.", 0, 100, 40, 120), self._box("Lic.", 50, 101, 90, 121)]
        )
        assert len(lines) == 1
        assert lines[0].text == "Mfg. Lic."

    def test_different_baselines_do_not(self):
        lines = group_lines(
            [self._box("JARODOL", 0, 10, 100, 40), self._box("Injection", 0, 60, 80, 75)]
        )
        assert [l.text for l in lines] == ["JARODOL", "Injection"]

    def test_reading_order_is_top_down_left_right(self):
        lines = group_lines(
            [
                self._box("second", 0, 50, 40, 60),
                self._box("first", 0, 10, 40, 20),
                self._box("line", 50, 51, 80, 61),
            ]
        )
        assert [l.text for l in lines] == ["first", "second line"]
