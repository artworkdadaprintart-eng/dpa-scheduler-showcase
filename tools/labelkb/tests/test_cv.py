"""Tests for the CV detectors, on synthetic labels drawn with OpenCV.

Each fixture draws the structure the detector exists for plus the decoys it
must ignore: a red brand *panel* next to the statutory red *band*, a filled
slab next to the warning *frame*, body text next to the *barcode*.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from labelkb.vision.cv import (
    Region,
    analyse,
    find_barcodes,
    find_red_bands,
    find_ruled_boxes,
)

WHITE = (255, 255, 255)
RED = (40, 40, 230)  # BGR


def blank(width=700, height=450):
    return np.full((height, width, 3), 255, dtype=np.uint8)


class TestRedBand:
    def test_vertical_band_at_edge_is_found(self):
        image = blank()
        cv2.rectangle(image, (660, 0), (699, 449), RED, -1)  # right-edge stripe
        bands = find_red_bands(image)
        assert len(bands) == 1
        band = bands[0]
        assert band.edge_touching
        assert band.aspect > 5
        assert band.x0 >= 655

    def test_square_red_panel_is_rejected(self):
        """A red brand block is not a statutory band."""
        image = blank()
        cv2.rectangle(image, (100, 100), (300, 300), RED, -1)
        assert find_red_bands(image) == []

    def test_band_and_panel_together_yield_only_the_band(self):
        image = blank()
        cv2.rectangle(image, (100, 100), (300, 300), RED, -1)  # decoy panel
        cv2.rectangle(image, (0, 0), (30, 449), RED, -1)  # left-edge band
        bands = find_red_bands(image)
        assert len(bands) == 1
        assert bands[0].x1 <= 40

    def test_blue_stripe_is_not_red(self):
        image = blank()
        cv2.rectangle(image, (660, 0), (699, 449), (230, 40, 40), -1)
        assert find_red_bands(image) == []


class TestRuledBox:
    def test_hairline_frame_is_found(self):
        image = blank()
        cv2.rectangle(image, (60, 300), (640, 420), (0, 0, 0), 2)  # warning frame
        boxes = find_ruled_boxes(image)
        assert len(boxes) == 1
        box = boxes[0]
        assert abs(box.x0 - 60) < 6 and abs(box.y1 - 420) < 6

    def test_filled_slab_is_rejected(self):
        image = blank()
        cv2.rectangle(image, (60, 300), (640, 420), (0, 0, 0), -1)
        assert find_ruled_boxes(image) == []

    def test_frame_with_text_inside_still_reads_as_frame(self):
        image = blank()
        cv2.rectangle(image, (60, 280), (640, 430), (0, 0, 0), 2)
        cv2.putText(
            image,
            "SCHEDULE H WARNING",
            (85, 360),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            2,
        )
        boxes = find_ruled_boxes(image)
        assert len(boxes) == 1
        assert boxes[0].contains(300, 355)

    def test_contains_maps_ocr_text_into_the_box(self):
        region = Region(kind="ruled_box", x0=60, y0=280, x1=640, y1=430, score=0.9)
        assert region.contains(300, 350)
        assert not region.contains(300, 100)

    def test_bbox_mm_conversion(self):
        region = Region(kind="ruled_box", x0=0, y0=0, x1=236, y1=118, score=1.0)
        assert region.bbox_mm(px_per_mm=11.811) == pytest.approx(
            [0, 0, 19.982, 9.991], abs=0.01
        )


class TestBarcode:
    @staticmethod
    def _draw_barcode(image, x=80, y=320, width=240, height=90):
        """EAN-like density: alternating bars and gaps of 1-4 modules, no long
        voids - a real 1D code never has one."""
        rng = np.random.default_rng(7)
        cursor = x
        while cursor < x + width:
            bar = int(rng.integers(2, 7))
            cv2.rectangle(image, (cursor, y), (cursor + bar, y + height), (0, 0, 0), -1)
            cursor += bar + int(rng.integers(2, 7))

    def test_bar_pattern_is_found(self):
        image = blank()
        self._draw_barcode(image)
        codes = find_barcodes(image)
        assert codes, "barcode zone not detected"
        best = codes[0]
        assert best.contains(200, 360)
        assert best.width > best.height

    def test_plain_text_is_not_a_barcode(self):
        image = blank()
        for row, text in enumerate(
            ["Each ml contains", "Diclofenac Sodium IP 25mg", "Water for Injection IP"]
        ):
            cv2.putText(
                image, text, (60, 120 + row * 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 1
            )
        assert find_barcodes(image) == []


class TestAnalyse:
    def test_full_label_reports_all_three(self, tmp_path):
        image = blank()
        cv2.rectangle(image, (0, 0), (30, 449), RED, -1)  # band
        cv2.rectangle(image, (330, 300), (640, 430), (0, 0, 0), 2)  # frame
        TestBarcode._draw_barcode(image, x=380, y=60, width=220, height=80)
        path = tmp_path / "label.png"
        cv2.imwrite(str(path), image)

        result = analyse(path)
        assert len(result["red_bands"]) == 1
        assert len(result["ruled_boxes"]) == 1
        assert len(result["barcodes"]) >= 1

    def test_unreadable_path_raises_cleanly(self, tmp_path):
        with pytest.raises(ValueError, match="could not read"):
            analyse(tmp_path / "missing.png")
