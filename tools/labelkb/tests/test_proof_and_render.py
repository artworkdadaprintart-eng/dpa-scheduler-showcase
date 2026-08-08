"""Tests for the proof-ticket parser and the PDF renderer.

The field names and values here are the real ones read off job 88116's proof.
What is synthesised is the *geometry*: the real proof could not be pulled into
this environment (a 600 KB PDF is far too large to move through the connector),
so each ticket is rebuilt in both layouts a form like this plausibly uses —
value to the right of its label, and value beneath it. The parser has to handle
either, which is exactly the uncertainty that made a geometry-driven parser the
right choice over one that trusts string order.

Field *mapping* against a genuine proof is a one-command check on the corpus
machine: ``labelkb proof <pdf>``.
"""

from __future__ import annotations

import pymupdf
import pytest

from labelkb.corpus.proof import parse_proof
from labelkb.render import (
    PageGeometry,
    effective_dpi,
    has_extractable_text,
    page_geometry,
    pt_to_mm,
    render_page,
)

TICKET = [
    ("Artwork No:", "88116"),
    ("Job Name:", "88116 - JARODOL 1ML 30X19"),
    ("Customer Name:", "Smayan Healthcare"),
    ("Size:", "30 x 19"),
    ("Repeat:", "297"),
    ("No of Colors:", "CMYK+V"),
    ("New Die:", "497"),
    ("Dated:", "07-Aug-2026"),
    ("Substrate:", "CHROMO 75 - STANDARD"),
]

FONT_SIZE = 9


def _write_proof(path, layout: str):
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    y = 100.0
    for label, value in TICKET:
        page.insert_text((60, y), label, fontsize=FONT_SIZE)
        if layout == "right":
            page.insert_text((230, y), value, fontsize=FONT_SIZE)
            y += 26
        else:
            page.insert_text((60, y + 13), value, fontsize=FONT_SIZE)
            y += 44
    doc.save(path)
    doc.close()
    return path


@pytest.mark.parametrize("layout", ["right", "below"])
class TestProofTicket:
    def test_reads_every_field(self, tmp_path, layout):
        pdf = _write_proof(tmp_path / f"proof_{layout}.pdf", layout)
        ticket = parse_proof(pdf)

        assert ticket.fields["artwork_no"] == "88116"
        assert ticket.fields["customer_name"] == "Smayan Healthcare"
        assert ticket.fields["substrate"] == "CHROMO 75 - STANDARD"
        assert ticket.fields["new_die"] == "497"
        assert ticket.fields["dated"] == "07-Aug-2026"

    def test_job_name_keeps_its_spaces(self, tmp_path, layout):
        pdf = _write_proof(tmp_path / f"proof_{layout}.pdf", layout)
        assert parse_proof(pdf).fields["job_name"] == "88116 - JARODOL 1ML 30X19"

    def test_die_size_is_typed(self, tmp_path, layout):
        pdf = _write_proof(tmp_path / f"proof_{layout}.pdf", layout)
        ticket = parse_proof(pdf)
        assert (ticket.die_w_mm, ticket.die_h_mm) == (30.0, 19.0)

    def test_repeat_is_numeric(self, tmp_path, layout):
        pdf = _write_proof(tmp_path / f"proof_{layout}.pdf", layout)
        assert parse_proof(pdf).repeat_mm == 297.0

    def test_cmyk_plus_varnish_expands(self, tmp_path, layout):
        """'CMYK+V' is five stations, not one colour."""
        pdf = _write_proof(tmp_path / f"proof_{layout}.pdf", layout)
        assert parse_proof(pdf).colours == ["cyan", "magenta", "yellow", "black", "v"]

    def test_no_of_colors_beats_artwork_no(self, tmp_path, layout):
        """Both labels contain 'No'; the longer phrase must win."""
        pdf = _write_proof(tmp_path / f"proof_{layout}.pdf", layout)
        fields = parse_proof(pdf).fields
        assert fields["no_of_colors"] == "CMYK+V"
        assert fields["artwork_no"] == "88116"


def test_proof_of_a_textless_pdf_is_empty(tmp_path):
    doc = pymupdf.open()
    doc.new_page(width=200, height=100)
    path = tmp_path / "blank.pdf"
    doc.save(path)
    doc.close()

    ticket = parse_proof(path)
    assert ticket.fields == {}
    assert ticket.die_w_mm is None


def _label_pdf(path, w_mm=30.0, h_mm=19.0, text: str | None = None):
    """A page sized exactly like a 30x19 mm label."""
    doc = pymupdf.open()
    page = doc.new_page(width=w_mm / 25.4 * 72, height=h_mm / 25.4 * 72)
    if text:
        page.insert_text((5, 10), text, fontsize=4)
    else:
        # Outlines only, as production artwork actually is.
        page.draw_rect(pymupdf.Rect(2, 2, 20, 12), color=(0, 0, 0), fill=(0, 0, 0))
    doc.save(path)
    doc.close()
    return path


class TestGeometry:
    def test_page_size_in_mm(self, tmp_path):
        geometry = page_geometry(_label_pdf(tmp_path / "label.pdf"))
        assert geometry.width_mm == pytest.approx(30.0, abs=0.05)
        assert geometry.height_mm == pytest.approx(19.0, abs=0.05)
        assert geometry.page_count == 1

    def test_matches_expected_die_either_way_round(self, tmp_path):
        geometry = page_geometry(_label_pdf(tmp_path / "label.pdf"))
        assert geometry.matches(30, 19)
        assert geometry.matches(19, 30), "orientation must not matter"
        assert not geometry.matches(110, 50)

    def test_uniform_bleed_is_a_match_not_a_mismatch(self, tmp_path):
        """The real job-88116 page box is 32x21 mm around a 30x19 die - 1 mm
        bleed per side. That must read as a match with the bleed reported."""
        geometry = page_geometry(_label_pdf(tmp_path / "bleed.pdf", w_mm=32.0, h_mm=21.0))
        assert geometry.matches(30, 19)
        assert geometry.bleed_mm(30, 19) == pytest.approx(1.0, abs=0.05)

    def test_exact_die_reports_zero_bleed(self, tmp_path):
        geometry = page_geometry(_label_pdf(tmp_path / "label.pdf"))
        assert geometry.bleed_mm(30, 19) == 0.0

    def test_nonuniform_excess_is_a_real_mismatch(self, tmp_path):
        """4 mm extra on one axis only is a wrong die, not bleed."""
        geometry = page_geometry(_label_pdf(tmp_path / "off.pdf", w_mm=34.0, h_mm=19.0))
        assert geometry.bleed_mm(30, 19) is None
        assert not geometry.matches(30, 19)

    def test_huge_excess_is_not_bleed(self, tmp_path):
        geometry = page_geometry(_label_pdf(tmp_path / "big.pdf", w_mm=40.0, h_mm=29.0))
        assert geometry.bleed_mm(30, 19) is None

    def test_pt_to_mm(self):
        assert pt_to_mm(72) == pytest.approx(25.4)


class TestRender:
    def test_renders_at_the_requested_scale(self, tmp_path):
        render = render_page(_label_pdf(tmp_path / "label.pdf"), tmp_path / "out.png", dpi=600)

        assert render.dpi == pytest.approx(600)
        assert render.px_per_mm == pytest.approx(600 / 25.4, abs=0.001)
        # 30 mm at 600 dpi is ~709 px.
        assert render.width_px == pytest.approx(709, abs=2)
        assert render.height_px == pytest.approx(449, abs=2)

    def test_measured_pixels_convert_back_to_mm(self, tmp_path):
        """The round trip that every compliance measurement depends on."""
        render = render_page(_label_pdf(tmp_path / "label.pdf"), tmp_path / "out.png", dpi=600)
        assert render.px_to_mm(render.width_px) == pytest.approx(30.0, abs=0.05)

    def test_png_is_written(self, tmp_path):
        out = tmp_path / "renders" / "88116.png"
        render_page(_label_pdf(tmp_path / "label.pdf"), out, dpi=300)
        assert out.exists() and out.stat().st_size > 0

    def test_absurd_dpi_is_clamped(self, tmp_path):
        """A huge dpi on a big page must not try to allocate a gigapixel."""
        geometry = PageGeometry(1, 500, 700, 1417, 1984)
        assert effective_dpi(geometry, 9600) < 9600
        assert effective_dpi(geometry, 72) == 72


class TestTextDetection:
    def test_outlined_artwork_reports_no_text(self, tmp_path):
        assert has_extractable_text(_label_pdf(tmp_path / "curves.pdf")) is False

    def test_live_text_is_detected(self, tmp_path):
        pdf = _label_pdf(tmp_path / "live.pdf", text="Paracetamol")
        assert has_extractable_text(pdf) is True
