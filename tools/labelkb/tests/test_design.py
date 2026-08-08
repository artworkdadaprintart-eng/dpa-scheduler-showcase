"""Tests for CREATE: the design generator, brief and exports.

The KB fixture mirrors test_normalize_kb: two approved diclofenac injections
at 30x19 and one pantoprazole at 34x25 — enough for exact, partial and unseen
paths to all behave differently, which is the point.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET

import pymupdf
import pytest

from labelkb.brief import build_brief
from labelkb.config import Config
from labelkb.design import generate_design, to_pdf, to_svg
from labelkb.export import export_design
from labelkb.kb import build_kb
from tests.test_normalize_kb import _record

WARNING_1 = "SCHEDULE H PRESCRIPTION DRUG - CAUTION"


@pytest.fixture(scope="module")
def kb(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("designkb")
    config = Config(kb_dir=str(tmp / "kb"))
    config.records_dir.mkdir(parents=True)
    records = [
        _record("88116", "Diclofenac Sodium Injection IP", "JARODOL",
                warnings=[WARNING_1, "Not to be sold by retail without the prescription"]),
        _record("87901", "Diclofenac Sodium Injection IP", "DIFELARE-AQ", ratio=0.45,
                warnings=[WARNING_1, "Not to be sold by retail without the prescription"]),
        _record("87850", "Pantoprazole Sodium Injection IP", "WELPEP", die=(34.0, 25.0)),
    ]
    for record in records:
        (config.records_dir / f"{record.job_no}.json").write_text(record.model_dump_json())
    build_kb(config)
    return config


@pytest.fixture(scope="module")
def design(kb):
    return generate_design("diclofenac 25mg injection", config=kb)


@pytest.fixture(scope="module")
def svg(kb):
    return to_svg(generate_design("diclofenac 25mg injection", config=kb))


class TestExactPrecedent:
    def test_die_comes_from_precedent(self, design):
        assert (design.die_w_mm, design.die_h_mm) == (30.0, 19.0)
        assert design.status == "exact"
        assert set(design.precedent_jobs) == {"88116", "87901"}

    def test_archetype_used(self, design):
        assert design.archetype_key == "injection|30x19"
        assert design.archetype_n == 2

    def test_statutory_wording_is_verbatim_precedent(self, design):
        warning = next(e for e in design.elements if e.role == "statutory_warning")
        assert warning.kind == "text"
        assert warning.text == WARNING_1

    def test_red_band_present_because_precedent_has_it(self, design):
        band = next(e for e in design.elements if e.role == "red_band")
        assert band.bbox_mm[2] == design.die_w_mm
        assert "2/2" in band.source or "precedent" in band.source

    def test_generic_sized_by_learned_ratio(self, design):
        brand = next(e for e in design.elements if e.role == "brand")
        generic = next(e for e in design.elements if e.role == "generic_name")
        ratio = generic.font_mm / brand.font_mm
        assert ratio == pytest.approx(0.425, abs=0.01)

    def test_no_schedule_confirm_needed(self, design):
        assert not any("Schedule status" in item for item in design.confirm_items)

    def test_all_mandatory_roles_present(self, design):
        roles = {e.role for e in design.elements}
        assert {
            "brand", "generic_name", "composition", "statutory_warning",
            "storage", "licence_no", "mrp", "batch_mfg_exp", "marketer", "barcode",
        } <= roles

    def test_every_element_cites_a_source(self, design):
        for element in design.elements:
            assert element.source, f"{element.role} has no source"


class TestUnseenGeneric:
    def test_flags_confirm_never_guesses(self, kb):
        design = generate_design("cefixime 200mg injection", config=kb)
        assert design.status == "none"
        assert any("CONFIRM" in item for item in design.confirm_items)
        assert any("Schedule status" in item for item in design.confirm_items)

    def test_still_produces_a_complete_draft(self, kb):
        design = generate_design("cefixime 200mg injection", die="70x30", config=kb)
        assert (design.die_w_mm, design.die_h_mm) == (70.0, 30.0)
        assert len(design.elements) >= 10


class TestSvg:
    def test_true_mm_scale(self, svg):
        root = ET.fromstring(svg)
        assert root.get("width") == "30.0mm"
        assert root.get("height") == "19.0mm"
        assert root.get("viewBox") == "0 0 30.0 19.0"

    def test_carries_the_disclaimer(self, svg):
        assert "DESIGN DRAFT" in svg

    def test_text_elements_present(self, svg):
        root = ET.fromstring(svg)
        ns = "{http://www.w3.org/2000/svg}"
        texts = [t.text for t in root.iter(f"{ns}text")]
        assert any(t and "diclofenac" in t.lower() for t in texts)
        assert any(t == "BRAND NAME" for t in texts)


class TestPdf:
    def test_pdf_page_is_die_sized(self, kb, tmp_path):
        design = generate_design("diclofenac 25mg injection", config=kb)
        out = tmp_path / "draft.pdf"
        to_pdf(design, out)
        with pymupdf.open(out) as doc:
            rect = doc[0].rect
            assert rect.width / 72 * 25.4 == pytest.approx(30.0, abs=0.05)
            assert rect.height / 72 * 25.4 == pytest.approx(19.0, abs=0.05)


class TestBriefAndExport:
    def test_brief_cites_precedent_and_flags_verify(self, kb):
        design = generate_design("diclofenac 25mg injection", config=kb)
        brief = build_brief(design, kb)
        assert "88116" in brief
        assert "VERIFY" in brief
        assert "2/2" in brief  # evidence counts, e.g. red band on 2/2 rx labels
        assert design.disclaimer in brief

    def test_export_writes_the_full_set(self, kb, tmp_path):
        design = generate_design("diclofenac 25mg injection", config=kb)
        written = export_design(design, tmp_path / "out", config=kb)
        for key in ("svg", "pdf", "brief_md", "design_json"):
            assert key in written
            assert (re.sub(r"^.*/", "", written[key])), written[key]
        data = json.loads(open(written["design_json"]).read())
        assert data["query"] == "diclofenac 25mg injection"
