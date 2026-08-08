"""End-to-end extraction test.

This is the pipeline proven in miniature, inside this container: a synthetic
Schedule H injection label is built as a real PDF (brand, generic, composition,
warnings, licence, MRP, storage, batch block, marketer, red band, barcode),
alongside a proof ticket and an Esko sidecar — then extract_label runs the
actual render -> OCR -> CV -> roles -> fuse chain and the record is checked
field by field. What this cannot vouch for is real-corpus typography; that is
what the golden-set pilot on the local machine is for.
"""

from __future__ import annotations

import json

import numpy as np
import pymupdf
import pytest

from labelkb.config import Config
from labelkb.extract import detect_regime, extract_label, run_batch
from tests.test_esko_info import FIXTURE as ESKO_FIXTURE

MM = 72 / 25.4  # points per mm

DIE_W, DIE_H = 89.0, 40.0


def _build_artwork(path):
    """An 89x40 mm Schedule H injection label, drawn as production artwork."""
    doc = pymupdf.open()
    page = doc.new_page(width=DIE_W * MM, height=DIE_H * MM)

    # Red band down the right edge (Schedule H practice).
    page.draw_rect(
        pymupdf.Rect((DIE_W - 6) * MM, 0, DIE_W * MM, DIE_H * MM),
        color=(0.85, 0.1, 0.1),
        fill=(0.85, 0.1, 0.1),
    )

    def text(y_mm, content, size, x_mm=3.0):
        page.insert_text((x_mm * MM, y_mm * MM), content, fontsize=size)

    text(7, "JARODOL", 16)                                             # brand
    text(11.5, "Diclofenac Sodium Injection IP", 6)                    # generic
    text(15, "Each ml contains: Diclofenac Sodium IP 25mg", 4)         # composition
    text(18, "SCHEDULE H PRESCRIPTION DRUG - CAUTION", 3.5)            # warning
    text(20.5, "Not to be sold by retail without the prescription", 3.5)
    text(23.5, "of a Registered Medical Practitioner", 3.5)
    text(26.5, "Store in a cool dry place protected from light", 3.5)  # storage
    text(29.5, "Mfg. Lic. No. MNB/25/417", 3.5)                        # licence
    text(32.5, "M.R.P. Rs. 18.50 incl. of all taxes", 3.5)             # MRP
    text(35.5, "Batch No.   Mfg. Dt.   Exp. Dt.", 3.5)                 # batch block
    text(38.2, "Marketed by Smayan Healthcare", 3.5)                   # marketer

    # A 1D barcode: alternating bars, EAN-ish density.
    rng = np.random.default_rng(11)
    cursor = 55.0
    while cursor < 78.0:
        bar = float(rng.integers(2, 6)) / 10
        page.draw_rect(
            pymupdf.Rect(cursor * MM, 26 * MM, (cursor + bar) * MM, 34 * MM),
            color=(0, 0, 0),
            fill=(0, 0, 0),
        )
        cursor += bar + float(rng.integers(2, 6)) / 10

    doc.save(path)
    doc.close()
    return path


def _build_proof(path):
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    y = 100.0
    for label, value in [
        ("Artwork No:", "88116"),
        ("Customer Name:", "Smayan Healthcare"),
        ("Size:", "89 x 40"),
        ("Repeat:", "297"),
        ("No of Colors:", "CMYK+V"),
        ("Substrate:", "CHROMO 75 - STANDARD"),
    ]:
        page.insert_text((60, y), label, fontsize=9)
        page.insert_text((230, y), value, fontsize=9)
        y += 26
    doc.save(path)
    doc.close()
    return path


@pytest.fixture(scope="module")
def extracted(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("extract")
    artwork = _build_artwork(tmp / "88116 - JARODOL 1ML 89X40.pdf")
    proof = _build_proof(tmp / "88116 ARTWORK FOR APPROVAL.pdf")
    sidecar = tmp / ".88116.pdf.info"
    sidecar.write_bytes(ESKO_FIXTURE)

    job = {
        "job_no": "88116",
        "customer": "Smayan Healthcare",
        "artwork_pdf": str(artwork),
        "proof_pdf": str(proof),
        "info_sidecar": str(sidecar),
        "parsed": {
            "brand": "JARODOL",
            "pack_size": "1ML",
            "dosage_form": None,
            "die_w_mm": DIE_W,
            "die_h_mm": DIE_H,
        },
    }
    config = Config(engine_profile="cpu-ocr-only", kb_dir=str(tmp / "kb"), render_dpi=300)
    record = extract_label(job, config=config)
    return record, job, config, tmp


class TestIdentity:
    def test_brand_and_customer(self, extracted):
        record, *_ = extracted
        assert record.job_no == "88116"
        assert record.brand == "JARODOL"
        assert record.customer == "Smayan Healthcare"

    def test_generic_name_found_by_vision(self, extracted):
        record, *_ = extracted
        assert "diclofenac" in record.generic_name.lower()

    def test_composition_captured(self, extracted):
        record, *_ = extracted
        assert record.composition_raw
        assert "25" in record.composition_raw[0]


class TestCompliance:
    def test_regime_is_allopathic(self, extracted):
        record, *_ = extracted
        assert record.regime == "allopathic"

    def test_rx_status_from_schedule_warning(self, extracted):
        record, *_ = extracted
        assert record.compliance.rx_status == "rx"

    def test_statutory_warnings_verbatim_lines(self, extracted):
        record, *_ = extracted
        joined = " ".join(record.compliance.statutory_warnings).lower().replace(" ", "")
        assert "scheduleh" in joined
        assert "nottobesold" in joined

    def test_red_band_detected_at_right_edge(self, extracted):
        record, *_ = extracted
        assert record.compliance.red_band
        x0 = record.compliance.red_band_bbox_mm[0]
        assert x0 > DIE_W * 0.8

    def test_barcode_detected(self, extracted):
        record, *_ = extracted
        assert record.compliance.has_barcode

    def test_housekeeping_fields(self, extracted):
        record, *_ = extracted
        assert "mnb/25/417" in record.compliance.licence_no.lower().replace(" ", "")
        assert record.compliance.mrp_text
        assert record.compliance.storage
        assert record.compliance.batch_block
        assert record.compliance.marketer

    def test_prominence_measured_not_guessed(self, extracted):
        record, *_ = extracted
        prominence = record.compliance.prominence
        assert prominence.brand_mm and prominence.generic_mm
        assert prominence.brand_mm > prominence.generic_mm
        assert 0.1 < prominence.ratio < 1.0


class TestPrintSpec:
    def test_die_size_from_page_box(self, extracted):
        record, *_ = extracted
        assert record.print_spec.die_w_mm == pytest.approx(DIE_W, abs=0.1)
        assert record.print_spec.die_h_mm == pytest.approx(DIE_H, abs=0.1)

    def test_esko_fields_flow_through(self, extracted):
        record, *_ = extracted
        assert record.print_spec.dieline_ard == "Label 30 x 19.ARD"
        assert record.print_spec.colour_count == 4

    def test_proof_fields_flow_through(self, extracted):
        record, *_ = extracted
        assert record.print_spec.substrate == "CHROMO 75 - STANDARD"
        assert record.print_spec.repeat_mm == 297.0


class TestZonesAndProvenance:
    def test_every_line_becomes_a_zone(self, extracted):
        record, *_ = extracted
        assert len(record.zones) >= 10
        for zone in record.zones:
            assert 0 <= zone.bbox_norm[0] <= 1.05
            assert zone.height_mm > 0

    def test_provenance_records_the_engine(self, extracted):
        record, *_ = extracted
        assert record.provenance.role_engine == "rules"
        assert record.provenance.ocr_mean_confidence > 0.8
        assert record.provenance.render_png.endswith("88116.png")


class TestBatch:
    def test_batch_writes_then_skips(self, extracted):
        _, job, config, _ = extracted
        first = run_batch([job], config=config)
        assert (first["extracted"], first["failed"]) == (1, 0)
        again = run_batch([job], config=config)
        assert again["skipped"] == 1 and again["extracted"] == 0

        record_path = config.records_dir / "88116.json"
        data = json.loads(record_path.read_text())
        assert data["brand"] == "JARODOL"

    def test_broken_job_is_recorded_not_fatal(self, extracted):
        _, _, config, tmp = extracted
        bad = {"job_no": "99999", "artwork_pdf": str(tmp / "missing.pdf"), "parsed": {}}
        result = run_batch([bad], config=config)
        assert result["failed"] == 1
        assert result["failures"][0]["job_no"] == "99999"


class TestRegimeDetection:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("An Ayurvedic Proprietary Medicine", "ayush"),
            ("FSSAI Lic No 10012 Food Supplement", "fssai"),
            ("Diclofenac Sodium Injection IP Mfg Lic", "allopathic"),
            ("Industrial adhesive sticker", "other"),
        ],
    )
    def test_detects(self, text, expected):
        assert detect_regime(text)[0] == expected
