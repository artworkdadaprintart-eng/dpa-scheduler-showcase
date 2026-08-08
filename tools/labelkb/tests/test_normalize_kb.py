"""Tests for name normalisation and KB aggregation."""

from __future__ import annotations

import json

import pytest

from labelkb.config import Config
from labelkb.extract import Compliance, LabelRecord, PrintSpec, Prominence, Zone
from labelkb.kb import build_kb, load_records, lookup_generic
from labelkb.normalize import generic_key, parse_composition, parse_query


class TestComposition:
    def test_simple_salt_and_strength(self):
        parsed = parse_composition("Each ml contains: Diclofenac Sodium IP 25mg")
        assert len(parsed) == 1
        ing = parsed[0]
        assert ing.base == "diclofenac"
        assert ing.salt == "sodium"
        assert (ing.strength_value, ing.strength_unit) == (25.0, "mg")

    def test_salt_in_parenthetical(self):
        parsed = parse_composition("Pantoprazole (as Sodium Sesquihydrate) 40 mg")
        assert parsed[0].base == "pantoprazole"
        assert parsed[0].salt == "sodium"
        assert parsed[0].strength_value == 40.0

    def test_combination_is_split(self):
        parsed = parse_composition(
            "Amoxycillin Trihydrate IP 500mg & Clavulanic Acid IP 125mg"
        )
        assert [i.base for i in parsed] == ["amoxycillin", "clavulanic acid"]

    def test_micrograms_normalise(self):
        parsed = parse_composition("Methylcobalamin 1500 mcg")
        assert parsed[0].strength_unit == "mcg"

    def test_ocr_collapsed_spaces_still_parse(self):
        """RapidOCR drops inter-word spaces; case boundaries restore them.
        This exact string came out of the live pipeline demo and previously
        produced the garbage key 'eachmlcontains diclofenacsodiumip'."""
        parsed = parse_composition("Eachmlcontains:DiclofenacSodiumIP25mg")
        assert len(parsed) == 1
        assert parsed[0].base == "diclofenac"
        assert parsed[0].salt == "sodium"
        assert (parsed[0].strength_value, parsed[0].strength_unit) == (25.0, "mg")

    def test_collapsed_and_spaced_yield_the_same_key(self):
        a = parse_composition("Eachmlcontains:DiclofenacSodiumIP25mg")
        b = parse_composition("Each ml contains: Diclofenac Sodium IP 25mg")
        assert generic_key(a) == generic_key(b) == "diclofenac"


class TestGenericKey:
    def test_salt_forms_collapse_to_one_key(self):
        a = parse_composition("Pantoprazole Sodium IP 40mg")
        b = parse_composition("PANTOPRAZOLE (AS SODIUM SESQUIHYDRATE) 40 MG")
        assert generic_key(a) == generic_key(b) == "pantoprazole"

    def test_combination_key_is_order_independent(self):
        a = parse_composition("Clavulanic Acid 125mg + Amoxycillin 500mg")
        b = parse_composition("Amoxycillin 500mg & Clavulanic Acid 125mg")
        assert generic_key(a) == generic_key(b) == "amoxycillin+clavulanic acid"


class TestQuery:
    def test_plain_query(self):
        q = parse_query("pantoprazole 40mg injection")
        assert q.key == "pantoprazole"
        assert q.strength_value == 40.0
        assert q.dosage_form == "injection"

    def test_combination_query(self):
        q = parse_query("amoxicillin + clavulanic acid 625mg tablet")
        assert q.key == "amoxicillin+clavulanic acid"
        assert q.dosage_form == "tablet"


def _record(job_no, base_line, brand, form="injection", die=(30.0, 19.0), ratio=0.4,
            red=True, warnings=None):
    return LabelRecord(
        job_no=job_no,
        customer="Smayan Healthcare",
        brand=brand,
        generic_name=base_line,
        composition_raw=[f"Each ml contains: {base_line} 25mg"],
        dosage_form=form,
        regime="allopathic",
        compliance=Compliance(
            rx_status="rx",
            statutory_warnings=warnings
            or ["SCHEDULE H PRESCRIPTION DRUG - CAUTION",
                "Not to be sold by retail without the prescription"],
            red_band=red,
            licence_no="Mfg. Lic. No. MNB/25/417",
            mrp_text="M.R.P. Rs. 18.50",
            storage="Store in a cool dry place",
            batch_block="Batch No. Mfg. Dt. Exp. Dt.",
            prominence=Prominence(brand_mm=4.0, generic_mm=4.0 * ratio, ratio=ratio),
        ),
        print_spec=PrintSpec(
            die_w_mm=die[0], die_h_mm=die[1],
            inks=["cyan", "magenta", "yellow", "black"],
            substrate="CHROMO 75 - STANDARD",
        ),
        zones=[
            Zone(role="brand", text=brand, bbox_norm=[0.1, 0.05, 0.6, 0.25],
                 height_mm=4.0, confidence=0.9),
            Zone(role="generic_name", text=base_line, bbox_norm=[0.1, 0.3, 0.7, 0.4],
                 height_mm=4.0 * ratio, confidence=0.9),
            Zone(role="statutory_warning", text="SCHEDULE H ...",
                 bbox_norm=[0.1, 0.6, 0.9, 0.7], height_mm=1.2, confidence=0.8),
        ],
    )


@pytest.fixture()
def kb(tmp_path):
    config = Config(kb_dir=str(tmp_path / "kb"))
    config.records_dir.mkdir(parents=True)
    records = [
        _record("88116", "Diclofenac Sodium Injection IP", "JARODOL"),
        _record("87901", "Diclofenac Sodium Injection IP", "DIFELARE-AQ", ratio=0.45),
        _record("87850", "Pantoprazole Sodium Injection IP", "WELPEP", die=(34.0, 25.0)),
    ]
    for record in records:
        (config.records_dir / f"{record.job_no}.json").write_text(
            record.model_dump_json()
        )
    return config


class TestBuildKb:
    def test_counts(self, kb):
        summary = build_kb(kb)
        assert summary == {
            "records": 3,
            "generics": 2,
            "archetypes": 2,
            "regimes": ["allopathic"],
        }

    def test_salt_variants_group_under_one_generic(self, kb):
        build_kb(kb)
        generics = json.loads((kb.kb / "generics.json").read_text())
        assert set(generics) == {"diclofenac", "pantoprazole"}
        assert generics["diclofenac"]["n"] == 2
        brands = {p["brand"] for p in generics["diclofenac"]["products"]}
        assert brands == {"JARODOL", "DIFELARE-AQ"}

    def test_archetype_keyed_by_form_and_die(self, kb):
        build_kb(kb)
        archetypes = json.loads((kb.kb / "archetypes.json").read_text())
        assert set(archetypes) == {"injection|30x19", "injection|34x25"}
        arch = archetypes["injection|30x19"]
        assert arch["n"] == 2
        assert arch["zones"]["brand"]["support"] == 2
        assert arch["prominence_ratio_median"] == pytest.approx(0.425)
        assert arch["common_substrate"] == "CHROMO 75 - STANDARD"

    def test_rules_evidence_counts_with_examples(self, kb):
        build_kb(kb)
        evidence = json.loads((kb.kb / "rules-evidence.json").read_text())
        rules = {r["statement"]: r for r in evidence["allopathic"]}
        red = rules["Red band present on Rx labels"]
        assert (red["support"], red["total"]) == (3, 3)
        assert red["confidence"] == 1.0
        assert "88116" in red["example_jobs"]

    def test_corrections_overlay_records(self, kb):
        (kb.kb / "corrections.jsonl").write_text(
            json.dumps({"job_no": "88116", "field": "brand", "value": "JARODOL-XR"}) + "\n"
        )
        records = {r.job_no: r for r in load_records(kb)}
        assert records["88116"].brand == "JARODOL-XR"
        # The extraction file itself is untouched.
        raw = json.loads((kb.records_dir / "88116.json").read_text())
        assert raw["brand"] == "JARODOL"


class TestLookup:
    def test_exact_hit(self, kb):
        build_kb(kb)
        result = lookup_generic("diclofenac 25mg injection", kb)
        assert result["status"] == "exact"
        assert result["profile"]["n"] == 2

    def test_salt_form_in_query_still_hits(self, kb):
        build_kb(kb)
        result = lookup_generic("diclofenac sodium injection", kb)
        assert result["status"] == "exact"

    def test_unseen_combination_finds_partial_overlap(self, kb):
        build_kb(kb)
        result = lookup_generic("diclofenac + paracetamol tablet", kb)
        assert result["status"] == "partial"
        assert result["related"][0]["overlap"] == ["diclofenac"]

    def test_totally_unseen_returns_same_form_neighbours(self, kb):
        build_kb(kb)
        result = lookup_generic("cefixime 200mg injection", kb)
        assert result["status"] == "none"
        assert result["same_form"], "should offer same-dosage-form precedents"
