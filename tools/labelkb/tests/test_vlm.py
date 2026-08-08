"""Tests for role assignment.

The VLM path is tested through an injected transport - no model server needed.
The rule engine is tested on the phrasing that actually appears on Indian
pharma labels, spaces collapsed the way RapidOCR collapses them.
"""

from __future__ import annotations

import json

import pytest

from labelkb.config import Config
from labelkb.vision.ocr import Line, OcrResult, TextBox
from labelkb.vision.vlm import (
    Role,
    assign_by_rules,
    assign_by_vlm,
    assign_roles,
    build_line_listing,
)


def _line(text: str, height: float = 20, y: float = 0) -> Line:
    return Line(boxes=[TextBox(text=text, confidence=0.95, x0=0, y0=y, x1=200, y1=y + height)])


LABEL_LINES = [
    _line("JARODOL", height=60, y=0),                                   # 0 brand
    _line("Diclofenac Sodium Injection IP", height=22, y=70),           # 1 generic
    _line("Each ml contains: Diclofenac Sodium IP 25mg", height=14, y=100),  # 2 composition
    _line("SCHEDULE H PRESCRIPTION DRUG-CAUTION", height=12, y=120),    # 3 warning
    _line("Not to be sold by retail without the prescription", height=12, y=135),  # 4 warning
    _line("Mfg.Lic.No. MNB/25/417", height=12, y=155),                  # 5 licence
    _line("M.R.P. Rs. 18.50 incl. of all taxes", height=12, y=170),     # 6 mrp
    _line("Store in a cool dry place protected from light", height=12, y=185),  # 7 storage
    _line("Batch No. B.No. Mfg.Dt. Exp.Dt.", height=12, y=200),         # 8 batch block
    _line("Marketed by Smayan Healthcare", height=12, y=215),           # 9 marketer
]


@pytest.fixture(scope="module")
def roles():
    assignments = assign_by_rules(LABEL_LINES)
    return {a.line_index: a.role for a in assignments}


class TestRuleEngine:
    @pytest.mark.parametrize(
        "index,expected",
        [
            (0, Role.BRAND),
            (1, Role.GENERIC_NAME),
            (2, Role.COMPOSITION),
            (3, Role.STATUTORY_WARNING),
            (4, Role.STATUTORY_WARNING),
            (5, Role.LICENCE_NO),
            (6, Role.MRP),
            (7, Role.STORAGE),
            (8, Role.BATCH_MFG_EXP),
            (9, Role.MARKETER),
        ],
    )
    def test_each_role(self, roles, index, expected):
        assert roles[index] == expected

    def test_collapsed_spaces_still_match(self):
        """RapidOCR often removes spaces; the rules must survive that."""
        lines = [_line("ScheduleH PrescriptionDrug"), _line("Mfg.Lic.No.MNB/25/417")]
        roles = {a.line_index: a.role for a in assign_by_rules(lines)}
        # Collapsed 'ScheduleH' still contains the schedule pattern.
        assert roles[0] == Role.STATUTORY_WARNING
        assert roles[1] == Role.LICENCE_NO

    def test_no_prominent_line_means_no_brand(self):
        """All-small lines: nothing should be promoted to brand."""
        lines = [
            _line("Store in a cool dry place", height=12),
            _line("random note", height=10),
        ]
        assignments = assign_by_rules(lines)
        assert Role.BRAND not in {a.role for a in assignments}

    def test_every_line_gets_exactly_one_role(self):
        assignments = assign_by_rules(LABEL_LINES)
        assert sorted(a.line_index for a in assignments) == list(range(len(LABEL_LINES)))


class TestVlmPath:
    def _result(self):
        return OcrResult(boxes=[], lines=LABEL_LINES[:3], px_per_mm=11.811)

    def _image(self, tmp_path):
        path = tmp_path / "label.png"
        path.write_bytes(b"\x89PNG\r\n\x1a\nfake")
        return path

    def _good_reply(self):
        return json.dumps(
            {
                "assignments": [
                    {"line": 0, "role": "brand", "confidence": 0.98},
                    {"line": 1, "role": "generic_name", "confidence": 0.95},
                    {"line": 2, "role": "composition", "confidence": 0.9},
                ]
            }
        )

    def test_valid_reply_is_parsed(self, tmp_path):
        assignments = assign_by_vlm(
            self._result(), self._image(tmp_path), "test-model", lambda payload: self._good_reply()
        )
        assert [a.role for a in assignments] == [
            Role.BRAND,
            Role.GENERIC_NAME,
            Role.COMPOSITION,
        ]
        assert all(a.engine == "vlm" for a in assignments)

    def test_bad_json_retries_then_succeeds(self, tmp_path):
        replies = iter(["not json at all", self._good_reply()])
        calls = []

        def transport(payload):
            calls.append(payload)
            return next(replies)

        assignments = assign_by_vlm(self._result(), self._image(tmp_path), "m", transport)
        assert len(assignments) == 3
        assert len(calls) == 2
        # The retry carries the error back to the model.
        assert "Invalid" in calls[1]["messages"][-1]["content"]

    def test_persistent_garbage_raises(self, tmp_path):
        with pytest.raises(RuntimeError, match="unusable"):
            assign_by_vlm(self._result(), self._image(tmp_path), "m", lambda p: "still not json")

    def test_missing_line_is_rejected(self, tmp_path):
        reply = json.dumps({"assignments": [{"line": 0, "role": "brand"}]})
        with pytest.raises(RuntimeError):
            assign_by_vlm(self._result(), self._image(tmp_path), "m", lambda p: reply)

    def test_unknown_role_coerces_to_other(self, tmp_path):
        reply = json.dumps(
            {
                "assignments": [
                    {"line": 0, "role": "hero_text", "confidence": 0.9},
                    {"line": 1, "role": "generic_name", "confidence": 0.9},
                    {"line": 2, "role": "composition", "confidence": 0.9},
                ]
            }
        )
        assignments = assign_by_vlm(self._result(), self._image(tmp_path), "m", lambda p: reply)
        assert assignments[0].role == Role.OTHER
        assert assignments[0].confidence == pytest.approx(0.1)

    def test_line_listing_carries_mm(self):
        listing = json.loads(build_line_listing(self._result()))
        assert listing[0]["text"] == "JARODOL"
        assert listing[0]["height_mm"] == pytest.approx(60 / 11.811, abs=0.02)


class TestOrchestration:
    def test_cpu_profile_uses_rules(self, tmp_path):
        config = Config(engine_profile="cpu-ocr-only")
        result = OcrResult(boxes=[], lines=LABEL_LINES, px_per_mm=11.811)
        outcome = assign_roles(result, tmp_path / "x.png", config=config)
        assert outcome.engine == "rules"

    def test_gpu_profile_uses_vlm_and_flags_disagreements(self, tmp_path):
        config = Config(engine_profile="gpu-full", vlm_model="test-model")
        image = tmp_path / "label.png"
        image.write_bytes(b"png")
        result = OcrResult(boxes=[], lines=LABEL_LINES[:3], px_per_mm=11.811)

        # VLM calls line 1 a brand; rules say generic_name -> disagreement.
        reply = json.dumps(
            {
                "assignments": [
                    {"line": 0, "role": "brand", "confidence": 0.9},
                    {"line": 1, "role": "brand", "confidence": 0.6},
                    {"line": 2, "role": "composition", "confidence": 0.9},
                ]
            }
        )
        outcome = assign_roles(result, image, config=config, transport=lambda p: reply)
        assert outcome.engine == "vlm"
        assert outcome.disagreements == [1]
        assert outcome.role_of(0) == Role.BRAND

    def test_vlm_failure_degrades_to_rules(self, tmp_path):
        config = Config(engine_profile="gpu-full", vlm_model="test-model")
        image = tmp_path / "label.png"
        image.write_bytes(b"png")
        result = OcrResult(boxes=[], lines=LABEL_LINES[:3], px_per_mm=11.811)

        outcome = assign_roles(
            result, image, config=config, transport=lambda p: "garbage forever"
        )
        assert outcome.engine == "rules"
        assert outcome.role_of(0) == Role.BRAND
