"""Tests for corpus discovery.

Every folder name below is a real one taken from the live corpus, so the parser
is measured against the filing conventions it will actually meet rather than
tidy invented examples.
"""

from __future__ import annotations

from pathlib import Path

from labelkb.corpus.walk import (
    build_manifest,
    is_gang_file,
    iter_job_folders,
    load_manifest,
    parse_job_folder,
)


class TestParseJobFolder:
    def test_plain_injection(self):
        p = parse_job_folder("88116 - JARODOL 1ML 30X19")
        assert (p.job_no, p.brand, p.pack_size) == ("88116", "JARODOL", "1ML")
        assert (p.die_w_mm, p.die_h_mm) == (30.0, 19.0)

    def test_dosage_form_word_ends_the_brand(self):
        p = parse_job_folder("87961 - ZEETAC SYRUP 100ML 110X50")
        assert p.brand == "ZEETAC"
        assert p.dosage_form == "syrup"
        assert p.pack_size == "100ML"
        assert (p.die_w_mm, p.die_h_mm) == (110.0, 50.0)

    def test_abbreviated_form_and_hyphenated_brand(self):
        p = parse_job_folder("87850 - NS-VIT SYP 200ML 110X50")
        assert p.brand == "NS-VIT"
        assert p.dosage_form == "syrup"

    def test_repeat_job_is_flagged(self):
        p = parse_job_folder("87979 - DIFELARE-AQ INJ 1ML REPEAT 30X19")
        assert p.brand == "DIFELARE-AQ"
        assert p.dosage_form == "injection"
        assert p.pack_size == "1ML"
        assert p.is_repeat is True

    def test_mixed_case_brand_survives(self):
        p = parse_job_folder("85069 - Methylcopexa-plus 2ml 34X25")
        assert p.brand == "Methylcopexa-plus"
        assert p.pack_size == "2ml"

    def test_no_pack_size(self):
        p = parse_job_folder("87989 - NUROMITE-PLUS 34X25")
        assert p.brand == "NUROMITE-PLUS"
        assert p.pack_size is None
        assert (p.die_w_mm, p.die_h_mm) == (34.0, 25.0)

    def test_brand_only(self):
        p = parse_job_folder("79503 - ACIZEX 30X24")
        assert p.brand == "ACIZEX"
        assert (p.die_w_mm, p.die_h_mm) == (30.0, 24.0)

    def test_strength_as_pack_size(self):
        p = parse_job_folder("87941 - WELPEP 40MG 70X27")
        assert p.brand == "WELPEP"
        assert p.pack_size == "40MG"

    def test_form_embedded_in_hyphenated_brand_is_kept(self):
        """COLIDEN-DROP is the brand; the form must not be split off it."""
        p = parse_job_folder("87774 - COLIDEN-DROP 30ML 89X40")
        assert p.brand == "COLIDEN-DROP"
        assert p.pack_size == "30ML"

    def test_non_job_folders_are_rejected(self):
        for name in ["New folder (6)", "Commom Folder", "Smayan Healthcare", "LOGO 26", "2026"]:
            assert parse_job_folder(name) is None, name


class TestGangDetection:
    def test_ganged_layouts_are_recognised(self):
        assert is_gang_file(
            "Mix Job No 87306, 87855  Job Name 87306 - MEFIMED P 60ML 92X45 UNIT - 1-export.pdf"
        )
        assert is_gang_file("… 70174 - Kufeze-LS 100ML REPEAT 110X50 UNIT - 2.pdf")

    def test_single_artwork_is_not_ganged(self):
        assert not is_gang_file("88116 - JARODOL 1ML 30X19.pdf")


def _make_corpus(root: Path) -> Path:
    """A miniature corpus mirroring the real tree, junk included."""
    job = root / "Smayan Healthcare" / "2026" / "AUGUST 2026" / "88116 - JARODOL 1ML 30X19"
    job.mkdir(parents=True)
    (job / "88116 - JARODOL 1ML 30X19.pdf").write_bytes(b"%PDF-1.6 artwork")
    (job / "88116 - JARODOL 1ML 30X19 ARTWORK FOR APPROVAL.pdf").write_bytes(b"%PDF-1.6 proof")
    (job / "Jarodol 10X1ml  inj Label.cdr").write_bytes(b"CDR")
    meta = job / ".metadata"
    meta.mkdir()
    (meta / ".88116 - JARODOL 1ML 30X19.pdf.info").write_bytes(b"\x00esko")

    other = root / "Smayan Healthcare" / "2026" / "AUGUST 2026" / "87961 - ZEETAC SYRUP 100ML 110X50"
    other.mkdir(parents=True)
    (other / "87961 - ZEETAC SYRUP 100ML 110X50.pdf").write_bytes(b"%PDF-1.6")

    # Junk that must not be picked up.
    (root / "New folder (6)").mkdir(parents=True)
    (root / "LOGO 26").mkdir(parents=True)
    (root / "Untitled-2.cdr").write_bytes(b"CDR")
    return root


def test_walk_finds_only_job_folders(tmp_path):
    root = _make_corpus(tmp_path)
    jobs = sorted(iter_job_folders(root), key=lambda j: j.job_no)

    assert [j.job_no for j in jobs] == ["87961", "88116"]
    assert all(j.customer == "Smayan Healthcare" for j in jobs)


def test_walk_classifies_files(tmp_path):
    root = _make_corpus(tmp_path)
    job = next(j for j in iter_job_folders(root) if j.job_no == "88116")

    assert job.artwork_pdf.endswith("88116 - JARODOL 1ML 30X19.pdf")
    assert "APPROVAL" in job.proof_pdf
    assert job.artwork_pdf != job.proof_pdf
    assert job.source_files and job.source_files[0].endswith(".cdr")
    assert job.info_sidecar.endswith(".pdf.info")
    assert job.artwork_size == len(b"%PDF-1.6 artwork")


def test_metadata_dir_is_not_walked_as_a_job(tmp_path):
    root = _make_corpus(tmp_path)
    assert all(".metadata" not in j.path for j in iter_job_folders(root))


def test_manifest_round_trips(tmp_path):
    root = _make_corpus(tmp_path)
    out = tmp_path / "kb" / "jobs.jsonl"

    assert build_manifest(root, out) == 2
    rows = load_manifest(out)
    assert {r["job_no"] for r in rows} == {"87961", "88116"}
    assert rows[0]["parsed"]["die_w_mm"] is not None
