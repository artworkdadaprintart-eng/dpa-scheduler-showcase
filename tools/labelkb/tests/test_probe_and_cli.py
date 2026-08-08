"""Tests for engine selection and the command line."""

from __future__ import annotations

import json

import pymupdf
import pytest

from labelkb.cli import main
from labelkb.probe import (
    PROFILE_CPU,
    PROFILE_GPU_FULL,
    PROFILE_GPU_QUANTISED,
    Probe,
    choose_profile,
    probe,
)


class TestProfileSelection:
    @pytest.mark.parametrize(
        "vram_gb,expected",
        [
            (0.0, PROFILE_CPU),
            (4.0, PROFILE_CPU),
            (8.0, PROFILE_GPU_QUANTISED),
            (12.0, PROFILE_GPU_QUANTISED),
            (16.0, PROFILE_GPU_FULL),
            (24.0, PROFILE_GPU_FULL),
        ],
    )
    def test_thresholds(self, vram_gb, expected):
        assert choose_profile(vram_gb) == expected

    def test_probe_runs_on_this_machine(self):
        result = probe()
        assert result.cpu_count >= 1
        assert result.profile in {PROFILE_CPU, PROFILE_GPU_QUANTISED, PROFILE_GPU_FULL}
        assert result.notes

    def test_a_too_small_gpu_is_called_out(self):
        """Finding a GPU and still falling back deserves an explanation."""
        result = probe()
        if result.gpus and result.profile == PROFILE_CPU:
            assert result.warnings

    def test_summary_mentions_the_profile(self):
        assert probe().profile in probe().summary()

    def test_max_vram_of_no_gpu_is_zero(self):
        assert Probe(platform="x", python="3.11", cpu_count=1, ram_gb=None).max_vram_gb == 0.0


def _label_pdf(path, w_mm=30.0, h_mm=19.0):
    doc = pymupdf.open()
    page = doc.new_page(width=w_mm / 25.4 * 72, height=h_mm / 25.4 * 72)
    page.draw_rect(pymupdf.Rect(2, 2, 20, 12), fill=(0, 0, 0))
    doc.save(path)
    doc.close()
    return path


class TestCli:
    def test_probe_json(self, capsys):
        assert main(["probe", "--json"]) == 0
        data = json.loads(capsys.readouterr().out)
        assert "profile" in data and "cpu_count" in data

    def test_geometry_reports_mm(self, tmp_path, capsys):
        pdf = _label_pdf(tmp_path / "label.pdf")
        assert main(["geometry", str(pdf), "--json"]) == 0
        data = json.loads(capsys.readouterr().out)
        assert data["width_mm"] == pytest.approx(30.0, abs=0.05)
        assert data["has_extractable_text"] is False

    def test_geometry_expect_flags_a_mismatch(self, tmp_path, capsys):
        pdf = _label_pdf(tmp_path / "label.pdf")
        main(["geometry", str(pdf), "--expect", "110x50", "--json"])
        assert json.loads(capsys.readouterr().out)["matches_expected"] is False

    def test_geometry_expect_rejects_nonsense(self, tmp_path, capsys):
        pdf = _label_pdf(tmp_path / "label.pdf")
        assert main(["geometry", str(pdf), "--expect", "big"]) == 2

    def test_render_writes_a_png(self, tmp_path, capsys):
        pdf = _label_pdf(tmp_path / "label.pdf")
        out = tmp_path / "r.png"
        assert main(["render", str(pdf), "--out", str(out), "--dpi", "300", "--json"]) == 0
        assert out.exists()
        assert json.loads(capsys.readouterr().out)["dpi"] == pytest.approx(300)

    def test_walk_dry_run_counts_jobs(self, tmp_path, capsys):
        job = tmp_path / "Smayan Healthcare" / "2026" / "AUGUST 2026" / "88116 - JARODOL 1ML 30X19"
        job.mkdir(parents=True)
        _label_pdf(job / "88116 - JARODOL 1ML 30X19.pdf")
        (tmp_path / "New folder (6)").mkdir()

        assert main(["walk", "--root", str(tmp_path), "--dry-run", "--json"]) == 0
        data = json.loads(capsys.readouterr().out)
        assert data["jobs"] == 1
        assert data["with_artwork_pdf"] == 1
        assert data["manifest"] is None

    def test_walk_rejects_a_missing_root(self, tmp_path, capsys):
        assert main(["walk", "--root", str(tmp_path / "nope"), "--dry-run"]) == 2
        assert "not found" in capsys.readouterr().err

    def test_missing_file_exits_cleanly(self, tmp_path, capsys):
        """A bad path must be a clean error, not a traceback."""
        assert main(["info", str(tmp_path / "absent.info")]) == 2
        assert "error:" in capsys.readouterr().err
