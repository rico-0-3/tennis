"""Unit test per scraping/download_data.py."""
import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent / "scraping"))
import download_data as dd

# ── Test manifest ─────────────────────────────────────────────────────────────

def test_load_manifest_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(dd, "MANIFEST_FILE", tmp_path / "manifest.json")
    assert dd.load_manifest() == {}


def test_save_and_load_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(dd, "MANIFEST_FILE", tmp_path / "manifest.json")
    dd.save_manifest({"2024.csv": "2024-01-01T00:00:00"})
    assert dd.load_manifest() == {"2024.csv": "2024-01-01T00:00:00"}


# ── Test needs_download ───────────────────────────────────────────────────────

def test_needs_download_file_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(dd, "SCRAPING_DIR", tmp_path)
    assert dd.needs_download("2010.csv", "mtime_x", {}) is True


def test_needs_download_mtime_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(dd, "SCRAPING_DIR", tmp_path)
    (tmp_path / "2010.csv").write_text("data")
    manifest = {"2010.csv": "mtime_x"}
    assert dd.needs_download("2010.csv", "mtime_x", manifest) is False


def test_needs_download_mtime_changed(tmp_path, monkeypatch):
    monkeypatch.setattr(dd, "SCRAPING_DIR", tmp_path)
    (tmp_path / "2010.csv").write_text("data")
    manifest = {"2010.csv": "mtime_old"}
    assert dd.needs_download("2010.csv", "mtime_new", manifest) is True


def test_needs_download_current_year_always(tmp_path, monkeypatch):
    """Il file dell'anno corrente viene sempre riscaricato."""
    import datetime
    cur = datetime.datetime.now().year
    monkeypatch.setattr(dd, "SCRAPING_DIR", tmp_path)
    monkeypatch.setattr(dd, "CURRENT_YEAR", cur)
    (tmp_path / f"{cur}.csv").write_text("data")
    manifest = {f"{cur}.csv": "mtime_x"}
    assert dd.needs_download(f"{cur}.csv", "mtime_x", manifest) is True


# ── Test is_atp_main ──────────────────────────────────────────────────────────

def test_is_atp_main_accepts_year_files():
    assert dd.is_atp_main("2023.csv") is True
    assert dd.is_atp_main("1999.csv") is True


def test_is_atp_main_rejects_challenger():
    assert dd.is_atp_main("2023_challenger.csv") is False


def test_is_atp_main_rejects_special():
    assert dd.is_atp_main("ATP_Database.csv") is False
    assert dd.is_atp_main("ongoing_tourneys.csv") is False
    assert dd.is_atp_main("atp_matches_amateur.csv") is False


# ── Test year_from_filename ───────────────────────────────────────────────────

def test_year_from_filename():
    assert dd.year_from_filename("2023.csv") == 2023
    assert dd.year_from_filename("1991.csv") == 1991
    assert dd.year_from_filename("ATP_Database.csv") == 0
