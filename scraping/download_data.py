"""
scraping/download_data.py
=========================
Download incrementale dati TML (TennisMyLife) via API.
Produce scraping/master_dataset.csv: dataset unificato 1991–oggi.

USO:
    python scraping/download_data.py          # download + build
    python scraping/download_data.py --build  # solo build (no download)
"""
import json
import os
import sys
import time
import argparse
import datetime
from pathlib import Path

import requests
import pandas as pd

API_URL  = "https://stats.tennismylife.org/api/data-files"
DATA_URL = "https://stats.tennismylife.org/data/{filename}"

SCRAPING_DIR   = Path(__file__).parent
MANIFEST_FILE  = SCRAPING_DIR / "tml_manifest.json"
OUTPUT_FILE    = SCRAPING_DIR / "master_dataset.csv"

FIRST_YEAR_WITH_STATS = 1991
CURRENT_YEAR = datetime.datetime.now().year


# ── Manifest ──────────────────────────────────────────────────────────────────

def load_manifest() -> dict:
    if MANIFEST_FILE.exists():
        return json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    return {}


def save_manifest(manifest: dict) -> None:
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


# ── Classificazione file ──────────────────────────────────────────────────────

def is_atp_main(filename: str) -> bool:
    """True se è un file ATP tour principale (no challenger, no special)."""
    name = filename.lower()
    return (
        name[:1].isdigit()
        and "_challenger" not in name
        and "amateur" not in name
        and "ongoing" not in name
        and "database" not in name
    )


def year_from_filename(filename: str) -> int:
    try:
        return int(filename.split(".")[0].split("_")[0])
    except (ValueError, IndexError):
        return 0


# ── Download ──────────────────────────────────────────────────────────────────

def needs_download(filename: str, remote_mtime: str, manifest: dict) -> bool:
    local_path = SCRAPING_DIR / filename
    if not local_path.exists():
        return True
    # Anno corrente e precedente → aggiorna sempre
    year = year_from_filename(filename)
    if year >= CURRENT_YEAR - 1:
        return True
    return manifest.get(filename) != remote_mtime


def get_remote_files() -> dict:
    resp = requests.get(API_URL, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    files = data.get("files")
    if not isinstance(files, list):
        raise ValueError(f"API TML risposta inattesa: {data}")
    return {f["name"]: f for f in files}


def download_file(filename: str) -> None:
    url = DATA_URL.format(filename=filename)
    resp = requests.get(url, timeout=90)
    resp.raise_for_status()
    dest = SCRAPING_DIR / filename
    dest.write_bytes(resp.content)
    print(f"  OK {filename} ({len(resp.content) // 1024} KB)")


def download_all(min_year: int = FIRST_YEAR_WITH_STATS) -> dict:
    print("Contatto API TML...")
    manifest = load_manifest()
    remote   = get_remote_files()

    to_fetch = [
        (name, info["mtime"])
        for name, info in remote.items()
        if is_atp_main(name)
        and year_from_filename(name) >= min_year
        and needs_download(name, info["mtime"], manifest)
    ]

    if not to_fetch:
        print("  Tutti i file gia aggiornati.")
    else:
        print(f"  -> {len(to_fetch)} file da scaricare/aggiornare...")
        for filename, mtime in sorted(to_fetch):
            try:
                download_file(filename)
                manifest[filename] = mtime
            except Exception as e:
                print(f"  Errore {filename}: {e}")
            time.sleep(0.3)

    save_manifest(manifest)
    return manifest


# ── Build master dataset ──────────────────────────────────────────────────────

def build_master_dataset(min_year: int = FIRST_YEAR_WITH_STATS) -> pd.DataFrame:
    files = sorted(
        [
            SCRAPING_DIR / f
            for f in os.listdir(SCRAPING_DIR)
            if f.endswith(".csv")
            and is_atp_main(f)
            and year_from_filename(f) >= min_year
        ],
        key=lambda p: year_from_filename(p.name),
    )

    if not files:
        raise FileNotFoundError(
            "Nessun CSV trovato in scraping/. Esegui download_all() prima."
        )

    dfs = []
    for path in files:
        try:
            df = pd.read_csv(path, low_memory=False)
            if "indoor" not in df.columns:
                df["indoor"] = "O"
            dfs.append(df)
        except Exception as e:
            print(f"  Errore leggendo {path.name}: {e}")

    if not dfs:
        raise FileNotFoundError(
            "Nessun CSV leggibile trovato in scraping/. Controlla i file scaricati."
        )

    master = pd.concat(dfs, ignore_index=True)
    master["tourney_date"] = pd.to_numeric(master["tourney_date"], errors="coerce")
    master = master.sort_values(["tourney_date", "match_num"]).reset_index(drop=True)
    master.to_csv(OUTPUT_FILE, index=False)
    print(
        f"  master_dataset.csv: {len(master):,} righe | "
        f"anni {min_year}-{CURRENT_YEAR}"
    )
    return master


# ── Entry point ───────────────────────────────────────────────────────────────

def run(min_year: int = FIRST_YEAR_WITH_STATS, build_only: bool = False) -> pd.DataFrame:
    if not build_only:
        print("\nDownload TML data...")
        download_all(min_year)
    print("\nBuilding master_dataset.csv...")
    return build_master_dataset(min_year)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true", help="Solo build, no download")
    parser.add_argument("--min-year", type=int, default=FIRST_YEAR_WITH_STATS)
    args = parser.parse_args()
    run(min_year=args.min_year, build_only=args.build)
