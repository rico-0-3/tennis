# Tennis Predictor v6.0 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ricostruire la pipeline di training e predizione del modello tennis ATP per passare da ~68% a ~71-73% di accuracy, risolvendo tutti i bug identificati (dati sparsi, calibrazione su test set, ELO semplice, nessun modello surface-specific).

**Architecture:** Download incrementale dati TML (1991–oggi) via API → feature engineering con Glicko-2 al posto di ELO + serve/return stats reali → 3 LGB surface-specific + ANN + XGB ensemble con selezione automatica del migliore → calibrazione corretta su validation set.

**Tech Stack:** Python 3.x, PyTorch, LightGBM, XGBoost, Optuna, scikit-learn, requests, pandas, joblib

---

## File Map

| File | Tipo | Responsabilità |
|---|---|---|
| `scraping/download_data.py` | Nuovo | Download incrementale TML via API, manifest, build master_dataset.csv |
| `scraping/tml_manifest.json` | Generato | Traccia mtime file già scaricati |
| `scraping/master_dataset.csv` | Generato | Dataset unificato 1991–oggi |
| `prediccion/glicko2.py` | Nuovo | Implementazione Glicko-2 standalone e testabile |
| `prediccion/train_ann.py` | Modifica major | Feature engineering v6, surface LGB, calibrazione fix, log-loss |
| `prediccion/prediction_engine.py` | Modifica | Feature list v6 (32 feature), routing LGB superficie, carica Glicko-2 |
| `tests/test_download.py` | Nuovo | Unit test download logic e manifest |
| `tests/test_glicko2.py` | Nuovo | Unit test Glicko-2 math |

---

## Task 1: Setup + Download Script

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/test_download.py`
- Create: `scraping/download_data.py`

- [ ] **Step 1.1: Installa pytest se mancante**

```powershell
cd "C:\Users\riccardo\Desktop\tennis"
.\venv\Scripts\python.exe -m pip install pytest requests --quiet
```

Expected: nessun errore

- [ ] **Step 1.2: Crea directory tests**

```powershell
New-Item -ItemType Directory -Force -Path "tests"
New-Item -ItemType File -Force -Path "tests\__init__.py"
```

- [ ] **Step 1.3: Scrivi i test per il download script**

Crea `tests/test_download.py`:

```python
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
```

- [ ] **Step 1.4: Esegui i test per verificare che falliscano (modulo non ancora creato)**

```powershell
cd "C:\Users\riccardo\Desktop\tennis"
.\venv\Scripts\python.exe -m pytest tests/test_download.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'download_data'`

- [ ] **Step 1.5: Implementa `scraping/download_data.py`**

Crea `scraping/download_data.py`:

```python
"""
scraping/download_data.py
=========================
Download incrementale dati TML (TennisMyLife) via API.
Produce scraping/master_dataset.csv: dataset unificato 1991–oggi.

USO:
    cd prediccion/
    python ../scraping/download_data.py          # download + build
    python ../scraping/download_data.py --build  # solo build (no download)
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
        name[0].isdigit()
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
    return {f["name"]: f for f in data["files"]}


def download_file(filename: str) -> None:
    url = DATA_URL.format(filename=filename)
    resp = requests.get(url, timeout=90)
    resp.raise_for_status()
    dest = SCRAPING_DIR / filename
    dest.write_bytes(resp.content)
    print(f"  ✓ {filename} ({len(resp.content) // 1024} KB)")


def download_all(min_year: int = FIRST_YEAR_WITH_STATS) -> dict:
    print("📡 Contatto API TML...")
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
        print("  ✓ Tutti i file già aggiornati.")
    else:
        print(f"  → {len(to_fetch)} file da scaricare/aggiornare...")
        for filename, mtime in sorted(to_fetch):
            try:
                download_file(filename)
                manifest[filename] = mtime
            except Exception as e:
                print(f"  ✗ Errore {filename}: {e}")
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
            # TML >= 2018 aggiunge colonna 'indoor'; normalizza per anni precedenti
            if "indoor" not in df.columns:
                df["indoor"] = "O"
            dfs.append(df)
        except Exception as e:
            print(f"  ⚠ Errore leggendo {path.name}: {e}")

    master = pd.concat(dfs, ignore_index=True)
    master["tourney_date"] = pd.to_numeric(master["tourney_date"], errors="coerce")
    master = master.sort_values(["tourney_date", "match_num"]).reset_index(drop=True)
    master.to_csv(OUTPUT_FILE, index=False)
    print(
        f"  ✓ master_dataset.csv: {len(master):,} righe | "
        f"anni {min_year}–{CURRENT_YEAR}"
    )
    return master


# ── Entry point ───────────────────────────────────────────────────────────────

def run(min_year: int = FIRST_YEAR_WITH_STATS, build_only: bool = False) -> pd.DataFrame:
    if not build_only:
        print("\n📥 Download TML data...")
        download_all(min_year)
    print("\n🔧 Building master_dataset.csv...")
    return build_master_dataset(min_year)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true", help="Solo build, no download")
    parser.add_argument("--min-year", type=int, default=FIRST_YEAR_WITH_STATS)
    args = parser.parse_args()
    run(min_year=args.min_year, build_only=args.build)
```

- [ ] **Step 1.6: Esegui i test — devono passare tutti**

```powershell
cd "C:\Users\riccardo\Desktop\tennis"
.\venv\Scripts\python.exe -m pytest tests/test_download.py -v
```

Expected output:
```
tests/test_download.py::test_load_manifest_missing PASSED
tests/test_download.py::test_save_and_load_manifest PASSED
tests/test_download.py::test_needs_download_file_missing PASSED
tests/test_download.py::test_needs_download_mtime_unchanged PASSED
tests/test_download.py::test_needs_download_mtime_changed PASSED
tests/test_download.py::test_needs_download_current_year_always PASSED
tests/test_download.py::test_is_atp_main_accepts_year_files PASSED
tests/test_download.py::test_is_atp_main_rejects_challenger PASSED
tests/test_download.py::test_is_atp_main_rejects_special PASSED
tests/test_download.py::test_year_from_filename PASSED
10 passed
```

- [ ] **Step 1.7: Esegui il download reale (richiede connessione internet)**

```powershell
cd "C:\Users\riccardo\Desktop\tennis\prediccion"
..\venv\Scripts\python.exe ..\scraping\download_data.py
```

Expected: scarica i CSV annuali TML 1991–2026 in `scraping/` e crea `scraping/master_dataset.csv` con ~450k righe. Può richiedere 5-10 minuti.

- [ ] **Step 1.8: Verifica master_dataset.csv**

```powershell
.\venv\Scripts\python.exe -c "
import pandas as pd
df = pd.read_csv('../scraping/master_dataset.csv', low_memory=False)
print(f'Righe: {len(df):,}')
print(f'Anni: {int(df.tourney_date.min()//10000)} - {int(df.tourney_date.max()//10000)}')
print(f'Colonne: {list(df.columns)}')
print(f'w_svpt non-zero: {(df.w_svpt.fillna(0) > 0).sum():,} / {len(df):,}')
print(df.head(2).to_string())
"
```

Expected: ~450k righe, anni 1991-2026, colonna `indoor` presente.

- [ ] **Step 1.9: Commit**

```powershell
cd "C:\Users\riccardo\Desktop\tennis"
git add scraping/download_data.py tests/__init__.py tests/test_download.py
git commit -m "feat: download incrementale TML con manifest + test"
```

---

## Task 2: Modulo Glicko-2

**Files:**
- Create: `tests/test_glicko2.py`
- Create: `prediccion/glicko2.py`

- [ ] **Step 2.1: Scrivi i test Glicko-2**

Crea `tests/test_glicko2.py`:

```python
"""Unit test per prediccion/glicko2.py."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "prediccion"))
from glicko2 import update, Glicko2Store, GLICKO_MU_0, GLICKO_PHI_0, GLICKO_SIGMA_0


def test_winner_rating_increases():
    """Il vincitore deve aumentare il proprio rating."""
    r0, rd0, sig0 = 1500.0, 200.0, 0.06
    r1, rd1, sig1 = 1500.0, 200.0, 0.06
    (rw, rdw, _), (rl, rdl, _) = update(r0, rd0, sig0, r1, rd1, sig1)
    assert rw > r0, "rating vincitore deve aumentare"
    assert rl < r1, "rating perdente deve diminuire"


def test_symmetric_update():
    """Due giocatori con stesso rating: cambio uguale e opposto."""
    r0, rd0, sig0 = 1500.0, 200.0, 0.06
    (rw, rdw, _), (rl, rdl, _) = update(r0, rd0, sig0, r0, rd0, sig0)
    delta_w = rw - r0
    delta_l = rl - r0
    assert abs(delta_w + delta_l) < 1e-6, "delta simmetrico"
    assert delta_w > 0


def test_strong_vs_weak_small_update():
    """Battere un giocatore molto più debole → piccolo guadagno di rating."""
    strong = (2000.0, 50.0, 0.06)  # forte, RD bassa
    weak   = (1200.0, 50.0, 0.06)
    (rw, _, _), _ = update(*strong, *weak)
    delta_strong = rw - strong[0]
    # Battendo un debole si guadagna poco
    assert delta_strong < 10, f"guadagno atteso < 10, got {delta_strong:.2f}"


def test_rd_decreases_after_match():
    """La RD deve diminuire dopo un match (più certezza)."""
    r0, rd0, sig0 = 1500.0, 350.0, 0.06
    (_, rdw, _), (_, rdl, _) = update(r0, rd0, sig0, r0, rd0, sig0)
    assert rdw < rd0, "RD vincitore deve diminuire"
    assert rdl < rd0, "RD perdente deve diminuire"


def test_rd_capped_at_350():
    """La RD non può superare 350 (valore iniziale massimo)."""
    (rw, rdw, _), (rl, rdl, _) = update(1500, 350, 0.06, 1500, 350, 0.06)
    assert rdw <= 350.0
    assert rdl <= 350.0


def test_store_initial_values():
    store = Glicko2Store()
    r, rd, sig = store.get("Novak Djokovic")
    assert r == GLICKO_MU_0
    assert rd == GLICKO_PHI_0
    assert sig == GLICKO_SIGMA_0


def test_store_update_match():
    store = Glicko2Store()
    store.update_match("Sinner", "Djokovic")
    r_s, _, _ = store.get("Sinner")
    r_d, _, _ = store.get("Djokovic")
    assert r_s > GLICKO_MU_0
    assert r_d < GLICKO_MU_0


def test_store_serialization():
    store = Glicko2Store()
    store.update_match("Alcaraz", "Medvedev")
    d = store.to_dict()
    store2 = Glicko2Store.from_dict(d)
    assert store2.get("Alcaraz") == store.get("Alcaraz")
    assert store2.get("Medvedev") == store.get("Medvedev")


def test_store_independent_surfaces():
    """Store separati per superficie sono indipendenti."""
    hard_store = Glicko2Store()
    clay_store = Glicko2Store()
    hard_store.update_match("Federer", "Nadal")  # Federer vince su Hard
    clay_store.update_match("Nadal", "Federer")  # Nadal vince su Clay
    r_fed_hard, _, _ = hard_store.get("Federer")
    r_fed_clay, _, _ = clay_store.get("Federer")
    assert r_fed_hard > GLICKO_MU_0
    assert r_fed_clay < GLICKO_MU_0
```

- [ ] **Step 2.2: Esegui test per verificare che falliscano**

```powershell
cd "C:\Users\riccardo\Desktop\tennis"
.\venv\Scripts\python.exe -m pytest tests/test_glicko2.py -v 2>&1 | head -5
```

Expected: `ModuleNotFoundError: No module named 'glicko2'`

- [ ] **Step 2.3: Implementa `prediccion/glicko2.py`**

Crea `prediccion/glicko2.py`:

```python
"""
prediccion/glicko2.py
=====================
Implementazione Glicko-2 per tennis (single-match update).

Riferimento: Glickman (2012) "Example of the Glicko-2 system"
http://www.glicko.net/glicko/glicko2.pdf
"""
import math

GLICKO_MU_0    = 1500.0   # rating iniziale
GLICKO_PHI_0   = 350.0    # rating deviation iniziale (massima incertezza)
GLICKO_SIGMA_0 = 0.06     # volatility iniziale
GLICKO_SCALE   = 173.7178 # fattore di scala Glicko-2


def _g(phi: float) -> float:
    return 1.0 / math.sqrt(1.0 + 3.0 * phi**2 / math.pi**2)


def _E(mu: float, mu_j: float, phi_j: float) -> float:
    return 1.0 / (1.0 + math.exp(-_g(phi_j) * (mu - mu_j)))


def _to_scale(r: float, rd: float):
    """Converte da scala Glicko-1 (r, RD) a scala Glicko-2 (mu, phi)."""
    return (r - 1500.0) / GLICKO_SCALE, rd / GLICKO_SCALE


def _from_scale(mu: float, phi: float):
    """Riconverte da scala Glicko-2 a scala Glicko-1."""
    return GLICKO_SCALE * mu + 1500.0, min(350.0, GLICKO_SCALE * phi)


def update(
    winner_r: float, winner_rd: float, winner_sigma: float,
    loser_r: float,  loser_rd: float,  loser_sigma: float,
) -> tuple[tuple, tuple]:
    """
    Aggiorna i rating Glicko-2 per un singolo match.

    Args:
        winner_r, winner_rd, winner_sigma: rating, RD e volatility del vincitore
        loser_r,  loser_rd,  loser_sigma:  rating, RD e volatility del perdente

    Returns:
        ((r_w, rd_w, sigma_w), (r_l, rd_l, sigma_l)) — nuovi valori post-match
    """
    mu_w, phi_w = _to_scale(winner_r, winner_rd)
    mu_l, phi_l = _to_scale(loser_r,  loser_rd)

    # ── Aggiornamento vincitore (outcome = 1.0) ───────────────────────────────
    g_l  = _g(phi_l)
    e_w  = _E(mu_w, mu_l, phi_l)
    v_w  = 1.0 / (g_l**2 * e_w * (1.0 - e_w))
    phi_star_w = math.sqrt(phi_w**2 + winner_sigma**2)
    phi_w_new  = 1.0 / math.sqrt(1.0 / phi_star_w**2 + 1.0 / v_w)
    mu_w_new   = mu_w + phi_w_new**2 * g_l * (1.0 - e_w)
    r_w_new, rd_w_new = _from_scale(mu_w_new, phi_w_new)

    # ── Aggiornamento perdente (outcome = 0.0) ────────────────────────────────
    g_w  = _g(phi_w)
    e_l  = _E(mu_l, mu_w, phi_w)
    v_l  = 1.0 / (g_w**2 * e_l * (1.0 - e_l))
    phi_star_l = math.sqrt(phi_l**2 + loser_sigma**2)
    phi_l_new  = 1.0 / math.sqrt(1.0 / phi_star_l**2 + 1.0 / v_l)
    mu_l_new   = mu_l + phi_l_new**2 * g_w * (0.0 - e_l)
    r_l_new, rd_l_new = _from_scale(mu_l_new, phi_l_new)

    return (r_w_new, rd_w_new, winner_sigma), (r_l_new, rd_l_new, loser_sigma)


class Glicko2Store:
    """
    Gestisce i rating Glicko-2 per un insieme di giocatori.
    Usato separatamente per ogni superficie (Hard/Clay/Grass) e per overall.
    """

    def __init__(self):
        # {player_name: (r, rd, sigma)}
        self._ratings: dict[str, tuple[float, float, float]] = {}

    def get(self, player: str) -> tuple[float, float, float]:
        """Restituisce (r, rd, sigma). Default se giocatore sconosciuto."""
        return self._ratings.get(player, (GLICKO_MU_0, GLICKO_PHI_0, GLICKO_SIGMA_0))

    def update_match(self, winner: str, loser: str) -> None:
        """Aggiorna i rating di vincitore e perdente dopo un match."""
        wr, wrd, wsig = self.get(winner)
        lr, lrd, lsig = self.get(loser)
        (wr_new, wrd_new, wsig_new), (lr_new, lrd_new, lsig_new) = update(
            wr, wrd, wsig, lr, lrd, lsig
        )
        self._ratings[winner] = (wr_new, wrd_new, wsig_new)
        self._ratings[loser]  = (lr_new, lrd_new, lsig_new)

    def to_dict(self) -> dict:
        return dict(self._ratings)

    @classmethod
    def from_dict(cls, d: dict) -> "Glicko2Store":
        store = cls()
        store._ratings = dict(d)
        return store
```

- [ ] **Step 2.4: Esegui i test — devono passare tutti**

```powershell
cd "C:\Users\riccardo\Desktop\tennis"
.\venv\Scripts\python.exe -m pytest tests/test_glicko2.py -v
```

Expected:
```
tests/test_glicko2.py::test_winner_rating_increases PASSED
tests/test_glicko2.py::test_symmetric_update PASSED
tests/test_glicko2.py::test_strong_vs_weak_small_update PASSED
tests/test_glicko2.py::test_rd_decreases_after_match PASSED
tests/test_glicko2.py::test_rd_capped_at_350 PASSED
tests/test_glicko2.py::test_store_initial_values PASSED
tests/test_glicko2.py::test_store_update_match PASSED
tests/test_glicko2.py::test_store_serialization PASSED
tests/test_glicko2.py::test_store_independent_surfaces PASSED
9 passed
```

- [ ] **Step 2.5: Commit**

```powershell
cd "C:\Users\riccardo\Desktop\tennis"
git add prediccion/glicko2.py tests/test_glicko2.py
git commit -m "feat: modulo Glicko-2 standalone con test"
```

---

## Task 3: Feature Engineering (train_ann.py — parte 1)

Sostituisce ELO con Glicko-2, aggiunge `indoor`, fixa serve stats, aggiorna FEATURES a 32.

**Files:**
- Modify: `prediccion/train_ann.py`

- [ ] **Step 3.1: Leggi le prime 100 righe di train_ann.py per orientarti**

```powershell
cd "C:\Users\riccardo\Desktop\tennis\prediccion"
Get-Content train_ann.py | Select-Object -First 100
```

- [ ] **Step 3.2: Aggiorna gli import e rimuovi il court_speed import**

In `prediccion/train_ann.py`, sostituisci il blocco imports da riga 1 fino a `from court_speed_helper import get_court_stats`:

**Trova** (blocco riga ~38-48):
```python
# ── Court Speed helper ────────────────────────────────────────────────────────
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scraping'))
try:
    from court_speed_helper import get_court_stats
    HAS_COURT_SPEED = True
    print("   ✅ court_speed_helper caricato")
except ImportError:
    HAS_COURT_SPEED = False
    print("   ⚠️  court_speed_helper non trovato — court_ace_pct / court_speed = 0")
    def get_court_stats(name, surface='Hard', year=2025):
        return 0.0, 0.0
```

**Sostituisci con:**
```python
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scraping'))
from glicko2 import Glicko2Store
```

- [ ] **Step 3.3: Aggiorna la lista FEATURES (32 feature, indici v6)**

**Trova** il blocco `FEATURES = [` (riga ~568):

```python
FEATURES = [
    'log_rank_ratio',           # 0   ranking compresso
    'log_pts_ratio',            # 1   punti ATP
    'diff_elo',                 # 2   Elo superficie
    'diff_elo_overall',         # 3   Elo overall
    'diff_streak',              # 4   striscia attiva
    'diff_recent_form',         # 5   media ultimi 10
    'surface_enc',              # 6   superficie
    'tourney_level',            # 7   livello torneo
    'round_enc',                # 8   round
    'is_best_of_5',             # 9   Bo3 vs Bo5
    'diff_h2h',                 # 10  H2H globale
    'diff_h2h_surface',         # 11  H2H superficie
    'diff_skill',               # 12  win-rate superficie (running)
    'diff_momentum',            # 13  rolling 10 su superficie
    'diff_fatigue',             # 14  minuti torneo corrente
    'diff_days_since_last',     # 15  riposo
    'diff_weeks_load',          # 16  match ultime 8 settimane  [v5]
    'diff_ace',                 # 17  ace
    'diff_1st_pct',             # 18  1st serve %
    'diff_1st_won',             # 19  1st serve won %
    'diff_2nd_won',             # 20  2nd serve won %
    'diff_bp_saved',            # 21  break point saved %
    'diff_return_pct',          # 22  return %
    'diff_bp_conv',             # 23  break point conv %
    'diff_return_1st',          # 24  return 1st serve %
    'diff_home',                # 25  vantaggio casa
    'diff_opponent_quality',    # 26  qualità avversari ultimi 5
    'diff_upset_tendency',      # 27  tendenza a perdere vs rank peggiori  [v5]
    'diff_late_round_wr',       # 28  win rate QF/SF/F  [v5]
    'level_weight',             # 29  peso torneo
]   # totale: 30 feature
```

**Sostituisci con:**
```python
FEATURES = [
    'log_rank_ratio',           # 0   ranking compresso
    'log_pts_ratio',            # 1   punti ATP
    'diff_glicko',              # 2   Glicko-2 superficie (ex diff_elo)
    'diff_glicko_overall',      # 3   Glicko-2 overall (ex diff_elo_overall)
    'diff_glicko_rd',           # 4   Glicko-2 RD diff — incertezza [NEW]
    'diff_streak',              # 5   striscia attiva
    'diff_recent_form',         # 6   media ultimi 10
    'surface_enc',              # 7   superficie
    'tourney_level',            # 8   livello torneo
    'round_enc',                # 9   round
    'is_best_of_5',             # 10  Bo3 vs Bo5
    'indoor',                   # 11  indoor/outdoor [NEW]
    'diff_h2h',                 # 12  H2H globale
    'diff_h2h_surface',         # 13  H2H superficie
    'diff_skill',               # 14  win-rate superficie (running)
    'diff_momentum',            # 15  rolling 10 su superficie
    'diff_fatigue',             # 16  minuti torneo corrente
    'diff_days_since_last',     # 17  riposo
    'diff_weeks_load',          # 18  match ultime 8 settimane
    'diff_ace',                 # 19  ace
    'diff_1st_pct',             # 20  1st serve %
    'diff_1st_won',             # 21  1st serve won %
    'diff_2nd_won',             # 22  2nd serve won %
    'diff_bp_saved',            # 23  break point saved %
    'diff_return_pct',          # 24  return %
    'diff_bp_conv',             # 25  break point conv %
    'diff_return_1st',          # 26  return 1st serve %
    'diff_home',                # 27  vantaggio casa
    'diff_opponent_quality',    # 28  qualità avversari ultimi 5
    'diff_upset_tendency',      # 29  tendenza upset
    'diff_late_round_wr',       # 30  win rate QF/SF/F
    'level_weight',             # 31  peso torneo
]   # totale: 32 feature — v6.0
```

- [ ] **Step 3.4: Aggiorna INTERACTION_SETS con i nuovi indici v6**

**Trova** il blocco `INTERACTION_SETS = {` e **sostituisci tutto il dizionario**:

```python
# Indici v6: 0=log_rank, 1=log_pts, 2=glicko, 3=glicko_ov, 4=glicko_rd,
# 5=streak, 6=form, 7=surface, 8=level, 9=round, 10=bo5, 11=indoor,
# 12=h2h, 13=h2h_surf, 14=skill, 15=momentum, 16=fatigue, 17=days_since,
# 18=weeks_load, 19=ace, 20=1st_pct, 21=1st_won, 22=2nd_won, 23=bp_saved,
# 24=return_pct, 25=bp_conv, 26=return_1st, 27=home, 28=opp_quality,
# 29=upset_tendency, 30=late_round_wr, 31=level_weight

INTERACTION_SETS = {
    'core': [
        (2, 14),  # glicko × skill
        (0, 15),  # rank × momentum
        (2, 15),  # glicko × momentum
        (14, 15), # skill × momentum
        (0, 1),   # rank × pts
        (2, 12),  # glicko × h2h
        (5, 15),  # streak × momentum
        (14, 16), # skill × fatigue
        (16, 15), # fatigue × momentum
        (1, 14),  # pts × skill
    ],
    'upset': [
        (2, 29),  # glicko × upset_tendency
        (0, 29),  # rank × upset_tendency
        (6, 29),  # form × upset_tendency
        (2, 30),  # glicko × late_round_wr
        (5, 29),  # streak × upset_tendency
        (15, 29), # momentum × upset_tendency
        (18, 29), # weeks_load × upset_tendency
        (2, 12),  # glicko × h2h
        (0, 1),   # rank × pts
        (2, 14),  # glicko × skill
    ],
    'serve_return': [
        (2, 14),  # glicko × skill
        (0, 15),  # rank × momentum
        (2, 15),  # glicko × momentum
        (19, 24), # ace × return_pct
        (21, 26), # 1st_won × return_1st
        (23, 25), # bp_saved × bp_conv
        (2, 12),  # glicko × h2h
        (14, 15), # skill × momentum
        (5, 15),  # streak × momentum
        (0, 1),   # rank × pts
    ],
    'context': [
        (2, 14),  # glicko × skill
        (0, 15),  # rank × momentum
        (2, 7),   # glicko × surface
        (14, 7),  # skill × surface
        (15, 7),  # momentum × surface
        (2, 8),   # glicko × level
        (0, 1),   # rank × pts
        (2, 12),  # glicko × h2h
        (5, 15),  # streak × momentum
        (14, 15), # skill × momentum
    ],
    'minimal': [
        (2, 14),  # glicko × skill
        (0, 15),  # rank × momentum
        (2, 15),  # glicko × momentum
        (2, 12),  # glicko × h2h
        (0, 1),   # rank × pts
    ],
}
DEFAULT_INTERACTION_PAIRS = INTERACTION_SETS['core']
N_INTERACTIONS = len(DEFAULT_INTERACTION_PAIRS)
```

- [ ] **Step 3.5: Aggiorna `carica_e_prepara()` — strutture running**

Nella funzione `carica_e_prepara()`, **trova** il blocco che inizializza le strutture running (righe ~152-175):

```python
    fatiga_t          = {}   # {(tourney_id, player): minutes}
    racha_t           = {}   # {(player, surf): [last 10 results]}
    h2h_t             = {}   # {(p1, p2): [wins_p1, wins_p2]}
    h2h_surf_t        = {}   # {(p1, p2, surf): [wins_p1, wins_p2]}
    serve_t           = {}   # {player: {stat: [rolling]}}
    return_t          = {}   # {player: {stat: [rolling]}}
    elo_surf          = {}   # {(player, surf): float}
    elo_overall       = {}   # {player: float}
    streak_t          = {}   # {player: int}
    recent_form_t     = {}   # {player: [last 10 results]}
    last_match_date_t = {}   # {player: int YYYYMMDD}
    opp_quality_t     = {}   # {player: [(result, rank_opp)]}
    wrate_running     = {}   # {(player, surf): [wins, total]} — FIX BUG1: non più globale
    close_match_t     = {}   # {player: [1/0 se 3° set, last 10]}
    upset_hist_t      = {}   # {player: [(result, rank_diff), last 20]}
    late_round_t      = {}   # {player: [results in QF/SF/F]}
    match_load_t      = {}   # {player: [match days, rolling]}

    ELO_DEFAULT = 1500.0
    K_BASE      = 32.0
    K_LEVEL_ELO = {'G': 1.25, 'M': 1.15, 'F': 1.1, 'A': 1.0,
                   'D': 0.95, 'C': 0.9,  'S': 0.85, 'E': 0.8}

    def get_elo(p, s): return elo_surf.get((p, s), ELO_DEFAULT)
```

**Sostituisci con:**
```python
    fatiga_t          = {}   # {(tourney_id, player): minutes}
    racha_t           = {}   # {(player, surf): [last 10 results]}
    h2h_t             = {}   # {(p1, p2): [wins_p1, wins_p2]}
    h2h_surf_t        = {}   # {(p1, p2, surf): [wins_p1, wins_p2]}
    serve_t           = {}   # {player: {stat: [rolling]}}
    return_t          = {}   # {player: {stat: [rolling]}}
    streak_t          = {}   # {player: int}
    recent_form_t     = {}   # {player: [last 10 results]}
    last_match_date_t = {}   # {player: int YYYYMMDD}
    opp_quality_t     = {}   # {player: [(result, rank_opp)]}
    wrate_running     = {}   # {(player, surf): [wins, total]}
    close_match_t     = {}   # {player: [1/0 se 3° set, last 10]}
    upset_hist_t      = {}   # {player: [(result, rank_diff), last 20]}
    late_round_t      = {}   # {player: [results in QF/SF/F]}
    match_load_t      = {}   # {player: [match days, rolling]}

    # Glicko-2: uno store per ogni superficie + overall
    g2_hard  = Glicko2Store()
    g2_clay  = Glicko2Store()
    g2_grass = Glicko2Store()
    g2_gen   = Glicko2Store()   # overall (tutte le superfici)

    SURF_STORE = {'Hard': g2_hard, 'Clay': g2_clay, 'Grass': g2_grass}
```

- [ ] **Step 3.6: Aggiorna il blocco lettura Glicko-2 e indoor nel loop for**

**Trova** il blocco `# --- Elo superficie (PRIMA dell'aggiornamento) ---` (righe ~279-287):

```python
        # --- Elo superficie (PRIMA dell'aggiornamento) ---
        elo_w    = get_elo(w, surf);     elo_l    = get_elo(l, surf)
        elo_ov_w = elo_overall.get(w, ELO_DEFAULT)
        elo_ov_l = elo_overall.get(l, ELO_DEFAULT)
        expected_w  = 1.0 / (1.0 + 10.0 ** ((elo_l - elo_w) / 400.0))
        expected_ov = 1.0 / (1.0 + 10.0 ** ((elo_ov_l - elo_ov_w) / 400.0))
        level_code  = str(row.get('tourney_level', 'A'))
        k_dynamic   = K_BASE * K_LEVEL_ELO.get(level_code, 1.0)
        # (aggiornamento dopo lettura)
```

**Sostituisci con:**
```python
        # --- Glicko-2 superficie (PRIMA dell'aggiornamento, no leakage) ---
        store_s = SURF_STORE.get(surf, g2_gen)
        g2_r_w,  g2_rd_w,  _ = store_s.get(w)
        g2_r_l,  g2_rd_l,  _ = store_s.get(l)
        g2_ov_w, g2_ov_rd_w, _ = g2_gen.get(w)
        g2_ov_l, g2_ov_rd_l, _ = g2_gen.get(l)
        # (aggiornamento dopo lettura)

        # --- Indoor ---
        indoor_val = 0.0 if str(row.get('indoor', 'O')).upper() in ('O', '', 'NAN') else 1.0
```

- [ ] **Step 3.7: Aggiorna il blocco serve stats — skip se svpt=0**

**Trova** la funzione `upd_serve(player, rd, pref):` (riga ~492). **Trova all'interno** il blocco che legge svpt:

```python
        def upd_serve(player, rd, pref):
            s = serve_t.setdefault(player, {})
            svpt = rd.get(f'{pref}_svpt', np.nan); fi = rd.get(f'{pref}_1stIn', np.nan)
```

**Sostituisci l'intera funzione `upd_serve`** con questa versione che skipping se svpt=0:

```python
        def upd_serve(player, rd_row, pref):
            svpt = rd_row.get(f'{pref}_svpt', np.nan)
            # Skip aggiornamento se dati mancanti (svpt=0 o NaN)
            if not svpt or pd.isna(svpt) or float(svpt) <= 0:
                return
            s  = serve_t.setdefault(player, {})
            fi = rd_row.get(f'{pref}_1stIn', np.nan)
            fw = rd_row.get(f'{pref}_1stWon', np.nan)
            sw = rd_row.get(f'{pref}_2ndWon', np.nan)
            bps  = rd_row.get(f'{pref}_bpSaved', np.nan)
            bpf  = rd_row.get(f'{pref}_bpFaced', np.nan)
            svpt_f = float(svpt)
            for k2, v in [
                    ('ace',     rd_row.get(f'{pref}_ace', np.nan)),
                    ('1st_pct', float(fi)/svpt_f if fi and not pd.isna(fi) else np.nan),
                    ('1st_won', float(fw)/float(fi) if fi and float(fi) > 0 and not pd.isna(fw) else np.nan),
                    ('2nd_won', float(sw)/(svpt_f - float(fi)) if fi and (svpt_f - float(fi)) > 0 and not pd.isna(sw) else np.nan),
                    ('bp_saved', float(bps)/float(bpf) if bpf and float(bpf) > 0 and not pd.isna(bps) else np.nan)]:
                if v is not np.nan and not (isinstance(v, float) and np.isnan(v)):
                    lst = s.setdefault(k2, []); lst.append(float(v))
                    if len(lst) > 10: lst.pop(0)
```

- [ ] **Step 3.8: Aggiorna la costruzione del vettore feature (blocco `diffs`)**

**Trova** il blocco `diffs = {` (riga ~376) e **sostituisci**:

```python
        diffs = {
            'log_rank_ratio':        np.log1p(rk_l)  - np.log1p(rk_w),
            'log_pts_ratio':         np.log1p(pts_w) - np.log1p(pts_l),
            'diff_elo':              elo_w   - elo_l,
            'diff_elo_overall':      elo_ov_w - elo_ov_l,
            'diff_streak':           float(str_w - str_l),
            'diff_recent_form':      form_w  - form_l,
            'diff_form_volatility':  fvol_w  - fvol_l,       # NEW v5
            'surface_enc':           float(row['surface_enc']),
            'tourney_level':         float(row['tourney_level_enc']),
            'round_enc':             float(row['round_enc']),
            'is_best_of_5':          is_bo5,
            'diff_h2h':              h2h_w   - h2h_l,
            'diff_h2h_surface':      h2h_s_w - h2h_s_l,
            'diff_skill':            sk_w    - sk_l,
            'diff_momentum':         mw      - ml,
            'diff_surface_trend':    st_w    - st_l,          # NEW v5
            'diff_fatigue':          f_w     - f_l,
            'diff_days_since_last':  days_since_w - days_since_l,
            'diff_weeks_load':       wload_w - wload_l,       # NEW v5
            'diff_ace':              sa_w['ace']      - sa_l['ace'],
            'diff_1st_pct':          sa_w['1st_pct']  - sa_l['1st_pct'],
            'diff_1st_won':          sa_w['1st_won']  - sa_l['1st_won'],
            'diff_2nd_won':          sa_w['2nd_won']  - sa_l['2nd_won'],
            'diff_bp_saved':         sa_w['bp_saved'] - sa_l['bp_saved'],
            'diff_return_pct':       ra_w['return_pct'] - ra_l['return_pct'],
            'diff_bp_conv':          ra_w['bp_conv']    - ra_l['bp_conv'],
            'diff_return_1st':       ra_w['return_1st'] - ra_l['return_1st'],
            'diff_home':             home_w  - home_l,
            'diff_opponent_quality': oq_w    - oq_l,
            'diff_close_match_pct':  cmp_w   - cmp_l,         # NEW v5
            'diff_upset_tendency':   up_w    - up_l,           # NEW v5
            'diff_late_round_wr':    lrwr_w  - lrwr_l,         # NEW v5
            'court_ace_pct':         court_ace,
            'court_speed':           court_spd,
            'level_weight':          lev_w,
        }
```

**Con:**
```python
        diffs = {
            'log_rank_ratio':        np.log1p(rk_l)  - np.log1p(rk_w),
            'log_pts_ratio':         np.log1p(pts_w) - np.log1p(pts_l),
            'diff_glicko':           g2_r_w    - g2_r_l,
            'diff_glicko_overall':   g2_ov_w   - g2_ov_l,
            'diff_glicko_rd':        g2_rd_w   - g2_rd_l,
            'diff_streak':           float(str_w - str_l),
            'diff_recent_form':      form_w    - form_l,
            'surface_enc':           float(row['surface_enc']),
            'tourney_level':         float(row['tourney_level_enc']),
            'round_enc':             float(row['round_enc']),
            'is_best_of_5':          is_bo5,
            'indoor':                indoor_val,
            'diff_h2h':              h2h_w     - h2h_l,
            'diff_h2h_surface':      h2h_s_w   - h2h_s_l,
            'diff_skill':            sk_w      - sk_l,
            'diff_momentum':         mw        - ml,
            'diff_fatigue':          f_w       - f_l,
            'diff_days_since_last':  days_since_w - days_since_l,
            'diff_weeks_load':       wload_w   - wload_l,
            'diff_ace':              sa_w['ace']        - sa_l['ace'],
            'diff_1st_pct':          sa_w['1st_pct']    - sa_l['1st_pct'],
            'diff_1st_won':          sa_w['1st_won']    - sa_l['1st_won'],
            'diff_2nd_won':          sa_w['2nd_won']    - sa_l['2nd_won'],
            'diff_bp_saved':         sa_w['bp_saved']   - sa_l['bp_saved'],
            'diff_return_pct':       ra_w['return_pct'] - ra_l['return_pct'],
            'diff_bp_conv':          ra_w['bp_conv']    - ra_l['bp_conv'],
            'diff_return_1st':       ra_w['return_1st'] - ra_l['return_1st'],
            'diff_home':             home_w    - home_l,
            'diff_opponent_quality': oq_w      - oq_l,
            'diff_upset_tendency':   up_w      - up_l,
            'diff_late_round_wr':    lrwr_w    - lrwr_l,
            'level_weight':          lev_w,
        }
```

- [ ] **Step 3.9: Aggiorna `_SYMM_KEYS` e aggiorna l'aggiornamento Glicko-2 post-match**

**Trova** `_SYMM_KEYS`:
```python
        _SYMM_KEYS = ('surface_enc', 'tourney_level', 'round_enc', 'is_best_of_5',
                      'level_weight', 'court_ace_pct', 'court_speed')
```
**Sostituisci con:**
```python
        _SYMM_KEYS = ('surface_enc', 'tourney_level', 'round_enc', 'is_best_of_5',
                      'indoor', 'level_weight')
```

**Trova** il blocco aggiornamento ELO post-match (righe ~459-462):
```python
        elo_surf[(w, surf)] = elo_w + k_dynamic * (1.0 - expected_w)
        elo_surf[(l, surf)] = elo_l + k_dynamic * (0.0 - (1.0 - expected_w))
        elo_overall[w]      = elo_ov_w + k_dynamic * (1.0 - expected_ov)
        elo_overall[l]      = elo_ov_l + k_dynamic * (0.0 - (1.0 - expected_ov))
```
**Sostituisci con:**
```python
        # Aggiorna Glicko-2 superficie e overall (post-match)
        store_s.update_match(w, l)
        g2_gen.update_match(w, l)
```

- [ ] **Step 3.10: Aggiorna il blocco joblib.dump alla fine di carica_e_prepara()**

**Trova** il blocco che salva gli artefatti ELO (righe ~543-557) e **sostituisci** le due righe ELO:

```python
    joblib.dump(elo_surf,          'elo_surface.pkl');           print("   → elo_surface.pkl")
    joblib.dump(elo_overall,       'elo_overall.pkl');           print("   → elo_overall.pkl")
```

**Con:**
```python
    # Glicko-2 stores (per superficie + overall)
    glicko2_surf = {
        'Hard': g2_hard.to_dict(), 'Clay': g2_clay.to_dict(),
        'Grass': g2_grass.to_dict(), 'Gen': g2_gen.to_dict()
    }
    joblib.dump(glicko2_surf, 'glicko2_stores.pkl'); print("   → glicko2_stores.pkl")
    # Salva anche serve_t e return_t (serve per il prediction engine)
    joblib.dump(serve_t,  'serve_stats.pkl');  print("   → serve_stats.pkl")
    joblib.dump(return_t, 'return_stats.pkl'); print("   → return_stats.pkl")
```

E aggiorna il `return` della funzione — **trova**:
```python
    return df_out, wrate_final, elo_surf, streak_t
```
**Sostituisci con:**
```python
    return df_out, wrate_final, glicko2_surf, streak_t
```

- [ ] **Step 3.11: Aggiorna il CSV_CANDIDATES nel main**

**Trova**:
```python
    CSV_CANDIDATES = [
        'historialTenis.csv', '../scraping/historialTenis.csv',
        'historial_tenis_COMPLETO.csv', '../scraping/historial_tenis_COMPLETO.csv',
    ]
```
**Sostituisci con:**
```python
    CSV_CANDIDATES = [
        '../scraping/master_dataset.csv',
        'master_dataset.csv',
        '../scraping/historialTenis.csv',   # fallback legacy
        'historialTenis.csv',
    ]
```

- [ ] **Step 3.12: Verifica sintattica**

```powershell
cd "C:\Users\riccardo\Desktop\tennis\prediccion"
..\venv\Scripts\python.exe -c "import train_ann; print('OK')" 2>&1 | head -20
```

Expected: `OK` (nessun SyntaxError)

- [ ] **Step 3.13: Commit**

```powershell
cd "C:\Users\riccardo\Desktop\tennis"
git add prediccion/train_ann.py
git commit -m "feat: feature engineering v6 — Glicko-2, indoor, serve stats fix, 32 feature"
```

---

## Task 4: Surface-Specific LGB + Calibrazione Fix + Log-Loss (train_ann.py — parte 2)

**Files:**
- Modify: `prediccion/train_ann.py`

- [ ] **Step 4.1: Aggiorna `train_lgb_optuna()` — ottimizza log-loss**

**Trova** nella funzione `train_lgb_optuna()` il blocco objective:

```python
        acc  = accuracy_score(y_val_np, (probs >= 0.5).astype(int))
        if acc > best_acc[0]: best_acc[0] = acc; best_model[0] = model
        return acc
```
**Sostituisci con:**
```python
        ll = log_loss(y_val_np, probs)
        if ll < best_acc[0]: best_acc[0] = ll; best_model[0] = model
        return ll
```

**Trova**:
```python
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        model = best_model[0]
        print(f"   LightGBM best val: {best_acc[0]:.4f}")
```
**Sostituisci con:**
```python
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        model = best_model[0]
        print(f"   LightGBM best val log-loss: {best_acc[0]:.4f}")
```

Inizializza anche `best_acc` con `float('inf')` invece di `0.0`:

**Trova**:
```python
    best_model = [None]; best_acc = [0.0]
```
**Sostituisci con (solo in `train_lgb_optuna`):**
```python
    best_model = [None]; best_acc = [float('inf')]
```

- [ ] **Step 4.2: Aggiorna `train_xgb_optuna()` — stessa cosa**

Stesso cambio in `train_xgb_optuna()`:
- `best_acc = [0.0]` → `best_acc = [float('inf')]`
- `if acc > best_acc[0]` → `if ll < best_acc[0]`; `return acc` → `return ll`
- `direction='maximize'` → `direction='minimize'`

- [ ] **Step 4.3: Aggiorna `optuna_search()` ANN — ottimizza log-loss**

**Trova** nella funzione `optuna_search()` l'objective:

```python
            acc_v, _ = valuta(model, X_val_sc, y_val_np)
            trial.set_user_attr('model', model)
            ...
            return acc_v
```
**Sostituisci con:**
```python
            _, ll_v = valuta(model, X_val_sc, y_val_np)
            trial.set_user_attr('model', model)
            ...
            return ll_v
```

**Trova**:
```python
        study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=SEED))
```
**Sostituisci con:**
```python
        study = optuna.create_study(direction='minimize',
                                    sampler=optuna.samplers.TPESampler(seed=SEED))
```

**Trova** l'ordinamento finale dei risultati:
```python
    return sorted(risultati, key=lambda x: x['test_acc'], reverse=True)
```
**Sostituisci con:**
```python
    return sorted(risultati, key=lambda x: x['test_acc'], reverse=True)  # mantieni ordinamento per accuracy finale
```
(Invariato — la selezione finale usa sempre accuracy, Optuna usa log-loss per la ricerca interna.)

- [ ] **Step 4.4: Aggiungi `train_lgb_surface_specific()` dopo `train_xgb_optuna()`**

Aggiungi questa funzione **dopo** la funzione `train_xgb_optuna()`:

```python
def train_lgb_surface_specific(X_tr_df, y_tr, X_val_df, y_val,
                                X_test_df, y_test,
                                sample_weights_tr=None, n_trials=TRIALS_GBM):
    """
    Allena un LightGBM separato per Hard, Clay, Grass.
    Restituisce un dict {surf_name: model} e le accuracy per superficie.
    """
    if not HAS_LGB:
        print("   ⚠ LightGBM non disponibile, skip surface models")
        return {}, {}

    SURFACES = {0: 'Hard', 1: 'Clay', 2: 'Grass'}
    models = {}
    accs   = {}

    for surf_enc, surf_name in SURFACES.items():
        mask_tr   = X_tr_df['surface_enc'] == surf_enc
        mask_val  = X_val_df['surface_enc'] == surf_enc
        mask_test = X_test_df['surface_enc'] == surf_enc

        n_tr = mask_tr.sum()
        if n_tr < 200:
            print(f"   ⚠ {surf_name}: solo {n_tr} esempi train — skip")
            continue

        X_tr_s  = X_tr_df[mask_tr][FEATURES].fillna(0).values
        y_tr_s  = y_tr[mask_tr].values if hasattr(y_tr[mask_tr], 'values') else y_tr[mask_tr]
        X_val_s = X_val_df[mask_val][FEATURES].fillna(0).values
        y_val_s = y_val[mask_val].values if hasattr(y_val[mask_val], 'values') else y_val[mask_val]
        X_te_s  = X_test_df[mask_test][FEATURES].fillna(0).values
        y_te_s  = y_test[mask_test].values if hasattr(y_test[mask_test], 'values') else y_test[mask_test]
        w_s     = sample_weights_tr[mask_tr] if sample_weights_tr is not None else None

        print(f"\n  🌍 LGB Surface: {surf_name} ({n_tr:,} train | {mask_val.sum():,} val)")
        model, acc, ll = train_lgb_optuna(
            X_tr_s, y_tr_s, X_val_s, y_val_s, X_te_s, y_te_s,
            sample_weights_tr=w_s, n_trials=n_trials
        )
        if model is not None:
            models[surf_name] = model
            accs[surf_name]   = acc
            print(f"    → {surf_name}: acc={acc:.4f} | log_loss={ll:.4f}")

    return models, accs
```

- [ ] **Step 4.5: Fix calibrazione — fit su validation set, non test set**

**Trova** nel main (sezione `# ── Calibrazione isotonica su test set`):

```python
    raw_probs_test = top5_ann_probs_test.flatten()
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(raw_probs_test, y_test_np)
    calibrated_probs_test = calibrator.predict(raw_probs_test)
```

**Sostituisci con:**
```python
    # FIX: calibratore fittato su VALIDATION SET, test set rimane pulito
    raw_probs_val  = np.mean([get_ann_probs(r['_model'], X_val_sc) for r in risultati[:5]], axis=0).flatten()
    raw_probs_test = top5_ann_probs_test.flatten()
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(raw_probs_val, y_val_np)            # <— val, non test
    calibrated_probs_test = calibrator.predict(raw_probs_test)
```

- [ ] **Step 4.6: Aggiungi le strategie surface LGB all'ensemble e alla selezione**

Nel main, **dopo** il blocco che calcola `lgb_model` e `xgb_model`, aggiungi:

```python
    # ── Surface-specific LGB ──────────────────────────────────────────────────
    # Passiamo i DataFrame originali (con surface_enc) per il routing
    surf_lgb_models, surf_lgb_accs = train_lgb_surface_specific(
        df_tr, y_tr, df_val, y_val, df_test, y_test,
        sample_weights_tr=combined_weights, n_trials=TRIALS_GBM
    )

    # Calcola accuracy ensemble surface LGB sul test set
    if surf_lgb_models:
        surf_map_enc = {'Hard': 0, 'Clay': 1, 'Grass': 2}
        surf_preds = np.full(len(y_test_np), 0.5)
        for sname, smodel in surf_lgb_models.items():
            smask = df_test['surface_enc'] == surf_map_enc[sname]
            if smask.sum() > 0:
                surf_preds[smask.values] = smodel.predict_proba(
                    df_test[smask][FEATURES].fillna(0).values
                )[:, 1]
        # Fallback su LGB global per superficie non coperta
        if lgb_model is not None:
            no_surf = ~df_test['surface_enc'].isin([0, 1, 2]).values
            if no_surf.sum() > 0:
                surf_preds[no_surf] = lgb_model.predict_proba(X_te_sc[no_surf])[:, 1]
        surf_acc = accuracy_score(y_test_np, (surf_preds >= 0.5).astype(int))
        surf_ll  = log_loss(y_test_np, surf_preds)
        print(f"\n   LGB Surface ensemble: acc={surf_acc:.4f} | log_loss={surf_ll:.4f}")
    else:
        surf_acc, surf_ll, surf_preds = 0.0, float('inf'), None
```

**Aggiungi `lgb_surface` alla `results_list`** — **trova**:
```python
    results_list = [
        {'Modello': 'ANN Best', ...
```

**Aggiungi alla lista** (dopo l'ultimo elemento prima della chiusura `]`):
```python
    if surf_lgb_models:
        results_list.append({
            'Modello': 'LGB Surface',
            'Accuracy': surf_acc, 'Log Loss': surf_ll,
            'Note': f'LGB Hard/Clay/Grass routing',
            '_strategy': 'lgb_surface'
        })
```

- [ ] **Step 4.7: Salva i modelli surface LGB nel `modelo_finale`**

**Trova** il blocco `modelo_finale = {` e aggiungi `'lgb_surface_models'`:

```python
    modelo_finale = {
        ...
        'ann_top5_uncertainty': top5_for_uncertainty,
    }
```

**Dopo `ann_top5_uncertainty`** aggiungi:
```python
        'lgb_surface_models': surf_lgb_models,   # {surf_name: model} per routing
```

**Nella sezione `if best_strategy == 'lgb_surface':`** (aggiungila nel blocco elif):

Trova il blocco:
```python
    elif best_strategy in ('ensemble_avg', 'ensemble_stacking'):
        modelo_finale['lgb_model'] = lgb_final; modelo_finale['xgb_model'] = xgb_final
```

**Prima di quel blocco** aggiungi:
```python
    if best_strategy == 'lgb_surface':
        # Re-training surface models su tutti i dati
        print("   Re-training LGB Surface models su tutti i dati...")
        surf_final_models = {}
        for sname, sm in surf_lgb_models.items():
            senc = {'Hard': 0, 'Clay': 1, 'Grass': 2}[sname]
            mask_all_s = df_ml['surface_enc'] == senc
            if mask_all_s.sum() > 0:
                X_all_s = X_all[mask_all_s]
                y_all_s = y_all_np[mask_all_s]
                w_all_s = all_weights[mask_all_s]
                m_s = lgb.LGBMClassifier(**sm.get_params())
                m_s.fit(X_all_s, y_all_s, sample_weight=w_all_s)
                surf_final_models[sname] = m_s
        modelo_finale['lgb_surface_models'] = surf_final_models
        modelo_finale['lgb_model'] = lgb_final  # fallback globale
```

- [ ] **Step 4.8: Verifica sintattica**

```powershell
cd "C:\Users\riccardo\Desktop\tennis\prediccion"
..\venv\Scripts\python.exe -c "import train_ann; print('OK')" 2>&1 | head -20
```

Expected: `OK`

- [ ] **Step 4.9: Commit**

```powershell
cd "C:\Users\riccardo\Desktop\tennis"
git add prediccion/train_ann.py
git commit -m "feat: surface LGB, calibrazione su val set, ottimizza log-loss"
```

---

## Task 5: Aggiornamento Prediction Engine

**Files:**
- Modify: `prediccion/prediction_engine.py`

- [ ] **Step 5.1: Aggiorna ANN_FEATURES e DEFAULT_INTERACTION_PAIRS**

**Trova**:
```python
ANN_FEATURES = [
    'log_rank_ratio', 'log_pts_ratio',
    'diff_elo', 'diff_elo_overall',
    ...
]  # 30 feature (v5.1)
```

**Sostituisci con:**
```python
ANN_FEATURES = [
    'log_rank_ratio', 'log_pts_ratio',
    'diff_glicko', 'diff_glicko_overall', 'diff_glicko_rd',
    'diff_streak', 'diff_recent_form',
    'surface_enc', 'tourney_level', 'round_enc',
    'is_best_of_5', 'indoor',
    'diff_h2h', 'diff_h2h_surface',
    'diff_skill', 'diff_momentum',
    'diff_fatigue', 'diff_days_since_last', 'diff_weeks_load',
    'diff_ace', 'diff_1st_pct', 'diff_1st_won',
    'diff_2nd_won', 'diff_bp_saved',
    'diff_return_pct', 'diff_bp_conv', 'diff_return_1st',
    'diff_home', 'diff_opponent_quality',
    'diff_upset_tendency', 'diff_late_round_wr',
    'level_weight',
]  # 32 feature (v6.0)
```

**Trova** `DEFAULT_INTERACTION_PAIRS` e **sostituisci**:
```python
DEFAULT_INTERACTION_PAIRS = [
    (2, 14), (0, 15), (2, 15), (14, 15), (0, 1),
    (2, 12), (5, 15), (14, 16), (16, 15), (1, 14),
]
N_INTERACTIONS = len(DEFAULT_INTERACTION_PAIRS)
```

- [ ] **Step 5.2: Aggiungi import Glicko2Store e strategia lgb_surface alla funzione `predici()`**

**All'inizio di `prediction_engine.py`** aggiungi import:
```python
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from glicko2 import Glicko2Store
```

**Nella funzione `predici()`**, aggiungi il caso `lgb_surface` — **trova**:
```python
    elif s == 'lgb':
        return float(modelo_finale['lgb_model'].predict_proba(input_sc)[:, 1][0]), \
               modelo_finale['model_name']
```

**Prima di quel blocco** aggiungi:
```python
    elif s == 'lgb_surface':
        surf_models = modelo_finale.get('lgb_surface_models', {})
        # Determina superficie dall'input_sc (indice 7 = surface_enc)
        surf_enc_val = int(round(float(input_sc[0, 7])))
        surf_name = {0: 'Hard', 1: 'Clay', 2: 'Grass'}.get(surf_enc_val)
        model_s = surf_models.get(surf_name) if surf_name else None
        if model_s is None:
            # Fallback: lgb globale
            model_s = modelo_finale.get('lgb_model')
        if model_s is None:
            raise ValueError("lgb_surface: nessun modello disponibile")
        return float(model_s.predict_proba(input_sc)[:, 1][0]), modelo_finale['model_name']
```

- [ ] **Step 5.3: Aggiorna `predecir_proximas.py` — ELO → Glicko-2**

**In `prediccion/predecir_proximas.py` riga 20**, aggiungi import:
```python
from prediction_engine import (
    ANN_FEATURES, SURFACE_MAP, LEVEL_LABEL, LEVEL_MULT_LABEL, ROUND_MAP_STR,
    BK_OVERROUND, TennisANNv3, _build_ann, _ann_prob, predici,
    calc_oq, days_since_last, weeks_load, upset_tendency, late_round_wr,
    Glicko2Store,   # <-- aggiungi questo
)
```

**Righe 62-63** — sostituisci caricamento artefatti ELO:
```python
# Vecchio:
res['elo_surface']      = load(pp('elo_surface.pkl'),       {})
res['elo_overall']      = load(pp('elo_overall.pkl'),       {})

# Nuovo:
glicko2_raw = load(pp('glicko2_stores.pkl'), {})
res['glicko2_surf_store'] = {
    s: Glicko2Store.from_dict(d) for s, d in glicko2_raw.items()
}
res['serve_stats']  = load(pp('serve_stats.pkl'),  {})
res['return_stats'] = load(pp('return_stats.pkl'), {})
```

**Righe 133-134 e 153-156** nella funzione `build_features()` — sostituisci:
```python
# Vecchio (righe 133-134):
elo_surface      = res['elo_surface']
elo_overall      = res['elo_overall']
# Vecchio (righe 152-156):
ELO_DEFAULT = 1500.0
elo1  = elo_surface.get((p1, superficie), ELO_DEFAULT)
elo2  = elo_surface.get((p2, superficie), ELO_DEFAULT)
elov1 = elo_overall.get(p1, ELO_DEFAULT)
elov2 = elo_overall.get(p2, ELO_DEFAULT)

# Nuovo:
g2_stores = res['glicko2_surf_store']
g2_s  = g2_stores.get(superficie, g2_stores.get('Gen', Glicko2Store()))
g2_gen = g2_stores.get('Gen', Glicko2Store())
g2_r1,  g2_rd1,  _ = g2_s.get(p1)
g2_r2,  g2_rd2,  _ = g2_s.get(p2)
g2_ov1, g2_ovrd1, _ = g2_gen.get(p1)
g2_ov2, g2_ovrd2, _ = g2_gen.get(p2)
```

**Righe 204-234** nel dizionario `row {}` — sostituisci le tre righe ELO e aggiungi indoor:
```python
# Vecchio (righe 207-208):
'diff_elo':              elo1 - elo2,
'diff_elo_overall':      elov1 - elov2,

# Nuovo:
'diff_glicko':           g2_r1  - g2_r2,
'diff_glicko_overall':   g2_ov1 - g2_ov2,
'diff_glicko_rd':        g2_rd1 - g2_rd2,
```

Aggiungi anche `indoor` subito dopo `is_best_of_5`:
```python
'is_best_of_5':          1.0 if best_of == 5 else 0.0,
'indoor':                0.0,   # outdoor di default (torneo live non ha info indoor)
```

Aggiorna il commento della funzione `build_features`:
```python
def build_features(p1, p2, superficie, livello, turno, res):
    """Costruisce le 32 feature v6 per una coppia di giocatori."""
```

**Aggiorna serve/return stats** usando `serve_stats.pkl` invece dei profili:
```python
# Dopo la lettura di sa1, sa2 (riga ~147), aggiungi:
serve_stats  = res.get('serve_stats', {})
return_stats = res.get('return_stats', {})

def _get_serve(player, key, default):
    vals = serve_stats.get(player, {}).get(key, [])
    return float(np.mean(vals)) if vals else default

def _get_return(player, key, default):
    vals = return_stats.get(player, {}).get(key, [])
    return float(np.mean(vals)) if vals else default

# Poi usa queste funzioni per le feature serve/return nel dict row:
'diff_ace':     _get_serve(p1, 'ace', 5.0)      - _get_serve(p2, 'ace', 5.0),
'diff_1st_pct': _get_serve(p1, '1st_pct', 0.62) - _get_serve(p2, '1st_pct', 0.62),
'diff_1st_won': _get_serve(p1, '1st_won', 0.70) - _get_serve(p2, '1st_won', 0.70),
'diff_2nd_won': _get_serve(p1, '2nd_won', 0.50) - _get_serve(p2, '2nd_won', 0.50),
'diff_bp_saved':    _get_serve(p1, 'bp_saved', 0.62) - _get_serve(p2, 'bp_saved', 0.62),
'diff_return_pct':  _get_return(p1, 'return_pct', 0.35) - _get_return(p2, 'return_pct', 0.35),
'diff_bp_conv':     _get_return(p1, 'bp_conv', 0.35)    - _get_return(p2, 'bp_conv', 0.35),
'diff_return_1st':  _get_return(p1, 'return_1st', 0.30) - _get_return(p2, 'return_1st', 0.30),
```

- [ ] **Step 5.4: Verifica che `build_features()` restituisca esattamente 32 feature nell'ordine corretto**

```powershell
cd "C:\Users\riccardo\Desktop\tennis\prediccion"
..\venv\Scripts\python.exe -c "
from prediction_engine import ANN_FEATURES
print(f'Feature count: {len(ANN_FEATURES)}')
for i, f in enumerate(ANN_FEATURES):
    print(f'  {i:2d}: {f}')
"
```

Expected: 32 feature, ordine identico a `FEATURES` in `train_ann.py`.

- [ ] **Step 5.5: Verifica sintattica**

```powershell
cd "C:\Users\riccardo\Desktop\tennis\prediccion"
..\venv\Scripts\python.exe -c "import prediction_engine; print('OK')" 2>&1 | head -20
```

Expected: `OK`

- [ ] **Step 5.6: Commit**

```powershell
cd "C:\Users\riccardo\Desktop\tennis"
git add prediccion/prediction_engine.py prediccion/predecir_proximas.py
git commit -m "feat: prediction engine v6 — Glicko-2, serve stats, lgb_surface routing"
```

---

## Task 6: Validazione End-to-End

**Files:** nessuno — solo esecuzione e verifica

- [ ] **Step 6.1: Esegui tutti i test unit**

```powershell
cd "C:\Users\riccardo\Desktop\tennis"
.\venv\Scripts\python.exe -m pytest tests/ -v
```

Expected: tutti i test passano (19 test totali).

- [ ] **Step 6.2: Esegui il training completo**

```powershell
cd "C:\Users\riccardo\Desktop\tennis\prediccion"
..\venv\Scripts\python.exe train_ann.py 2>&1 | Tee-Object -FilePath training_log.txt
```

Può richiedere 30-120 minuti (100 trial Optuna ANN + 50 LGB × 3 superfici + 50 XGB).

- [ ] **Step 6.3: Verifica risultati finali**

```powershell
cd "C:\Users\riccardo\Desktop\tennis\prediccion"
Get-Content training_log.txt | Select-String "CLASSIFICA|Miglior|test_acc|Surface|val_score"
```

Expected:
- Test accuracy migliore ≥ 69% (target 71-73%)
- `lgb_surface` o `ensemble_avg` dovrebbe essere tra i top candidati
- Nessun errore in output

- [ ] **Step 6.4: Verifica artefatti generati**

```powershell
cd "C:\Users\riccardo\Desktop\tennis\prediccion"
Get-ChildItem *.pkl | Select-Object Name, @{N='KB';E={[math]::Round($_.Length/1KB)}}
```

Expected: presenza di `glicko2_stores.pkl`, `serve_stats.pkl`, `return_stats.pkl`, `modelo_finale.pkl`, `lgb_hard.pkl` ecc.

- [ ] **Step 6.5: Smoke test del prediction engine**

```powershell
cd "C:\Users\riccardo\Desktop\tennis\prediccion"
..\venv\Scripts\python.exe -c "
import joblib, numpy as np
from prediction_engine import predici, ANN_FEATURES
import torch

m = joblib.load('modelo_finale.pkl')
scaler = m['scaler']
# Feature vector fittizio (32 feature)
fv = np.zeros((1, len(ANN_FEATURES)), dtype=np.float32)
fv[0, 0] = 0.5   # log_rank_ratio
fv[0, 2] = 100.0  # diff_glicko
fv_sc = scaler.transform(fv)
fv_t  = torch.tensor(fv_sc.astype(np.float32))
prob, nome = predici(fv_sc, fv_t, m)
print(f'Probabilità: {prob:.3f} | Modello: {nome}')
assert 0 < prob < 1, 'Probabilità fuori range'
print('OK')
"
```

Expected: probabilità tra 0 e 1, nessun errore.

- [ ] **Step 6.6: Commit finale**

```powershell
cd "C:\Users\riccardo\Desktop\tennis"
git add prediccion/training_log.txt
git commit -m "feat: tennis predictor v6.0 — validazione end-to-end completata"
```
