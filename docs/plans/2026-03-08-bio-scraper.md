# Bio Scraper (tennisstats.com) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Scrape real age (DOB) and height for every player from tennisstats.com, save to a persistent JSON file, and use it inside generar_perfiles.py instead of the accumulating `age + 1.5` hack.

**Architecture:** A new standalone script `scraper_bio_jugadores.py` fetches each player page with `requests` + `BeautifulSoup`, parses `.data-table-row` cells, and writes `bio_jugadores.json`. `generar_perfiles.py` loads that file at startup and overrides height/age. `aggiorna_tutto.py` exposes a `ESEGUI_BIO` flag that triggers the scraper inside FASE 5 before `generar_perfiles.py` runs.

**Tech Stack:** Python 3, `requests`, `BeautifulSoup4`, `json`, `unicodedata` (stdlib), `joblib`, `pandas`

---

## Context

- Working directory for all scraping scripts: `scraping/`
- URL pattern: `https://tennisstats.com/players/{slug}` where slug = lowercase name with hyphens (e.g. "Carlos Alcaraz" → "carlos-alcaraz")
- Page renders without JavaScript — simple `requests.get` is sufficient
- Data-table HTML structure: `.data-table .data-table-row` with two `.item` cells (label + value)
- DOB format on site: "August 16, 2001" → parse with `datetime.strptime(s, "%B %d, %Y")`
- Height format on site: "1.91m" → `float("1.91") * 100 = 191` cm
- Player list: `ranking_2026.csv` column `player_real` (already cleaned names)
- Bio file persists across runs: only overwritten when `ESEGUI_BIO = True`

---

## Task 1: Create `scraping/scraper_bio_jugadores.py`

**Files:**
- Create: `scraping/scraper_bio_jugadores.py`

**Step 1: Write the script**

```python
"""
scraper_bio_jugadores.py
Scrapes DOB and height for every player in ranking_2026.csv
from https://tennisstats.com/players/{slug}

OUTPUT: bio_jugadores.json  →  {"Carlos Alcaraz": {"dob": "2003-05-05", "ht": 183}, ...}
"""
import json
import time
import unicodedata
import re
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://tennisstats.com/players/{slug}"
OUTPUT   = "bio_jugadores.json"
DELAY    = 0.6   # seconds between requests (be polite)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def name_to_slug(name: str) -> str:
    """'Carlos Alcaraz' → 'carlos-alcaraz'"""
    # Remove accents
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = nfkd.encode("ascii", "ignore").decode("ascii")
    # Lowercase, hyphens for spaces, strip non-alphanumeric except hyphen
    slug = ascii_name.lower().strip()
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"\s+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    return slug


def parse_data_table(soup: BeautifulSoup) -> dict:
    """Extract label→value pairs from .data-table-row elements."""
    data = {}
    for row in soup.select(".data-table-row"):
        cells = row.select(".item")
        if len(cells) >= 2:
            label = cells[0].get_text(strip=True).lower()
            value = cells[1].get_text(strip=True)
            data[label] = value
    return data


def parse_dob(raw: str) -> str | None:
    """'November 27, 1998' → '1998-11-27' (ISO format)"""
    try:
        dt = datetime.strptime(raw.strip(), "%B %d, %Y")
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return None


def parse_height(raw: str) -> int | None:
    """'1.93m' or '193 cm' → 193"""
    raw = raw.strip().lower()
    # Format: "1.93m"
    m = re.match(r"^(\d+\.\d+)m$", raw)
    if m:
        return round(float(m.group(1)) * 100)
    # Format: "193 cm" or "193cm"
    m = re.match(r"^(\d+)\s*cm$", raw)
    if m:
        return int(m.group(1))
    return None


def scrape_player(name: str, session: requests.Session) -> dict | None:
    slug = name_to_slug(name)
    url  = BASE_URL.format(slug=slug)
    try:
        resp = session.get(url, timeout=10)
        if resp.status_code == 404:
            print(f"   ⚠️  404 — {name} ({slug})")
            return None
        resp.raise_for_status()
        soup  = BeautifulSoup(resp.text, "html.parser")
        table = parse_data_table(soup)
        dob   = parse_dob(table.get("date of birth", ""))
        ht    = parse_height(table.get("height", ""))
        if not dob and not ht:
            print(f"   ⚠️  Dati non trovati — {name} ({slug})")
            return None
        return {"dob": dob, "ht": ht}
    except Exception as e:
        print(f"   ❌  Errore {name}: {e}")
        return None


def main():
    # Load player list from ranking
    try:
        df = pd.read_csv("ranking_2026.csv")
        # Use already-cleaned real names
        if "player_real" in df.columns:
            players = df["player_real"].dropna().unique().tolist()
        else:
            players = df["player"].dropna().unique().tolist()
    except Exception as e:
        print(f"❌  Impossibile caricare ranking_2026.csv: {e}")
        return

    print(f"👤 BIO SCRAPER — {len(players)} giocatori da scrapare")

    # Load existing bio to keep entries not scraped this run
    try:
        with open(OUTPUT, "r", encoding="utf-8") as f:
            bio = json.load(f)
        print(f"   📂 Bio esistente caricata ({len(bio)} voci)")
    except FileNotFoundError:
        bio = {}

    session = requests.Session()
    session.headers.update(HEADERS)

    ok = 0
    errors = 0
    for i, name in enumerate(players, 1):
        print(f"   [{i}/{len(players)}] {name}", end=" ... ", flush=True)
        result = scrape_player(name, session)
        if result:
            bio[name] = result
            print(f"✅  DOB={result['dob']}  ht={result['ht']}cm")
            ok += 1
        else:
            errors += 1
        time.sleep(DELAY)

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(bio, f, ensure_ascii=False, indent=2)

    print(f"\n✅  bio_jugadores.json salvato — {ok} OK, {errors} errori")


if __name__ == "__main__":
    main()
```

**Step 2: Test manually with one player**

```bash
cd scraping
python -c "
import requests, json
from bs4 import BeautifulSoup
resp = requests.get('https://tennisstats.com/players/carlos-alcaraz',
    headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
soup = BeautifulSoup(resp.text, 'html.parser')
rows = soup.select('.data-table-row')
for r in rows:
    cells = r.select('.item')
    if len(cells) >= 2:
        print(cells[0].get_text(strip=True), '->', cells[1].get_text(strip=True))
"
```

Expected output includes lines like:
```
Date of Birth -> May 5, 2003
Height -> 1.83m
Age -> 22
```

If the structure differs (e.g. `<tr>` instead of `.data-table-row`), update `parse_data_table` to match.

**Step 3: Run the full scraper**

```bash
cd scraping
python scraper_bio_jugadores.py
```

Expected: creates `scraping/bio_jugadores.json` with correct DOB and height for each player.

---

## Task 2: Update `generar_perfiles.py` to use bio data

**Files:**
- Modify: `scraping/generar_perfiles.py`

**Step 1: Load bio_jugadores.json at the start of the script**

After the existing `import` block (around line 4), add:

```python
import json
from datetime import date

# Carica bio reale da tennisstats.com (se disponibile)
try:
    with open("bio_jugadores.json", "r", encoding="utf-8") as _f:
        BIO_REAL = json.load(_f)
    print(f"   ✅ bio_jugadores.json caricato ({len(BIO_REAL)} giocatori)")
except FileNotFoundError:
    BIO_REAL = {}
    print("   ⚠️  bio_jugadores.json non trovato — uso dati storici ATP")
```

**Step 2: Add a helper function to compute age from DOB**

Right after the block above, add:

```python
def calcola_eta(dob_str: str) -> float:
    """'1998-11-27' → età attuale in anni (float)."""
    try:
        dob = date.fromisoformat(dob_str)
        today = date.today()
        return (today - dob).days / 365.25
    except Exception:
        return 25.0
```

**Step 3: Override height and age from bio data in the profile-building loop**

In the section that finalises each player profile (second loop, lines ~189–220 of current file), add at the beginning of `for jugador, datos in perfiles.items()`:

```python
    # Sovrascrittura da bio reale (tennisstats.com) — altezza e DOB sempre aggiornati
    if jugador in BIO_REAL:
        bio = BIO_REAL[jugador]
        if bio.get("dob"):
            perfiles[jugador]["age"]      = calcola_eta(bio["dob"])
            perfiles[jugador]["birth_year"] = int(bio["dob"][:4])
        if bio.get("ht"):
            perfiles[jugador]["ht"] = bio["ht"]
```

**Step 4: Verify**

```bash
cd scraping
python generar_perfiles.py
python -c "
import joblib
p = joblib.load('perfiles_jugadores.pkl')
for name in ['Novak Djokovic', 'Carlos Alcaraz', 'Jannik Sinner']:
    x = p.get(name, {})
    print(name, '→ age:', round(x.get('age',0),1), 'ht:', x.get('ht'))
"
```

Expected (approximate, depending on today's date):
```
Novak Djokovic → age: 38.8  ht: 188
Carlos Alcaraz → age: 22.8  ht: 183
Jannik Sinner  → age: 24.6  ht: 191
```

---

## Task 3: Update `enriquecer_2026.py` to use DOB from bio

**Files:**
- Modify: `scraping/enriquecer_2026.py`

Replace the static `p.get('age', 25) + 1.5` with real age from `birth_year`:

**Step 1:** At the top of the file, after the existing imports, add:

```python
from datetime import date as _date

def _eta_da_birth_year(birth_year):
    """Calcola l'età attuale da anno di nascita (intero)."""
    try:
        return _date.today().year - int(birth_year)
    except Exception:
        return 25
```

**Step 2:** Replace both occurrences of `p.get('age', 25) + 1.5` (lines ~78 and ~94):

*Old (winner):*
```python
        w_age.append(p.get('age', 25) + 1.5) # Sumamos 1.5 años porque es 2026
```

*New (winner):*
```python
        if 'birth_year' in p:
            w_age.append(_eta_da_birth_year(p['birth_year']))
        else:
            w_age.append(p.get('age', 25))  # fallback senza accumulo
```

*Old (loser):*
```python
        l_age.append(p.get('age', 25) + 1.5)
```

*New (loser):*
```python
        if 'birth_year' in p:
            l_age.append(_eta_da_birth_year(p['birth_year']))
        else:
            l_age.append(p.get('age', 25))  # fallback senza accumulo
```

---

## Task 4: Update `aggiorna_tutto.py`

**Files:**
- Modify: `aggiorna_tutto.py`

**Step 1: Add the `ESEGUI_BIO` flag** (after existing flags, ~line 43):

```python
ESEGUI_BIO         = False   # True = scraping bio (DOB + altezza) da tennisstats.com
```

**Step 2: Add FASE 5b before the existing FASE 5 generar_perfiles call** (after line 205):

```python
    # ══════════════════════════════════════════════════════════════════════════
    # FASE 5b — Bio giocatori (DOB + altezza da tennisstats.com)
    # ══════════════════════════════════════════════════════════════════════════
    if ESEGUI_BIO:
        sezione("5️⃣b  FASE 5b — Bio giocatori (tennisstats.com)")
        ok_bio = esegui("scraper_bio_jugadores.py", SCRAPING, "Bio giocatori (DOB + altezza)")
        if ok_bio:
            copia_se_esiste(
                os.path.join(SCRAPING, "bio_jugadores.json"),
                os.path.join(PREDICCION, "bio_jugadores.json")
            )
```

**Step 3: Also copy bio_jugadores.json inside FASE 5 block** (after the existing copy of perfiles_jugadores.pkl, ~line 210) to ensure prediccion always has the latest:

```python
            copia_se_esiste(
                os.path.join(SCRAPING, "bio_jugadores.json"),
                os.path.join(PREDICCION, "bio_jugadores.json")
            )
```

**Step 4: Update the docstring** at the top of the file to mention FASE 5b:

```
  FASE 5b: Bio giocatori (DOB + altezza da tennisstats.com) — solo se ESEGUI_BIO=True
```

---

## Task 5: Final end-to-end test

**Step 1:** Set `ESEGUI_BIO = True` in `aggiorna_tutto.py`

**Step 2:** Run:
```bash
python aggiorna_tutto.py
```

**Step 3:** Verify profiles:
```bash
cd scraping
python -c "
import joblib, json
p = joblib.load('perfiles_jugadores.pkl')
b = json.load(open('bio_jugadores.json'))
for name in ['Novak Djokovic', 'Carlos Alcaraz']:
    print(name, '→ profile age:', round(p[name]['age'],1), 'bio dob:', b.get(name,{}).get('dob'))
"
```

**Step 4:** Restore `ESEGUI_BIO = False` so subsequent auto-updates don't re-scrape each time.

---

## Notes

- If `parse_data_table` returns no data, tennisstats may have changed their HTML. Check with the manual curl test in Task 1 Step 2.
- `bio_jugadores.json` in `prediccion/` is also read by any future usage in the Streamlit app.
- The `generar_perfiles.py` fix from the previous debugging session (only reading age from pre-2026 rows) is still needed as a safety net and should remain in place.
