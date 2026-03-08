"""
scraper_bio_jugadores.py
Scrapes DOB and height for every player in ranking_2026.csv
from https://tennisstats.com/players/{slug}

OUTPUT: bio_jugadores.json  ->  {"Carlos Alcaraz": {"dob": "2003-05-05", "ht": 183}, ...}
"""
import os
import json, time, unicodedata, re
from datetime import datetime
import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://tennisstats.com/players/{slug}"
OUTPUT   = "bio_jugadores.json"
DELAY    = 0.6

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

def name_to_slug(name: str) -> str:
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = nfkd.encode("ascii", "ignore").decode("ascii")
    slug = ascii_name.lower().strip()
    slug = re.sub(r"[^a-z0-9\s-]", "", slug)
    slug = re.sub(r"\s+", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    return slug

def parse_data_table(soup):
    # Site uses .data-table-row divs with two .item child divs per row.
    # Label cell has additional class 'fitem'; value cell has class 'bold'.
    # Keys confirmed from live inspection: 'Birthday', 'Height'
    data = {}
    for row in soup.select(".data-table-row"):
        cells = row.select(".item")
        if len(cells) >= 2:
            label = cells[0].get_text(strip=True).lower()
            value = cells[1].get_text(strip=True)
            data[label] = value
    return data

def parse_dob(raw: str):
    # Site returns e.g. "May 5, 2003"
    for fmt in ["%B %d, %Y", "%d %B %Y", "%Y-%m-%d"]:
        try:
            return datetime.strptime(raw.strip(), fmt).strftime("%Y-%m-%d")
        except Exception:
            pass
    return None

def parse_height(raw: str):
    raw = raw.strip().lower()
    # Site returns e.g. "1.83m"
    m = re.match(r"^(\d+\.\d+)m$", raw)
    if m: return round(float(m.group(1)) * 100)
    # Fallback: "183cm" or "183 cm"
    m = re.match(r"^(\d+)\s*cm$", raw)
    if m: return int(m.group(1))
    return None

def scrape_player(name, session):
    slug = name_to_slug(name)
    url = BASE_URL.format(slug=slug)
    try:
        resp = session.get(url, timeout=10)
        if resp.status_code == 404:
            print(f"   warning  404 — {name} ({slug})")
            return None
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        table = parse_data_table(soup)
        # Label on site is 'Birthday', not 'Date of Birth'
        dob = parse_dob(table.get("birthday", ""))
        ht  = parse_height(table.get("height", ""))
        if not dob and not ht:
            print(f"   warning  Dati non trovati — {name} ({slug})")
            return None
        return {"dob": dob, "ht": ht}
    except Exception as e:
        print(f"   ERROR  {name}: {e}")
        return None


def main():
    try:
        df = pd.read_csv("estadisticas_jugadores_avanzadas.csv")
        players = df["player"].dropna().unique().tolist()
    except Exception as e:
        print(f"ERROR  Impossibile caricare estadisticas_jugadores_avanzadas.csv: {e}")
        return

    print(f"BIO SCRAPER — {len(players)} giocatori da scrapare")

    try:
        with open(OUTPUT, "r", encoding="utf-8") as f:
            bio = json.load(f)
        print(f"   Bio esistente caricata ({len(bio)} voci)")
    except FileNotFoundError:
        bio = {}

    session = requests.Session()
    session.headers.update(HEADERS)

    def save_bio(bio_dict):
        """Atomic write: write to .tmp then rename."""
        tmp = OUTPUT + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(bio_dict, f, ensure_ascii=False, indent=2)
        os.replace(tmp, OUTPUT)

    ok = errors = skipped = 0
    for i, name in enumerate(players, 1):
        if name in bio:
            skipped += 1
            continue
        print(f"   [{i}/{len(players)}] {name}", end=" ... ", flush=True)
        result = scrape_player(name, session)
        if result:
            bio[name] = result
            print(f"✅  DOB={result['dob']}  ht={result['ht']}cm")
            ok += 1
        else:
            errors += 1
        time.sleep(DELAY)
        # Save incrementally every 10 new entries
        if (ok + errors) % 10 == 0:
            save_bio(bio)

    save_bio(bio)
    print(f"\n✅  bio_jugadores.json salvato — {ok} OK, {errors} errori, {skipped} già cached")

if __name__ == "__main__":
    main()
