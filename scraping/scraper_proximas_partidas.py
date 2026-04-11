# scraping/scraper_proximas_partidas.py
"""
Scarica le partite ATP/Challenger upcoming da tennisexplorer.com/next/.
Output: scraping/proximas_partidas.json

Struttura HTML TennisExplorer:
  - row class "head flags"   → nome torneo (td class "t-name")
  - row con "bott" in class  → prima riga di un match (p1, td class "t-name")
  - riga successiva          → seconda riga dello stesso match (p2)
  - Upcoming: td[2] class "nbr" | Completato: td[2] class "result" con testo
"""

import json
import os
import re
import sys
import time
import datetime
import requests
from bs4 import BeautifulSoup

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SCRAPING = os.path.dirname(os.path.abspath(__file__))
OUTPUT   = os.path.join(SCRAPING, "proximas_partidas.json")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}

# Tornei da saltare (non ATP, non Challenger)
SKIP_KEYWORDS = ["utr", "itf", "futures", "wta", "fed cup", "davis", "billie jean",
                 "wheelchair", "junior", "boys", "girls", "doubles"]

# ── Mapping torneo → superficie ───────────────────────────────────────────────
SUPERFICIE_MAP = {
    "australian open": "Hard", "roland garros": "Clay", "wimbledon": "Grass",
    "us open": "Hard", "indian wells": "Hard", "miami": "Hard",
    "monte carlo": "Clay", "monte-carlo": "Clay", "rome": "Clay",
    "canada": "Hard", "montreal": "Hard", "toronto": "Hard",
    "cincinnati": "Hard", "shanghai": "Hard", "paris": "Hard",
    "rotterdam": "Hard", "dubai": "Hard", "acapulco": "Hard",
    "rio": "Clay", "buenos aires": "Clay", "santiago": "Clay",
    "barcelona": "Clay", "hamburg": "Clay", "munich": "Clay",
    "geneva": "Clay", "lyon": "Clay", "marrakech": "Clay",
    "estoril": "Clay", "bastad": "Clay", "kitzbuhel": "Clay",
    "umag": "Clay", "gstaad": "Clay", "cordoba": "Clay",
    "florence": "Clay", "bucharest": "Clay", "belgrade": "Clay",
    "halle": "Grass", "queens": "Grass", "queen": "Grass",
    "stuttgart": "Grass", "mallorca": "Grass", "eastbourne": "Grass",
    "newport": "Grass", "hertogenbosch": "Grass",
    "washington": "Hard", "atlanta": "Hard", "los cabos": "Hard",
    "beijing": "Hard", "tokyo": "Hard", "vienna": "Hard",
    "basel": "Hard", "metz": "Hard", "chengdu": "Hard",
    "antwerp": "Hard", "stockholm": "Hard", "adelaide": "Hard",
    "brisbane": "Hard", "auckland": "Hard", "doha": "Hard",
    "montpellier": "Hard", "dallas": "Hard", "delray": "Hard",
    "winston-salem": "Hard", "zhuhai": "Hard",
    "marseille": "Hard", "sofia": "Hard", "astana": "Hard",
    "sarasota": "Hard", "wuning": "Hard",
    # Challenger city → surface (aggiunte per challenger comuni)
    "campinas": "Clay", "buenos": "Clay", "monza": "Clay",
    "mexico city": "Hard",
}

# ── Dimensione draw per torneo (usata per mappare 1R/2R al turno corretto) ────
# Masters 1000 hanno draw da 96 (i top seeds entrano al 2° turno)
# → il "1R" reale è la partita tra le posizioni 33-96 = Ottavi di finale (16esimi)
DRAW_SIZE_MAP = {
    "grand slam":    128,   # 128 draw — 1R=128mi, 2R=64mi, 3R=32mi, 4R=16mi
    "masters 1000":   96,   # 96 draw  — 1R=Ottavi (16mi), 2R=Quarti... seeds entrano al 2R
    "atp 500":        48,   # 48/32 draw — 1R=32mi, 2R=16mi, 3R=Quarti
    "atp 250":        32,   # 32 draw — 1R=32mi, 2R=16mi
    "challenger":     32,   # 32 draw — 1R=32mi
}

# Round map per dimensione draw: DRAW_SIZE → {codice_round → turno_canonico}
ROUND_MAP_BY_DRAW = {
    128: {"1R": "128mi",  "2R": "64mi", "3R": "32mi",
          "4R": "Ottavi di finale (16mi)", "QF": "Quarti",
          "SF": "Semifinale", "F": "Finale", "RR": "Round Robin"},
    96:  {"1R": "Ottavi di finale (16mi)", "2R": "Quarti",
          "SF": "Semifinale", "F": "Finale", "RR": "Round Robin"},
    48:  {"1R": "32mi", "2R": "Ottavi di finale (16mi)",
          "QF": "Quarti", "SF": "Semifinale", "F": "Finale"},
    32:  {"1R": "32mi", "2R": "Ottavi di finale (16mi)",
          "QF": "Quarti", "SF": "Semifinale", "F": "Finale"},
}

ROUND_MAP = {
    # Codici ATP → chiavi CANONICHE di ROUND_MAP_STR (prediction_engine.py)
    # Questi sono i default se non conosciamo il draw size del torneo
    "1R": "64mi",
    "2R": "32mi",
    "3R": "Ottavi di finale (16mi)",
    "4R": "Quarti",
    "QF": "Quarti",
    "SF": "Semifinale",
    "F":  "Finale",
    "RR": "Round Robin",
    "Q1": "128mi",
    "Q2": "64mi",
}

LIVELLO_MAP = {
    "australian open": "Grand Slam", "roland garros": "Grand Slam",
    "wimbledon": "Grand Slam", "us open": "Grand Slam",
    "indian wells": "Masters 1000", "miami": "Masters 1000",
    "monte carlo": "Masters 1000", "monte-carlo": "Masters 1000",
    "rome": "Masters 1000",
    "canada": "Masters 1000", "montreal": "Masters 1000", "toronto": "Masters 1000",
    "cincinnati": "Masters 1000", "shanghai": "Masters 1000", "paris": "Masters 1000",
    "rotterdam": "ATP 500", "dubai": "ATP 500", "acapulco": "ATP 500",
    "rio": "ATP 500", "barcelona": "ATP 500", "halle": "ATP 500",
    "queens": "ATP 500", "hamburg": "ATP 500", "washington": "ATP 500",
    "beijing": "ATP 500", "tokyo": "ATP 500", "vienna": "ATP 500",
    "basel": "ATP 500",
}

# Madrid è sia Masters 1000 che Challenger — deve essere controllato per primo
# nel contesto del nome torneo completo.
MADRID_MASTERS_KEYWORDS = ["madrid masters", "mutua madrid", "madrid open"]


def _superficie(name: str) -> str:
    n = name.lower()
    for key, surf in SUPERFICIE_MAP.items():
        if key in n:
            return surf
    # Default challenger → Clay (la maggior parte è su terra)
    if "challenger" in n:
        return "Clay"
    return "Hard"


def _livello(name: str) -> str:
    n = name.lower()
    # ⚠️ Controlla PRIMA se è un challenger (evita falsi match su "madrid", ecc.)
    if "challenger" in n:
        return "Challenger"
    # Poi controlla madrid masters specificamente
    if "madrid" in n:
        return "Masters 1000"
    for key, liv in LIVELLO_MAP.items():
        if key in n:
            return liv
    return "ATP 250"


def _draw_size(livello: str) -> int:
    """Restituisce la dimensione standard del tabellone per livello torneo."""
    return DRAW_SIZE_MAP.get(livello.lower(), 64)


def _map_round(raw_code: str, livello: str) -> str:
    """Mappa il codice round (es. '1R') al turno canonico in base al draw size."""
    draw = _draw_size(livello)
    rmap = ROUND_MAP_BY_DRAW.get(draw, ROUND_MAP)
    return rmap.get(raw_code, ROUND_MAP.get(raw_code, "N/D"))



def _clean_name(raw: str) -> str:
    """Rimuove il seed "(N)" dal nome giocatore."""
    return re.sub(r'\(\d+\)\s*$', '', raw).strip()


def _is_atp(torneo_name: str) -> bool:
    """True se è un torneo ATP/Challenger (non ITF/Futures/WTA)."""
    t = torneo_name.lower()
    return not any(kw in t for kw in SKIP_KEYWORDS)


def _fetch_round_map(tournament_url: str, livello: str) -> dict:
    """
    Visita la pagina del torneo e restituisce {cognome_giocatore: turno_canonico}.
    Usa _map_round() con il livello torneo per il mapping corretto del draw size.
    """
    base = "https://www.tennisexplorer.com"
    url  = base + tournament_url
    result = {}
    try:
        resp = requests.get(url, headers=HEADERS, timeout=5)
        if resp.status_code != 200:
            return result
        soup = BeautifulSoup(resp.text, "html.parser")
        for row in soup.select("tr"):
            cls = " ".join(row.get("class", []))
            if "bott" not in cls:
                continue
            tds = row.find_all("td")
            if len(tds) < 3:
                continue
            # td[1] contiene il codice round (es. "2R", "QF")
            round_code = tds[1].get_text(strip=True)
            round_it   = _map_round(round_code, livello)  # draw-size aware!
            # td con class "t-name" contiene il nome del giocatore
            td_name = row.find("td", class_="t-name")
            if td_name:
                name = _clean_name(td_name.get_text(strip=True))
                if name:
                    result[name.split()[0].lower()] = round_it
    except Exception:
        pass
    return result


def _fetch_soup(url: str) -> BeautifulSoup | None:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        if resp.status_code != 200:
            return None
        return BeautifulSoup(resp.text, "html.parser")
    except Exception:
        return None


def fetch_upcoming() -> list:
    """Scarica le prossime partite ATP da tennisexplorer.com/next/ per oggi + 3 giorni."""
    today = datetime.date.today()
    soups = []
    for delta in range(4):
        d = today + datetime.timedelta(days=delta)
        url = (f"https://www.tennisexplorer.com/next/"
               f"?type=atp-single&year={d.year}&month={d.month:02d}&day={d.day:02d}")
        print(f"   📅  Scarico {d}...", flush=True)
        s = _fetch_soup(url)
        if s:
            soups.append((d, s))
        time.sleep(0.5)

    if not soups:
        s = _fetch_soup("https://www.tennisexplorer.com/next/")
        if s:
            soups = [(today, s)]

    if not soups:
        print("   ⚠️  TennisExplorer: nessuna risposta valida")
        return []

    # ── Raccogli URL torneo da tutti i soup ───────────────────────────────────
    torneo_urls = {}
    for _d, soup in soups:
        for row in soup.select("tr"):
            cls = " ".join(row.get("class", []))
            if "head" in cls:
                td = row.find("td", class_="t-name")
                if td:
                    name = td.get_text(strip=True)
                    a = td.find("a")
                    if a and a.get("href") and name not in torneo_urls:
                        torneo_urls[name] = a["href"]

    # ── Pre-carica le round map (una richiesta per torneo ATP) ────────────────
    round_maps = {}
    for tname, turl in torneo_urls.items():
        if _is_atp(tname):
            lv = _livello(tname)
            print(f"   🔍  Round info: {tname} [{lv}]", flush=True)
            round_maps[tname] = _fetch_round_map(turl, lv)
            print(f"       → {len(round_maps[tname])} giocatori mappati", flush=True)
            time.sleep(0.3)

    # ── Costruisci partite con deduplicazione ─────────────────────────────────
    seen    = set()
    partite = []

    for match_date, soup in soups:
        rows   = soup.select("tr")
        torneo = "Unknown"
        i = 0
        while i < len(rows):
            row = rows[i]
            cls = " ".join(row.get("class", []))

            if "head" in cls:
                td = row.find("td", class_="t-name")
                if td:
                    torneo = td.get_text(strip=True)
                i += 1
                continue

            if "bott" in cls:
                tds = row.find_all("td")
                if len(tds) < 2:
                    i += 1
                    continue
                td_name = row.find("td", class_="t-name")
                if not td_name:
                    i += 1
                    continue
                p1_raw = td_name.get_text(strip=True)

                td_result = row.find("td", class_=re.compile(r'\bnbr\b|\bresult\b'))
                is_upcoming = td_result is not None and "nbr" in (td_result.get("class") or [])

                p2_raw = ""
                if i + 1 < len(rows):
                    next_row = rows[i + 1]
                    next_cls = " ".join(next_row.get("class", []))
                    if "bott" not in next_cls and "head" not in next_cls:
                        td2 = next_row.find("td", class_="t-name")
                        if td2:
                            p2_raw = td2.get_text(strip=True)
                        i += 1

                p1 = _clean_name(p1_raw)
                p2 = _clean_name(p2_raw)

                if p1 and p2 and p1 != p2 and _is_atp(torneo) and is_upcoming:
                    key = (min(p1, p2), max(p1, p2), torneo.lower())
                    if key not in seen:
                        seen.add(key)
                        lv   = _livello(torneo)
                        surf = _superficie(torneo)
                        rmap = round_maps.get(torneo, {})
                        r_code = (rmap.get(p1.split()[0].lower())
                                  or rmap.get(p2.split()[0].lower()))
                        turno = r_code if r_code else _map_round("1R", lv)
                        partite.append({
                            "p1":         p1,
                            "p2":         p2,
                            "torneo":     torneo,
                            "livello":    lv,
                            "superficie": surf,
                            "turno":      turno,
                            "data":       match_date.strftime("%Y-%m-%d"),
                        })

            i += 1

    return partite


def main():
    print("🌐  Scraping prossime partite ATP (TennisExplorer)...")

    partite = fetch_upcoming()

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(partite, f, ensure_ascii=False, indent=2)

    print(f"   ✅  {len(partite)} partite salvate → {OUTPUT}")


if __name__ == "__main__":
    main()
