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
    "monte carlo": "Clay", "monte-carlo": "Clay", "madrid": "Clay", "rome": "Clay",
    "canada": "Hard", "montreal": "Hard", "toronto": "Hard",
    "cincinnati": "Hard", "shanghai": "Hard", "paris": "Hard",
    "rotterdam": "Hard", "dubai": "Hard", "acapulco": "Hard",
    "rio": "Clay", "buenos aires": "Clay", "santiago": "Clay",
    "barcelona": "Clay", "hamburg": "Clay", "munich": "Clay",
    "geneva": "Clay", "lyon": "Clay", "marrakech": "Clay",
    "estoril": "Clay", "bastad": "Clay", "kitzbuhel": "Clay",
    "umag": "Clay", "gstaad": "Clay", "cordoba": "Clay",
    "halle": "Grass", "queens": "Grass", "queen": "Grass",
    "stuttgart": "Grass", "mallorca": "Grass", "eastbourne": "Grass",
    "newport": "Grass", "hertogenbosch": "Grass",
    "washington": "Hard", "atlanta": "Hard", "los cabos": "Hard",
    "beijing": "Hard", "tokyo": "Hard", "vienna": "Hard",
    "basel": "Hard", "metz": "Hard", "chengdu": "Hard",
    "antwerp": "Hard", "stockholm": "Hard", "adelaide": "Hard",
    "brisbane": "Hard", "auckland": "Hard", "doha": "Hard",
    "montpellier": "Hard", "dallas": "Hard", "delray": "Hard",
    "winston-salem": "Hard", "zhuhai": "Hard", "florence": "Hard",
    "marseille": "Hard", "sofia": "Hard", "astana": "Hard",
    "belgrade": "Clay", "bucharest": "Clay",
}

ROUND_MAP = {
    "1R": "1° Turno", "2R": "2° Turno", "3R": "3° Turno", "4R": "Ottavi",
    "QF": "Quarti", "SF": "Semifinale", "F": "Finale",
    "RR": "Round Robin", "Q1": "Qualif. 1°", "Q2": "Qualif. 2°",
}

LIVELLO_MAP = {
    "australian open": "Grand Slam", "roland garros": "Grand Slam",
    "wimbledon": "Grand Slam", "us open": "Grand Slam",
    "indian wells": "Masters 1000", "miami": "Masters 1000",
    "monte carlo": "Masters 1000", "monte-carlo": "Masters 1000",
    "madrid": "Masters 1000", "rome": "Masters 1000",
    "canada": "Masters 1000", "montreal": "Masters 1000", "toronto": "Masters 1000",
    "cincinnati": "Masters 1000", "shanghai": "Masters 1000", "paris": "Masters 1000",
    "rotterdam": "ATP 500", "dubai": "ATP 500", "acapulco": "ATP 500",
    "rio": "ATP 500", "barcelona": "ATP 500", "halle": "ATP 500",
    "queens": "ATP 500", "hamburg": "ATP 500", "washington": "ATP 500",
    "beijing": "ATP 500", "tokyo": "ATP 500", "vienna": "ATP 500",
    "basel": "ATP 500",
}


def _superficie(name: str) -> str:
    n = name.lower()
    for key, surf in SUPERFICIE_MAP.items():
        if key in n:
            return surf
    return "Hard"


def _livello(name: str) -> str:
    n = name.lower()
    for key, liv in LIVELLO_MAP.items():
        if key in n:
            return liv
    if "challenger" in n:
        return "Challenger"
    return "ATP 250"


def _clean_name(raw: str) -> str:
    """Rimuove il seed "(N)" dal nome giocatore."""
    return re.sub(r'\(\d+\)\s*$', '', raw).strip()


def _is_atp(torneo_name: str) -> bool:
    """True se è un torneo ATP/Challenger (non ITF/Futures/WTA)."""
    t = torneo_name.lower()
    return not any(kw in t for kw in SKIP_KEYWORDS)


def _fetch_round_map(tournament_url: str) -> dict:
    """
    Visita la pagina del torneo e restituisce {nome_giocatore_pulito: turno_it}.
    tournament_url è relativo, es. '/monte-carlo/2026/atp-men/'
    """
    base = "https://www.tennisexplorer.com"
    url  = base + tournament_url
    result = {}
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
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
            round_it   = ROUND_MAP.get(round_code, round_code)
            # td con class "t-name" contiene il nome del giocatore
            td_name = row.find("td", class_="t-name")
            if td_name:
                name = _clean_name(td_name.get_text(strip=True))
                if name:
                    # Indicizza per cognome (prima parola) per matchare sia
                    # "Lastname" (pagina torneo) che "Lastname F." (pagina /next/)
                    result[name.split()[0].lower()] = round_it
    except Exception:
        pass
    return result


def fetch_upcoming() -> list:
    """Scarica le prossime partite ATP da tennisexplorer.com/next/."""
    url = "https://www.tennisexplorer.com/next/"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        if resp.status_code != 200:
            print(f"   ⚠️  TennisExplorer: HTTP {resp.status_code}")
            return []
    except Exception as e:
        print(f"   ⚠️  TennisExplorer: {e}")
        return []

    soup  = BeautifulSoup(resp.text, "html.parser")
    rows  = soup.select("tr")

    # ── Prima passata: raccogli URL torneo dalle righe "head" ──────────────────
    torneo_urls = {}   # torneo_name → url_relativo
    current_name = "Unknown"
    for row in rows:
        cls = " ".join(row.get("class", []))
        if "head" in cls:
            td = row.find("td", class_="t-name")
            if td:
                current_name = td.get_text(strip=True)
                a = td.find("a")
                if a and a.get("href"):
                    torneo_urls[current_name] = a["href"]

    # ── Pre-carica le round map per ogni torneo ATP ────────────────────────────
    round_maps = {}   # torneo_name → {player → turno_it}
    for tname, turl in torneo_urls.items():
        if _is_atp(tname):
            print(f"   🔍  Round info: {tname}")
            round_maps[tname] = _fetch_round_map(turl)
            time.sleep(0.5)

    # ── Seconda passata: costruisci le partite ────────────────────────────────
    partite  = []
    torneo   = "Unknown"
    i = 0

    while i < len(rows):
        row = rows[i]
        cls = " ".join(row.get("class", []))

        # ── Riga torneo (header) ──────────────────────────────────────────────
        if "head" in cls:
            td = row.find("td", class_="t-name")
            if td:
                torneo = td.get_text(strip=True)
            i += 1
            continue

        # ── Prima riga di un match (contiene "bott") ──────────────────────────
        if "bott" in cls:
            tds = row.find_all("td")
            if len(tds) < 2:
                i += 1
                continue

            # Cerca il td con il nome del giocatore (class "t-name")
            td_name = row.find("td", class_="t-name")
            if not td_name:
                i += 1
                continue
            p1_raw = td_name.get_text(strip=True)

            # Controlla se è upcoming: td con class "nbr" (senza risultato)
            # vs "result" (con risultato = già giocato)
            td_result = row.find("td", class_=re.compile(r'\bnbr\b|\bresult\b'))
            is_upcoming = td_result is not None and "nbr" in (td_result.get("class") or [])

            # Guarda la riga successiva per p2
            p2_raw = ""
            if i + 1 < len(rows):
                next_row = rows[i + 1]
                next_cls = " ".join(next_row.get("class", []))
                # La riga di p2 NON ha "bott" e NON ha "head"
                if "bott" not in next_cls and "head" not in next_cls:
                    td2 = next_row.find("td", class_="t-name")
                    if td2:
                        p2_raw = td2.get_text(strip=True)
                    i += 1  # consuma anche la riga di p2

            p1 = _clean_name(p1_raw)
            p2 = _clean_name(p2_raw)

            if p1 and p2 and p1 != p2 and _is_atp(torneo) and is_upcoming:
                # Round: cerca per cognome (prima parola, lowercase)
                rmap  = round_maps.get(torneo, {})
                turno = (rmap.get(p1.split()[0].lower())
                         or rmap.get(p2.split()[0].lower())
                         or "N/D")

                partite.append({
                    "p1":         p1,
                    "p2":         p2,
                    "torneo":     torneo,
                    "livello":    _livello(torneo),
                    "superficie": _superficie(torneo),
                    "turno":      turno,
                    "data":       datetime.date.today().strftime("%Y-%m-%d"),
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
