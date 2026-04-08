# scraping/scraper_proximas_partidas.py
"""
Scarica le partite ATP upcoming (prossimi 7 giorni) da SofaScore API.
Output: scraping/proximas_partidas.json
"""

import json
import os
import time
import datetime
import requests

SCRAPING = os.path.dirname(os.path.abspath(__file__))
OUTPUT   = os.path.join(SCRAPING, "proximas_partidas.json")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Referer": "https://www.sofascore.com/",
}

# ── Mapping torneo → superficie ───────────────────────────────────────────────
SUPERFICIE_MAP = {
    "Australian Open": "Hard", "Roland Garros": "Clay", "Wimbledon": "Grass",
    "US Open": "Hard", "Indian Wells": "Hard", "Miami": "Hard",
    "Monte Carlo": "Clay", "Madrid": "Clay", "Rome": "Clay",
    "Canada": "Hard", "Montreal": "Hard", "Toronto": "Hard",
    "Cincinnati": "Hard", "Shanghai": "Hard", "Paris": "Hard",
    "Rotterdam": "Hard", "Dubai": "Hard", "Acapulco": "Hard",
    "Rio": "Clay", "Buenos Aires": "Clay", "Santiago": "Clay",
    "Barcelona": "Clay", "Hamburg": "Clay", "Munich": "Clay",
    "Geneva": "Clay", "Lyon": "Clay", "Marrakech": "Clay",
    "Estoril": "Clay", "Bastad": "Clay", "Kitzbuhel": "Clay",
    "Umag": "Clay", "Gstaad": "Clay", "Cordova": "Clay",
    "Halle": "Grass", "Queens": "Grass",
    "Stuttgart": "Grass", "Mallorca": "Grass", "Eastbourne": "Grass",
    "Newport": "Grass", "Hertogenbosch": "Grass",
    "Washington": "Hard", "Atlanta": "Hard", "Los Cabos": "Hard",
    "Beijing": "Hard", "Tokyo": "Hard", "Vienna": "Hard",
    "Basel": "Hard", "Metz": "Hard", "Chengdu": "Hard",
    "Antwerp": "Hard", "Stockholm": "Hard", "Adelaide": "Hard",
    "Brisbane": "Hard", "Auckland": "Hard", "Doha": "Hard",
    "Montpellier": "Hard", "Dallas": "Hard", "Delray Beach": "Hard",
    "Winston-Salem": "Hard", "Zhuhai": "Hard", "Florence": "Hard",
}

# ── Mapping torneo → livello ──────────────────────────────────────────────────
LIVELLO_MAP = {
    "Australian Open": "Grand Slam", "Roland Garros": "Grand Slam",
    "Wimbledon": "Grand Slam", "US Open": "Grand Slam",
    "Indian Wells": "Masters 1000", "Miami": "Masters 1000",
    "Monte Carlo": "Masters 1000", "Madrid": "Masters 1000",
    "Rome": "Masters 1000", "Canada": "Masters 1000",
    "Montreal": "Masters 1000", "Toronto": "Masters 1000",
    "Cincinnati": "Masters 1000", "Shanghai": "Masters 1000",
    "Paris": "Masters 1000",
    "Rotterdam": "ATP 500", "Dubai": "ATP 500", "Acapulco": "ATP 500",
    "Rio": "ATP 500", "Barcelona": "ATP 500", "Halle": "ATP 500",
    "Queens": "ATP 500", "Hamburg": "ATP 500", "Washington": "ATP 500",
    "Beijing": "ATP 500", "Tokyo": "ATP 500", "Vienna": "ATP 500",
    "Basel": "ATP 500",
}

# ── Mapping round SofaScore → label italiano ─────────────────────────────────
ROUND_MAP = {
    "Final": "Finale", "Finals": "Finale",
    "Semifinal": "Semifinale", "Semifinals": "Semifinale", "Semi-finals": "Semifinale",
    "Quarterfinal": "Quarti", "Quarterfinals": "Quarti", "Quarter-finals": "Quarti",
    "Round of 16": "Ottavi di finale (16mi)",
    "Round of 32": "32mi", "Round of 64": "64mi", "Round of 128": "128mi",
    "Round Robin": "Round Robin",
    "1st Round": "128mi", "2nd Round": "64mi", "3rd Round": "32mi",
    "4th Round": "Ottavi di finale (16mi)",
}


def _superficie(torneo_name: str) -> str:
    for key, surf in SUPERFICIE_MAP.items():
        if key.lower() in torneo_name.lower():
            return surf
    return "Hard"


def _livello(torneo_name: str, categoria: str) -> str:
    for key, liv in LIVELLO_MAP.items():
        if key.lower() in torneo_name.lower():
            return liv
    cat_lower = categoria.lower()
    if "challenger" in cat_lower:
        return "Challenger"
    return "ATP 250"


def _round_ita(round_name: str) -> str:
    return ROUND_MAP.get(round_name, round_name or "N/D")


def fetch_day(date_str: str) -> list:
    url = f"https://api.sofascore.com/api/v1/sport/tennis/scheduled-events/{date_str}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            print(f"   ⚠️  SofaScore {date_str}: HTTP {resp.status_code}")
            return []
        data = resp.json()
    except Exception as e:
        print(f"   ⚠️  SofaScore {date_str}: {e}")
        return []

    partite = []
    for ev in data.get("events", []):
        try:
            if ev.get("status", {}).get("type") != "notstarted":
                continue

            torneo    = ev.get("tournament", {})
            cat_name  = torneo.get("category", {}).get("name", "")

            if not any(x in cat_name.upper() for x in ["ATP", "CHALLENGER"]):
                continue

            torneo_name = torneo.get("uniqueTournament", {}).get("name") \
                          or torneo.get("name", "Unknown")
            p1 = ev.get("homeTeam", {}).get("name", "")
            p2 = ev.get("awayTeam", {}).get("name", "")
            if not p1 or not p2:
                continue

            round_name = ev.get("roundInfo", {}).get("name", "")
            ts = ev.get("startTimestamp", 0)
            data_partita = datetime.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d") \
                           if ts else date_str

            partite.append({
                "p1":         p1,
                "p2":         p2,
                "torneo":     torneo_name,
                "livello":    _livello(torneo_name, cat_name),
                "superficie": _superficie(torneo_name),
                "turno":      _round_ita(round_name),
                "data":       data_partita,
            })
        except Exception:
            continue

    return partite


def main():
    print("🌐  Scraping prossime partite ATP (SofaScore)...")
    today = datetime.date.today()
    tutte = []
    viste = set()

    for delta in range(8):
        day = today + datetime.timedelta(days=delta)
        day_str = day.strftime("%Y-%m-%d")
        print(f"   📅  {day_str}...")
        giornata = fetch_day(day_str)
        for p in giornata:
            key = (p["p1"], p["p2"], p["torneo"], p["turno"])
            if key not in viste:
                viste.add(key)
                tutte.append(p)
        time.sleep(0.5)

    tutte.sort(key=lambda x: (x["data"], x["torneo"], x["turno"]))
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(tutte, f, ensure_ascii=False, indent=2)

    print(f"   ✅  {len(tutte)} partite salvate → {OUTPUT}")


if __name__ == "__main__":
    main()
