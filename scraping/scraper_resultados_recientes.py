"""
scraping/scraper_resultados_recientes.py
=========================================
Scarica da tennisexplorer.com i risultati ATP dei giorni mancanti tra
l'ultima data in master_dataset.csv e ieri, e li appende al CSV.

Struttura HTML tennisexplorer results:
  - tr.head.flags        → header torneo (td.t-name = nome torneo, <a href> = link torneo)
  - tr.*.bott            → PRIMA riga match = VINCITORE
      td.t-name          → nome vincitore
      td.result          → set vinti dal vincitore (es. "3")
      td.score (lista)   → punteggio ogni set vincitore (es. "6","7","6")
  - tr.* (senza bott)    → SECONDA riga match = PERDENTE
      td.t-name          → nome perdente
      td.result          → set vinti dal perdente
      td.score (lista)   → punteggio ogni set perdente

Uso standalone: python scraper_resultados_recientes.py
Integrato in run_pipeline.py come FASE 2b (dopo download TML).
"""

import os
import re
import sys
import time
import datetime
import requests
import pandas as pd
from bs4 import BeautifulSoup

_NAME_PARTICLES = frozenset({
    'de', 'del', 'van', 'der', 'von', 'da', 'di', 'dos', 'das',
    'las', 'los', 'el', 'le', 'la', 'du', 'des', 'den', 'ter', 'ten', 'al', 'y',
})

def normalize_player_name(name):
    """Forma canonica: prima lettera maiuscola, particelle nobiliari minuscole."""
    if not name or not isinstance(name, str):
        return name
    parts = ' '.join(name.strip().split()).split()
    return ' '.join(
        p.lower() if (i > 0 and p.lower() in _NAME_PARTICLES) else p.capitalize()
        for i, p in enumerate(parts)
    )

# ── Path ──────────────────────────────────────────────────────────────────────

SCRAPING_DIR = os.path.dirname(os.path.abspath(__file__))
MASTER_CSV   = os.path.join(SCRAPING_DIR, "master_dataset.csv")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

BASE_RESULTS = "https://www.tennisexplorer.com/results/?type=atp-single"
BASE_TOUR    = "https://www.tennisexplorer.com"

# ── Mapping tornei (identico a scraper_proximas_partidas) ─────────────────────

SUPERFICIE_MAP = {
    "australian open": "Hard", "roland garros": "Clay", "french open": "Clay",
    "wimbledon": "Grass", "us open": "Hard",
    "indian wells": "Hard", "miami": "Hard",
    "monte carlo": "Clay", "monte-carlo": "Clay", "madrid": "Clay",
    "rome": "Clay",
    "canada": "Hard", "montreal": "Hard", "toronto": "Hard",
    "cincinnati": "Hard", "shanghai": "Hard", "paris": "Hard",
    "rotterdam": "Hard", "dubai": "Hard", "acapulco": "Hard",
    "rio": "Clay", "buenos aires": "Clay", "santiago": "Clay",
    "barcelona": "Clay", "hamburg": "Clay", "munich": "Clay",
    "geneva": "Clay", "lyon": "Clay", "marrakech": "Clay",
    "florence": "Clay", "bucharest": "Clay", "belgrade": "Clay",
    "halle": "Grass", "queens": "Grass", "queen": "Grass",
    "stuttgart": "Grass", "mallorca": "Grass", "eastbourne": "Grass",
    "newport": "Grass", "hertogenbosch": "Grass",
    "washington": "Hard", "beijing": "Hard", "tokyo": "Hard",
    "vienna": "Hard", "basel": "Hard", "metz": "Hard",
    "antwerp": "Hard", "stockholm": "Hard", "adelaide": "Hard",
    "brisbane": "Hard", "auckland": "Hard", "doha": "Hard",
    "montpellier": "Hard", "dallas": "Hard", "delray": "Hard",
    "marseille": "Hard", "sofia": "Hard", "astana": "Hard",
    "sarasota": "Hard",
}

LIVELLO_MAP = {
    "australian open": "Grand Slam", "roland garros": "Grand Slam",
    "french open": "Grand Slam", "wimbledon": "Grand Slam", "us open": "Grand Slam",
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

SKIP_KEYWORDS = [
    "wta", "itf", "futures", "junior", "wheelchair",
    "double", "doppel", "mixed", "fed cup", "davis",
    "billie", "united cup", "hopman",
]

DRAW_SIZE_MAP = {
    "Grand Slam": 128, "Masters 1000": 96, "ATP 500": 48,
    "ATP 250": 32, "Challenger": 32,
}

# Round map per draw size (codici tennisexplorer → formato TML)
ROUND_MAP_BY_DRAW = {
    128: {"1R": "R128", "2R": "R64", "3R": "R32", "4R": "R16",
          "QF": "QF",   "SF": "SF",  "F":  "F"},
    96:  {"1R": "R128", "2R": "R64", "3R": "R32", "QF": "QF",
          "SF": "SF",   "F":  "F"},
    48:  {"1R": "R64",  "2R": "R32", "QF": "QF",  "SF": "SF", "F": "F"},
    32:  {"1R": "R32",  "2R": "R16", "QF": "QF",  "SF": "SF", "F": "F"},
}
ROUND_MAP_FALLBACK = {
    "QF": "QF", "SF": "SF", "F": "F", "RR": "RR",
    "1R": "R64", "2R": "R32", "3R": "R16", "4R": "QF",
}


# ── Helper ────────────────────────────────────────────────────────────────────

def _superficie(name: str) -> str:
    n = name.lower()
    for key, surf in SUPERFICIE_MAP.items():
        if key in n:
            return surf
    return "Clay" if "challenger" in n else "Hard"


def _livello(name: str) -> str:
    n = name.lower()
    if "challenger" in n:
        return "Challenger"
    if "madrid" in n and "challenger" not in n:
        return "Masters 1000"
    for key, lv in LIVELLO_MAP.items():
        if key in n:
            return lv
    return "ATP 250"


def _is_atp(name: str) -> bool:
    n = name.lower()
    return not any(kw in n for kw in SKIP_KEYWORDS)


def _clean_name(raw: str) -> str:
    return re.sub(r'\s*\(\d+\)\s*$', '', raw).strip()


def _draw_size(livello: str) -> int:
    return DRAW_SIZE_MAP.get(livello, 32)


def _map_round(code: str, livello: str) -> str:
    draw = _draw_size(livello)
    rmap = ROUND_MAP_BY_DRAW.get(draw, ROUND_MAP_FALLBACK)
    return rmap.get(code.upper(), ROUND_MAP_FALLBACK.get(code.upper(), "R32"))


def _fetch_soup(url: str) -> BeautifulSoup | None:
    try:
        r = requests.get(url, headers=HEADERS, timeout=12)
        if r.status_code == 200:
            return BeautifulSoup(r.text, "html.parser")
    except Exception:
        pass
    return None


def _fetch_round_map(tourney_url: str, livello: str) -> dict:
    """Visita la pagina del torneo → {cognome_lowercase: round_code}."""
    url = BASE_TOUR + tourney_url
    result = {}
    try:
        soup = _fetch_soup(url)
        if not soup:
            return result
        for row in soup.select("tr"):
            tds = row.find_all("td")
            if len(tds) < 2:
                continue
            round_code = tds[1].get_text(strip=True)
            td_name    = row.find("td", class_="t-name")
            if td_name:
                name = _clean_name(td_name.get_text(strip=True))
                if name:
                    result[name.split()[0].lower()] = _map_round(round_code, livello)
    except Exception:
        pass
    return result


def _build_name_index(df: pd.DataFrame) -> dict:
    """
    Costruisce un indice {cognome_iniziale: nome_completo} dai nomi TML.
    Esempio: "sinner_j" → "Jannik Sinner"
    """
    idx = {}
    for col in ["winner_name", "loser_name"]:
        for name in df[col].dropna().unique():
            parts = str(name).strip().split()
            if len(parts) >= 2:
                surname  = parts[-1].lower()
                initial  = parts[0][0].lower()
                key      = f"{surname}_{initial}"
                if key not in idx:
                    idx[key] = name
                # Anche solo cognome (per casi ambigui)
                if surname not in idx:
                    idx[surname] = name
    return idx


def _resolve_name(abbreviated: str, name_index: dict) -> str:
    """
    Converte "Cognome I." → nome completo TML.
    Gestisce cognomi composti: "De Minaur A." → "Alex de Minaur"
    L'iniziale è sempre l'ultimo token; prova ogni parola precedente come cognome.
    """
    s = abbreviated.strip()
    parts = s.split()
    if len(parts) < 2:
        return s
    # L'ultima parte è l'iniziale del nome (es. "A." → "a")
    initial = parts[-1].rstrip('.').lower()
    if not initial:
        return s
    # Prova ogni parola del cognome (es. "De Minaur A." → prova "de_a", poi "minaur_a")
    for word in parts[:-1]:
        key = f"{word.lower()}_{initial}"
        full = name_index.get(key)
        if full:
            return full
    # Fallback: solo l'ultima parola del cognome senza iniziale
    full = name_index.get(parts[-2].lower())
    if full:
        return full
    return s  # fallback: nome abbreviato originale


def _build_score(w_sets: list, l_sets: list) -> str:
    """Costruisce una stringa score tipo '6-4 7-5 6-3' dai set del vincitore e perdente."""
    parts = []
    for i in range(max(len(w_sets), len(l_sets))):
        ws = w_sets[i] if i < len(w_sets) else "0"
        ls = l_sets[i] if i < len(l_sets) else "0"
        # Tiebreak format: "62" = 6(2), "70" = 7(0) → semplifica a "6" e "7"
        ws_n = ws[0] if len(ws) > 1 and ws[0].isdigit() else ws
        ls_n = ls[0] if len(ls) > 1 and ls[0].isdigit() else ls
        if ws_n.isdigit() and ls_n.isdigit():
            parts.append(f"{ws_n}-{ls_n}")
    return " ".join(parts)


# ── Scraping giornaliero ──────────────────────────────────────────────────────

def scrape_day(date: datetime.date) -> list[dict]:
    """
    Scarica i risultati ATP completati per un singolo giorno.
    Ritorna lista di dict compatibili con master_dataset.csv.
    """
    url  = f"{BASE_RESULTS}&year={date.year}&month={date.month:02d}&day={date.day:02d}"
    soup = _fetch_soup(url)
    if soup is None:
        return []

    rows = soup.select("tr")

    # ── Pre-carica round map per ogni torneo ATP ──────────────────────────────
    torneo_urls = {}
    for row in rows:
        cls_list = row.get("class", [])
        if "head" in cls_list:
            td = row.find("td", class_="t-name")
            if td:
                name = td.get_text(strip=True)
                a    = td.find("a")
                if a and a.get("href") and name not in torneo_urls and _is_atp(name):
                    torneo_urls[name] = a["href"]

    round_maps = {}
    for tname, turl in torneo_urls.items():
        lv = _livello(tname)
        round_maps[tname] = _fetch_round_map(turl, lv)
        time.sleep(0.3)

    # ── Parsing righe match ───────────────────────────────────────────────────
    out    = []
    torneo = "Unknown"
    mc     = 0
    i      = 0

    while i < len(rows):
        row      = rows[i]
        cls_list = row.get("class", [])

        # Header torneo
        if "head" in cls_list:
            td = row.find("td", class_="t-name")
            if td:
                torneo = td.get_text(strip=True)
            i += 1
            continue

        # Riga vincitore (ha "bott" nella classe)
        if "bott" in cls_list:
            td_name = row.find("td", class_="t-name")
            if not td_name:
                i += 1
                continue

            winner_raw = td_name.get_text(strip=True)

            # Set scores vincitore
            w_score_tds = row.find_all("td", class_="score")
            w_sets = [t.get_text(strip=True) for t in w_score_tds
                      if t.get_text(strip=True)]

            # Sets vinti vincitore (per validare match completato)
            td_res = row.find("td", class_="result")
            sets_w = td_res.get_text(strip=True) if td_res else ""

            # Riga perdente (riga successiva, senza "bott" e senza "head")
            loser_raw = ""
            l_sets    = []
            if i + 1 < len(rows):
                nxt      = rows[i + 1]
                nxt_cls  = nxt.get("class", [])
                if "bott" not in nxt_cls and "head" not in nxt_cls:
                    td2 = nxt.find("td", class_="t-name")
                    if td2:
                        loser_raw = td2.get_text(strip=True)
                    l_score_tds = nxt.find_all("td", class_="score")
                    l_sets = [t.get_text(strip=True) for t in l_score_tds
                               if t.get_text(strip=True)]
                    i += 1  # consuma la riga del perdente

            winner = _clean_name(winner_raw)
            loser  = _clean_name(loser_raw)

            # Validazioni
            if not winner or not loser or winner == loser:
                i += 1
                continue
            if not _is_atp(torneo):
                i += 1
                continue
            if not sets_w.isdigit() or int(sets_w) < 1:
                i += 1
                continue  # match non completato o walkover

            # Score string
            score = _build_score(w_sets, l_sets)
            if not score or not re.search(r'\d-\d', score):
                i += 1
                continue

            # Round: cerca per cognome del vincitore nella round map
            lv      = _livello(torneo)
            rmap    = round_maps.get(torneo, {})
            surname = winner.split()[-1].lower() if " " in winner else winner.lower()
            # tennisexplorer usa "Cognome N." → prima parola è cognome
            surname_first = winner.split()[0].lower()
            rnd = (rmap.get(surname_first)
                   or rmap.get(surname)
                   or "R32")

            surf = _superficie(torneo)
            bo   = 5 if lv == "Grand Slam" else 3
            mc  += 1
            tid  = f"{date.year}-TE-{re.sub(r'[^a-z0-9]', '', torneo.lower())[:18]}"

            out.append({
                "tourney_id":         tid,
                "tourney_name":       torneo,
                "surface":            surf,
                "draw_size":          _draw_size(lv),
                "tourney_level":      lv,
                "indoor":             0,
                "tourney_date":       int(date.strftime("%Y%m%d")),
                "match_num":          mc,
                "winner_id":          None, "winner_seed":        None,
                "winner_entry":       None, "winner_name":        winner,
                "winner_hand":        None, "winner_ht":          None,
                "winner_ioc":         None, "winner_age":         None,
                "winner_rank":        None, "winner_rank_points": None,
                "loser_id":           None, "loser_seed":         None,
                "loser_entry":        None, "loser_name":         loser,
                "loser_hand":         None, "loser_ht":           None,
                "loser_ioc":          None, "loser_age":          None,
                "loser_rank":         None, "loser_rank_points":  None,
                "score":              score,
                "best_of":            bo,
                "round":              rnd,
                "minutes":            None,
                "w_ace": None, "w_df": None, "w_svpt": None,
                "w_1stIn": None, "w_1stWon": None, "w_2ndWon": None,
                "w_SvGms": None, "w_bpSaved": None, "w_bpFaced": None,
                "l_ace": None, "l_df": None, "l_svpt": None,
                "l_1stIn": None, "l_1stWon": None, "l_2ndWon": None,
                "l_SvGms": None, "l_bpSaved": None, "l_bpFaced": None,
            })

        i += 1

    return out


# ── Entry point ───────────────────────────────────────────────────────────────

def update_master_dataset(lag_days: int = 1) -> int:
    """
    Appende a master_dataset.csv i risultati mancanti da tennisexplorer.
    lag_days=1 → arriva fino a ieri (default, dati consolidati).
    Ritorna il numero di nuove righe aggiunte.
    """
    if not os.path.exists(MASTER_CSV):
        print(f"   ❌ {MASTER_CSV} non trovato")
        sys.exit(1)

    df = pd.read_csv(MASTER_CSV, low_memory=False)
    df["tourney_date"] = pd.to_numeric(df["tourney_date"], errors="coerce")

    last_int   = int(df["tourney_date"].max())
    y, rest    = divmod(last_int, 10000)
    m, d_      = divmod(rest, 100)
    last_date  = datetime.date(y, m, d_)
    start_date = last_date + datetime.timedelta(days=1)
    end_date   = datetime.date.today() - datetime.timedelta(days=lag_days)

    if start_date > end_date:
        print(f"   ✅ master_dataset già aggiornato a {last_date} (nessun gap)")
        return 0

    days = (end_date - start_date).days + 1
    print(f"   📅 Gap: {last_date} → {end_date}  ({days} giorni da colmare)")
    print(f"   🌐 Scarico da TennisExplorer...", flush=True)

    # Indice nomi TML per risolvere "Cognome I." → nome completo
    name_index = _build_name_index(df)

    # Indice deduplicazione (winner, loser, date)
    existing = set(
        zip(
            df["winner_name"].str.lower().fillna(""),
            df["loser_name"].str.lower().fillna(""),
            df["tourney_date"].astype(int),
        )
    )

    all_new = []
    cur = start_date
    while cur <= end_date:
        day_rows = scrape_day(cur)
        added = 0
        for r in day_rows:
            # Risolvi nomi abbreviati → nomi completi TML, poi normalizza
            r["winner_name"] = normalize_player_name(_resolve_name(r["winner_name"], name_index))
            r["loser_name"]  = normalize_player_name(_resolve_name(r["loser_name"],  name_index))
            key = (r["winner_name"].lower(), r["loser_name"].lower(), r["tourney_date"])
            if key not in existing:
                existing.add(key)
                all_new.append(r)
                added += 1
        n_atp = sum(1 for r in day_rows if r.get("tourney_level") != "Challenger")
        print(f"   {cur}  →  {added} nuove  |  {n_atp} ATP trovate in pagina", flush=True)
        time.sleep(0.8)
        cur += datetime.timedelta(days=1)

    if not all_new:
        print("   ℹ️  Nessuna partita nuova — TML già aggiornato o pagine vuote")
        return 0

    df_new      = pd.DataFrame(all_new, columns=df.columns)
    df_combined = pd.concat([df, df_new], ignore_index=True)
    df_combined.to_csv(MASTER_CSV, index=False)
    n = len(all_new)
    print(f"\n   ✅ Aggiunte {n} partite → master_dataset.csv ora aggiornato a {end_date}")
    return n


if __name__ == "__main__":
    n = update_master_dataset()
    print(f"\n🎾 Totale partite aggiunte: {n}")
