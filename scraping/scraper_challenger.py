"""
scraper_challenger.py
Scarica le partite dei Challenger ATP da atptour.com.

Logica:
  - challenger_matches_storico.csv  → creato UNA VOLTA con anni 2022-2025
    Se il file esiste già, skippa gli anni storici.
  - challenger_matches_2026.csv     → aggiornato ad ogni run (incrementale)

Come aggiungerlo al pipeline:
  actualizador_maestro.py → inserire 'scraper_challenger.py' prima di fusionar_historico_final.py
  fusionar_historico_final.py → già aggiornato per leggere i file Challenger
"""

import pandas as pd
import undetected_chromedriver as uc
from bs4 import BeautifulSoup
from datetime import date, timedelta
import time
import random
import os
import re

IS_CI = os.environ.get("CI", "").lower() == "true"

# ── Configurazione ─────────────────────────────────────────────────────────────
STORICO_FILE   = "challenger_matches_2025.csv"   # 2025 scrapato UNA VOLTA
CORRENTE_FILE  = "challenger_matches_2026.csv"   # 2026+, aggiornato ogni run

ANNI_STORICI   = []   # 2022-2024 da Sackmann, 2025 già scrapato
ANNO_CORRENTE  = 2026
MAX_TORNEO_GG  = 16
OGGI           = date.today()

BASE_ARCHIVE   = "https://www.atptour.com/en/scores/results-archive?year={year}&tournamentType=ch"
BASE_CURRENT   = "https://www.atptour.com/en/scores/current-challenger"
BASE_RESULTS   = "https://www.atptour.com/en/scores/archive/{slug}/{tid}/{year}/results"

# ID tornei Challenger su ATP solitamente >= 5000 (ATP 250/500/1000 < 5000)
# Usato come euristico di fallback nella detection
CHALLENGER_ID_MIN = 5000

# ── Chrome ─────────────────────────────────────────────────────────────────────
def _get_chrome_version():
    """Auto-detect della versione di Chrome installata (Linux + Windows)."""
    import subprocess as _sp
    # Linux / Mac
    for cmd in ['google-chrome', 'google-chrome-stable', 'chromium-browser', 'chromium']:
        try:
            out = _sp.check_output([cmd, '--version'], text=True, stderr=_sp.DEVNULL)
            ver = int(out.strip().split()[-1].split('.')[0])
            print(f"   ℹ️  Chrome rilevato: {out.strip()} → version_main={ver}")
            return ver
        except Exception:
            continue
    # Windows: registro di sistema (stabile, esclude beta/canary)
    try:
        import winreg
        for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
            for sub in (r'SOFTWARE\Google\Chrome\BLBeacon',
                        r'SOFTWARE\WOW6432Node\Google\Chrome\BLBeacon'):
                try:
                    key = winreg.OpenKey(hive, sub)
                    ver_str, _ = winreg.QueryValueEx(key, 'version')
                    winreg.CloseKey(key)
                    ver = int(ver_str.split('.')[0])
                    print(f"   ℹ️  Chrome (registro): {ver_str} → version_main={ver}")
                    return ver
                except Exception:
                    continue
    except ImportError:
        pass
    return None

CHROME_VERSION = _get_chrome_version()

def avvia_chrome():
    options = uc.ChromeOptions()
    options.add_argument("--start-maximized")
    if IS_CI:
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
    return uc.Chrome(options=options, version_main=CHROME_VERSION)

# ── Parse date ─────────────────────────────────────────────────────────────────
MESI_EN = {
    'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,
    'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12
}

def parse_tourney_date(text):
    if not text: return None
    text = text.strip()
    m = re.search(r'(\w+)\s+(\d+),?\s+(\d{4})', text)
    if m:
        mese = m.group(1).lower()[:3]
        if mese in MESI_EN:
            return int(m.group(3)) * 10000 + MESI_EN[mese] * 100 + int(m.group(2))
    m = re.search(r'(\d+)[\-–]\d+\s+(\w+),?\s+(\d{4})', text)
    if m:
        mese = m.group(2).lower()[:3]
        if mese in MESI_EN:
            return int(m.group(3)) * 10000 + MESI_EN[mese] * 100 + int(m.group(1))
    m = re.search(r'(\w+)\s+(\d+)[\-–]\d+,?\s+(\d{4})', text)
    if m:
        mese = m.group(1).lower()[:3]
        if mese in MESI_EN:
            return int(m.group(3)) * 10000 + MESI_EN[mese] * 100 + int(m.group(2))
    return None

def estrai_tourney_date_da_pagina(soup):
    spans = soup.select('.date-location span')
    if len(spans) >= 2:
        d = parse_tourney_date(spans[-1].get_text())
        if d: return d
    el = soup.select_one('.date-location')
    if el:
        d = parse_tourney_date(el.get_text())
        if d: return d
    return None

def _yyyymmdd_to_date(v):
    v = int(v)
    return date(v // 10000, (v % 10000) // 100, v % 100)

# ── Detection superficie ───────────────────────────────────────────────────────
def rileva_superficie(soup):
    for el in soup.select('.surface, .court-type, [class*="surface"], [class*="court"]'):
        t = el.get_text(strip=True).lower()
        if 'clay'   in t: return 'Clay'
        if 'grass'  in t: return 'Grass'
        if 'carpet' in t: return 'Carpet'
        if 'hard'   in t: return 'Hard'
    # Fallback: cerca nel testo dell'header pagina
    header = soup.select_one('.header-wrapper, .tournament-header, h1, .page-header')
    if header:
        t = header.get_text(' ', strip=True).lower()
        if 'clay'  in t: return 'Clay'
        if 'grass' in t: return 'Grass'
    return 'Hard'

# ── Discovery tornei Challenger da pagina archivio anno ───────────────────────
def get_challenger_links_da_archivio(driver, anno):
    """
    Va su results-archive?year=ANNO e trova i link dei tornei Challenger.
    Criteri di detection (in ordine di priorità):
      1. Testo "Challenger" o "125" nel container del link
      2. Classe CSS contenente "challenger"
      3. ID numerico del torneo >= CHALLENGER_ID_MIN (euristico ATP)
    """
    url = BASE_ARCHIVE.format(year=anno)
    print(f"   📋 Scopro Challenger {anno}: {url}")
    driver.get(url)
    time.sleep(5)
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)

    soup = BeautifulSoup(driver.page_source, 'html.parser')
    tornei = []
    seen = set()

    for a in soup.find_all('a', href=True):
        href = a['href']
        m = re.search(r'/en/scores/archive/([^/]+)/(\d+)/(\d+)/results', href)
        if not m: continue
        slug, tid, year_str = m.group(1), m.group(2), m.group(3)
        if int(year_str) != anno: continue
        key = (slug, tid)
        if key in seen: continue
        seen.add(key)

        is_challenger = False
        nome = slug.replace('-', ' ').title()

        # Risali l'albero HTML fino a 8 livelli per trovare marker Challenger
        parent = a
        for _ in range(8):
            parent = parent.parent
            if parent is None: break
            cls   = ' '.join(parent.get('class', [])).lower()
            testo = parent.get_text(' ', strip=True).lower()

            if 'challenger' in cls or 'challenger' in testo or ' 125' in testo:
                is_challenger = True
            # Cerca nome torneo nei tag heading
            for tag in ['h2', 'h3', 'h4']:
                el = parent.find(tag)
                if el:
                    nome_raw = el.get_text(strip=True)
                    if nome_raw and 3 < len(nome_raw) < 60:
                        nome = nome_raw
                        break
            if is_challenger:
                break

        # Euristico fallback: ID numerico >= CHALLENGER_ID_MIN
        if not is_challenger:
            try:
                if int(tid) >= CHALLENGER_ID_MIN:
                    is_challenger = True
            except ValueError:
                pass

        # Slug/nome esplicito
        if 'challenger' in slug.lower() or 'challenger' in nome.lower():
            is_challenger = True

        if is_challenger:
            tornei.append((slug, tid, nome))
            print(f"      ✓ {nome}  ({slug}/{tid})")

    if not tornei:
        print(f"      ⚠️  Nessun Challenger trovato — prova a controllare il selettore HTML")
    return tornei


# ── Discovery tornei Challenger correnti ──────────────────────────────────────
def get_challenger_links_correnti(driver):
    """Va su current-challenger e trova link dei tornei in corso/recenti."""
    print(f"   📋 Scopro Challenger correnti: {BASE_CURRENT}")
    driver.get(BASE_CURRENT)
    time.sleep(5)
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)

    soup = BeautifulSoup(driver.page_source, 'html.parser')
    tornei = []
    seen = set()

    for a in soup.find_all('a', href=True):
        href = a['href']

        # Pattern diretto: /en/scores/archive/{slug}/{tid}/{anno}/results
        m = re.search(r'/en/scores/archive/([^/]+)/(\d+)/(\d+)/results', href)
        if m:
            slug, tid, year_str = m.group(1), m.group(2), m.group(3)
            key = (slug, tid)
            if key not in seen:
                seen.add(key)
                nome = slug.replace('-', ' ').title()
                full_url = BASE_RESULTS.format(slug=slug, tid=tid, year=year_str)
                tornei.append((slug, tid, nome, full_url, int(year_str)))
            continue

        # Pattern overview torneo: /en/tournaments/{slug}/{tid}/overview
        m2 = re.search(r'/en/tournaments/([^/]+)/(\d+)/overview', href)
        if m2:
            slug, tid = m2.group(1), m2.group(2)
            key = (slug, tid)
            if key not in seen:
                seen.add(key)
                nome = slug.replace('-', ' ').title()
                full_url = BASE_RESULTS.format(slug=slug, tid=tid, year=ANNO_CORRENTE)
                tornei.append((slug, tid, nome, full_url, ANNO_CORRENTE))

    return tornei


# ── Scraping partite da pagina results ────────────────────────────────────────
def scrapa_partite_torneo(driver, url, torneo_id, torneo_nome, anno):
    driver.get(url)
    time.sleep(5)
    if "just a moment" in driver.title.lower():
        print("      🛑 Cloudflare — attendo 12 s...")
        time.sleep(12)
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)

    soup = BeautifulSoup(driver.page_source, 'html.parser')
    matches_raw = soup.find_all('div', class_='match')

    tourney_date = estrai_tourney_date_da_pagina(soup) or 0
    surface      = rileva_superficie(soup)

    partite = []
    for m in matches_raw:
        try:
            round_el  = m.find('div', class_='match-header')
            round_txt = round_el.get_text(strip=True).split("-")[0].strip() if round_el else 'R'

            players = m.find_all('div', class_='stats-item')
            if len(players) < 2: continue

            n1 = players[0].find('div', class_='name')
            n2 = players[1].find('div', class_='name')
            if not n1 or not n2: continue

            p1 = n1.get_text(strip=True).split("(")[0].strip()
            p2 = n2.get_text(strip=True).split("(")[0].strip()

            if players[0].find('div', class_='winner'):
                winner, loser = p1, p2
                w_node, l_node = players[0], players[1]
            else:
                winner, loser = p2, p1
                w_node, l_node = players[1], players[0]

            score_parts = []
            sw = w_node.select('.score-item span')
            sl = l_node.select('.score-item span')
            for k in range(min(len(sw), len(sl))):
                v1, v2 = sw[k].get_text(strip=True), sl[k].get_text(strip=True)
                if v1 and v2: score_parts.append(f"{v1}-{v2}")

            partite.append({
                'tourney_id':    torneo_id,
                'tourney_name':  torneo_nome,
                'tourney_level': 'C',
                'surface':       surface,
                'tourney_date':  tourney_date,
                'winner_name':   winner,
                'loser_name':    loser,
                'score':         " ".join(score_parts),
                'round':         round_txt,
                'minutes':       0,
                'scraping_date': str(OGGI),
            })
        except Exception:
            continue

    return partite, tourney_date

COLS_EXTRA = [
    'draw_size', 'match_num', 'winner_id', 'winner_seed', 'winner_entry',
    'winner_hand', 'winner_ht', 'winner_ioc', 'winner_age',
    'loser_id', 'loser_seed', 'loser_entry', 'loser_hand', 'loser_ht',
    'loser_ioc', 'loser_age', 'best_of',
    'winner_rank', 'winner_rank_points', 'loser_rank', 'loser_rank_points',
]

def aggiungi_colonne_extra(df):
    for c in COLS_EXTRA:
        if c not in df.columns:
            df[c] = 0
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
print("🎾 SCRAPER CHALLENGER ATP")
print("=" * 60)

driver = avvia_chrome()

# ── A. STORICO 2022-2025 ───────────────────────────────────────────────────────
# Carica tornei già scaricati (challenger_matches_storico.csv + historialTenis.csv)
tornei_storico_gia_presenti = set()

if os.path.exists(STORICO_FILE):
    df_storico_exist = pd.read_csv(STORICO_FILE)
    tornei_storico_gia_presenti.update(df_storico_exist['tourney_id'].unique())
else:
    df_storico_exist = pd.DataFrame()

# Controlla anche historialTenis.csv — evita di riscrapeire ciò che è già nel dataset
HISTORIAL = os.path.join(os.path.dirname(__file__), 'historialTenis.csv')
if os.path.exists(HISTORIAL):
    try:
        df_hist = pd.read_csv(HISTORIAL, usecols=['tourney_id'], low_memory=False)
        n_hist = len(df_hist['tourney_id'].unique())
        tornei_storico_gia_presenti.update(df_hist['tourney_id'].unique())
        print(f"   → {n_hist} tornei già in historialTenis.csv — saranno saltati")
    except Exception as e:
        print(f"   ⚠️  Impossibile leggere historialTenis.csv: {e}")

n_anni_mancanti = sum(
    1 for a in ANNI_STORICI
    if not any(str(tid).startswith(f"{a}-chall-") for tid in tornei_storico_gia_presenti)
)
if n_anni_mancanti == 0 and df_storico_exist is not None and not df_storico_exist.empty:
    print(f"\n✅ {STORICO_FILE} completo ({len(df_storico_exist)} partite) — skip storico")
else:
    print(f"\n📥 Storico Challenger: {len(tornei_storico_gia_presenti)} tornei già presenti — "
          f"scarico solo i mancanti...")

# Determina se ci sono anni da completare
anni_da_scaricare = [
    a for a in ANNI_STORICI
    if not all(
        f"{a}-chall-{slug}" in tornei_storico_gia_presenti
        for slug, _, _ in []  # placeholder: verifichiamo torneo per torneo sotto
    )
]

storico_aggiornato = False

for anno in ANNI_STORICI:
    print(f"\n📅 Anno {anno}:")
    try:
        tornei = get_challenger_links_da_archivio(driver, anno)
        print(f"   → {len(tornei)} Challenger trovati")
        nuovi = [(s, t, n) for s, t, n in tornei
                 if f"{anno}-chall-{s}" not in tornei_storico_gia_presenti]
        if not nuovi:
            print(f"   ⏭️  Tutti già scaricati — skip")
            continue

        partite_anno = []
        for slug, tid, nome in nuovi:
            torneo_id = f"{anno}-chall-{slug}"
            url = BASE_RESULTS.format(slug=slug, tid=tid, year=anno)
            print(f"   🌍 {nome}  →  {url}")
            try:
                partite, tdate = scrapa_partite_torneo(driver, url, torneo_id, nome, anno)
                print(f"      ✅ {len(partite)} partite  |  data: {tdate}")
                partite_anno.extend(partite)
                tornei_storico_gia_presenti.add(torneo_id)
            except Exception as e:
                print(f"      ❌ {e}")
            time.sleep(random.uniform(2, 5))

        # Salvataggio incrementale dopo ogni anno
        if partite_anno:
            df_nuovi = aggiungi_colonne_extra(pd.DataFrame(partite_anno))
            df_storico_exist = pd.concat([df_storico_exist, df_nuovi], ignore_index=True) \
                               if not df_storico_exist.empty else df_nuovi
            df_storico_exist.to_csv(STORICO_FILE, index=False)
            storico_aggiornato = True
            print(f"   💾 Salvato incrementalmente: {len(df_storico_exist)} partite totali")

    except Exception as e:
        print(f"   ❌ Errore anno {anno}: {e}")

if storico_aggiornato:
    print(f"\n✅ Storico completo: {len(df_storico_exist)} partite → {STORICO_FILE}")
elif not tornei_storico_gia_presenti:
    print("\n⚠️  Nessuna partita storica trovata — controlla i selettori HTML")


# ── B. CORRENTE 2026 (incrementale, salvataggio per torneo) ───────────────────
print(f"\n🔄 Aggiorno Challenger {ANNO_CORRENTE}...")

df_2026_esistente       = pd.DataFrame()
tornei_2026_completati  = set()
tornei_2026_in_corso    = set()

if os.path.exists(CORRENTE_FILE):
    df_2026_esistente = pd.read_csv(CORRENTE_FILE)
    for tid in df_2026_esistente['tourney_id'].unique():
        righe = df_2026_esistente[df_2026_esistente['tourney_id'] == tid]
        try:
            data_t = _yyyymmdd_to_date(int(righe['tourney_date'].max()))
        except Exception:
            data_t = OGGI
        if data_t + timedelta(days=MAX_TORNEO_GG) >= OGGI:
            tornei_2026_in_corso.add(tid)
        else:
            tornei_2026_completati.add(tid)
    print(f"   → {len(df_2026_esistente)} partite esistenti | "
          f"{len(tornei_2026_completati)} completati | {len(tornei_2026_in_corso)} in corso")

# Scopri tornei 2026 da archive + current-challenger
tornei_2026 = {}   # torneo_id → (url, nome)

try:
    for slug, tid, nome in get_challenger_links_da_archivio(driver, ANNO_CORRENTE):
        key = f"{ANNO_CORRENTE}-chall-{slug}"
        tornei_2026[key] = (BASE_RESULTS.format(slug=slug, tid=tid, year=ANNO_CORRENTE), nome)
except Exception as e:
    print(f"   ⚠️  Archive {ANNO_CORRENTE}: {e}")

try:
    for item in get_challenger_links_correnti(driver):
        slug, tid, nome, url, anno_c = item
        key = f"{anno_c}-chall-{slug}"
        tornei_2026[key] = (url, nome)
except Exception as e:
    print(f"   ⚠️  Current-challenger: {e}")

print(f"   → {len(tornei_2026)} Challenger {ANNO_CORRENTE} trovati")

# df di lavoro: mantieni i tornei completati, riscrape quelli in corso/nuovi
df_2026_work = df_2026_esistente[
    df_2026_esistente['tourney_id'].isin(tornei_2026_completati)
].copy() if not df_2026_esistente.empty else pd.DataFrame()

for torneo_id, (url, nome) in tornei_2026.items():
    if torneo_id in tornei_2026_completati:
        print(f"   ⏭️  {nome} — completato (>{MAX_TORNEO_GG} gg), skip")
        continue
    stato = "in corso, riscarico" if torneo_id in tornei_2026_in_corso else "nuovo"
    print(f"   🌍 {nome} [{stato}]  →  {url}")
    try:
        partite, tdate = scrapa_partite_torneo(driver, url, torneo_id, nome, ANNO_CORRENTE)
        if not partite:
            print(f"      ⚠️  0 partite — torneo futuro. Stop.")
            break
        print(f"      ✅ {len(partite)} partite  |  data: {tdate}")
        df_nuovo = aggiungi_colonne_extra(pd.DataFrame(partite))
        df_2026_work = pd.concat([df_2026_work, df_nuovo], ignore_index=True) \
                       if not df_2026_work.empty else df_nuovo
        df_2026_work.to_csv(CORRENTE_FILE, index=False)
    except Exception as e:
        print(f"      ❌ {e}")
    time.sleep(random.uniform(2, 5))

driver.quit()

if not df_2026_work.empty:
    print(f"\n💾 {CORRENTE_FILE}: {len(df_2026_work)} partite totali")
else:
    print(f"\n✅ Nessuna nuova partita {ANNO_CORRENTE} — {CORRENTE_FILE} invariato")

print("\n🎾 Scraping Challenger completato!")
