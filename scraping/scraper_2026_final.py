import pandas as pd
import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from datetime import date, datetime, timedelta
import time
import random
import os
import re

# --- CONFIGURACIÓN ---
ARCHIVO_ENTRADA = "atp_torneos_2026_final.csv"
ARCHIVO_SALIDA = "atp_matches_2026_indetectable.csv"
ANIO = 2026

# FORCE_RESCRAPE = True  → riscarica TUTTO ignorando le date (equivale al primo run)
# FORCE_RESCRAPE = False → comportamento normale (incrementale)
FORCE_RESCRAPE = False

print(f"🥷 INICIANDO MODO INDETECTABLE ({ANIO})...")
if FORCE_RESCRAPE:
    print("   ⚡ FORCE_RESCRAPE = True — riscarico tutto da zero")

# ── Helper: converte data testuale → intero YYYYMMDD ─────────────────────────
MESI_EN = {
    'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,
    'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12
}
def parse_tourney_date(text):
    """Tenta di estrarre una data YYYYMMDD da testo come 'Jan 06, 2026' o '6-19 Jan 2026'."""
    if not text:
        return None
    text = text.strip()
    # Formato: "Jan 06, 2026" o "January 06, 2026"
    m = re.search(r'(\w+)\s+(\d+),?\s+(\d{4})', text)
    if m:
        mese = m.group(1).lower()[:3]
        giorno = int(m.group(2))
        anno = int(m.group(3))
        if mese in MESI_EN:
            return anno * 10000 + MESI_EN[mese] * 100 + giorno
    # Formato: "6-19 Jan, 2026" o "6-19 Jan 2026" → prende il giorno iniziale
    m = re.search(r'(\d+)[\-–]\d+\s+(\w+),?\s+(\d{4})', text)
    if m:
        giorno = int(m.group(1))
        mese = m.group(2).lower()[:3]
        anno = int(m.group(3))
        if mese in MESI_EN:
            return anno * 10000 + MESI_EN[mese] * 100 + giorno
    # Formato: "Jan 6-19, 2026"
    m = re.search(r'(\w+)\s+(\d+)[\-–]\d+,?\s+(\d{4})', text)
    if m:
        mese = m.group(1).lower()[:3]
        giorno = int(m.group(2))
        anno = int(m.group(3))
        if mese in MESI_EN:
            return anno * 10000 + MESI_EN[mese] * 100 + giorno
    return None

def estrai_tourney_date_da_pagina(soup):
    """Estrae la data del torneo dalla pagina ATP results.
    L'HTML ha: <div class="date-location"><span>City</span> | <span>4-11 Jan, 2026</span></div>
    """
    # Selettore preciso: secondo span dentro .date-location
    spans = soup.select('.date-location span')
    if len(spans) >= 2:
        d = parse_tourney_date(spans[-1].get_text())
        if d:
            return d
    # Fallback: tutto il testo di .date-location
    el = soup.select_one('.date-location')
    if el:
        d = parse_tourney_date(el.get_text())
        if d:
            return d
    return None

# 1. PREPARAR DATOS
try:
    df_torneos = pd.read_csv(ARCHIVO_ENTRADA)
    urls = []
    names = []
    print("🔗 Procesando enlaces...")
    for link in df_torneos['Link_Resultados']:
        parts = link.strip().split('/')
        try:
            if 'tournaments' in parts:
                idx = parts.index('tournaments')
                nombre = parts[idx+1]
                id_t = parts[idx+2]
            elif 'archive' in parts:
                idx = parts.index('archive')
                nombre = parts[idx+1]
                id_t = parts[idx+2]
            else:
                continue
            new_link = f"https://www.atptour.com/en/scores/archive/{nombre}/{id_t}/{ANIO}/results"
            urls.append(new_link)
            names.append(nombre)
        except:
            pass
except Exception as e:
    print(f"❌ Error leyendo CSV: {e}")
    exit()

# 2. INICIAR NAVEGADOR INDETECTABLE
options = uc.ChromeOptions()
options.add_argument("--start-maximized")

print("🚀 Lanzando Chrome parcheado (puede tardar unos segundos)...")
driver = uc.Chrome(options=options, version_main=145)

# ── CARICA DATI ESISTENTI (logica incrementale + date-aware) ─────────────────
MAX_TORNEO_GG = 16
OGGI = date.today()

all_matches = []
tornei_gia_scaricati = set()
tornei_in_corso      = set()
df_esistente = pd.DataFrame()

if FORCE_RESCRAPE:
    print(f"📂 FORCE_RESCRAPE: ignoro CSV esistente — riscrivo tutto")
elif os.path.exists(ARCHIVO_SALIDA):
    try:
        df_letto = pd.read_csv(ARCHIVO_SALIDA)

        for tid in df_letto['tourney_id'].unique():
            righe = df_letto[df_letto['tourney_id'] == tid]

            # Usa tourney_date (YYYYMMDD) per capire se il torneo è ancora in corso
            try:
                td_val = int(righe['tourney_date'].max())
                anno = td_val // 10000
                mese = (td_val % 10000) // 100
                giorno = td_val % 100
                data_torneo = date(anno, mese, giorno)
            except Exception:
                data_torneo = OGGI  # se non si riesce a leggere, trattalo come in corso

            # Se tourney_date + MAX_TORNEO_GG è ancora nel futuro → torneo in corso
            fine_stimata = data_torneo + timedelta(days=MAX_TORNEO_GG)
            if fine_stimata >= OGGI:
                tornei_in_corso.add(tid)
            else:
                tornei_gia_scaricati.add(tid)

        df_esistente = df_letto[~df_letto['tourney_id'].isin(tornei_in_corso)].copy()

        n_tot = len(tornei_gia_scaricati) + len(tornei_in_corso)
        print(f"📂 CSV trovato: {len(df_letto)} partite, {n_tot} tornei")
        if tornei_in_corso:
            print(f"   🔄 Da riscarica ({len(tornei_in_corso)}): {sorted(tornei_in_corso)}")
        if tornei_gia_scaricati:
            print(f"   ⏭️  Vecchi (>{MAX_TORNEO_GG} gg): {len(tornei_gia_scaricati)} tornei saltati")

    except Exception as e:
        print(f"⚠️  Impossibile leggere il CSV ({e}) — ripartiamo da zero")
else:
    print(f"📂 Nessun CSV precedente — scarico tutto da zero")

# ── Mappa date note (indice torneo → tourney_date YYYYMMDD) ──────────────────
# Popolata dai dati esistenti + aggiornata durante il loop corrente
known_dates = {}  # int(i) -> int(YYYYMMDD)
if not df_esistente.empty and 'tourney_id' in df_esistente.columns and 'tourney_date' in df_esistente.columns:
    for tid, td in df_esistente[['tourney_id', 'tourney_date']].drop_duplicates().values:
        try:
            idx = int(str(tid).split('-')[-1])
            known_dates[idx] = int(td)
        except Exception:
            pass

def _yyyymmdd_to_date(v):
    v = int(v)
    return date(v // 10000, (v % 10000) // 100, v % 100)

def _date_to_yyyymmdd(d):
    return d.year * 10000 + d.month * 100 + d.day

def stima_data_vicini(i, known, n_total):
    """Stima tourney_date per il torneo i interpolando tra vicini noti.
    - prev + next noti: punto medio
    - solo prev noti:   prev + 7 giorni
    - nessun vicino:    data odierna (torneo in corso)
    """
    prev_date = None
    next_date = None
    for j in range(i - 1, -1, -1):
        if j in known:
            prev_date = known[j]
            break
    for j in range(i + 1, n_total):
        if j in known:
            next_date = known[j]
            break

    if prev_date and next_date:
        pd_ = _yyyymmdd_to_date(prev_date)
        nd_ = _yyyymmdd_to_date(next_date)
        stima = pd_ + (nd_ - pd_) // 2
        motivo = f"interpolazione tra {prev_date} e {next_date}"
    elif prev_date:
        from datetime import timedelta as _td
        stima = _yyyymmdd_to_date(prev_date) + _td(weeks=1)
        motivo = f"prev={prev_date} + 7gg"
    else:
        stima = OGGI
        motivo = "nessun vicino noto → data odierna"

    return _date_to_yyyymmdd(stima), motivo

# 3. BUCLE DE TORNEOS
for i, url in enumerate(urls):
    torneo = names[i]
    torneo_id = f"{ANIO}-{torneo}-{i}"
    print(f"\n🌍 [{i+1}/{len(urls)}] {torneo}")
    print(f"   Link: {url}")

    if not FORCE_RESCRAPE:
        if torneo_id in tornei_gia_scaricati:
            print(f"   ⏭️  Gia' scaricato (>{MAX_TORNEO_GG} gg fa) — salto")
            continue
        if torneo_id in tornei_in_corso:
            print(f"   🔄 Riscarico (recente o senza data)")

    try:
        driver.get(url)
        time.sleep(5)

        if "just a moment" in driver.title.lower():
            print("🛑 Cloudflare detectado. 10 secondi...")
            time.sleep(10)

        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)

        soup = BeautifulSoup(driver.page_source, 'html.parser')
        matches = soup.find_all('div', class_='match')

        if not matches:
            print("   ⚠️ 0 partidos — torneo futuro. Stop.")
            break

        # ── Estrai data torneo dalla pagina (YYYYMMDD) ───────────────────────
        tourney_date = estrai_tourney_date_da_pagina(soup)
        if tourney_date:
            known_dates[i] = tourney_date
            print(f"   📅 Data torneo rilevata: {tourney_date}")
        else:
            tourney_date, motivo = stima_data_vicini(i, known_dates, len(urls))
            known_dates[i] = tourney_date
            print(f"   ⚠️  Data non leggibile — stimata: {tourney_date} ({motivo})")

        print(f"   ✅ {len(matches)} partite trovate")

        count = 0
        for m in matches:
            try:
                round_txt = m.find('div', class_='match-header').get_text(strip=True).split("-")[0]
                players = m.find_all('div', class_='stats-item')
                if len(players) < 2: continue

                p1 = players[0].find('div', class_='name').get_text(strip=True).split("(")[0].strip()
                p2 = players[1].find('div', class_='name').get_text(strip=True).split("(")[0].strip()

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

                all_matches.append({
                    'tourney_id':   f"{ANIO}-{torneo}-{i}",
                    'tourney_name': torneo,
                    'surface':      'Hard',
                    'tourney_date': tourney_date,   # ← data YYYYMMDD corretta
                    'winner_name':  winner,
                    'loser_name':   loser,
                    'score':        " ".join(score_parts),
                    'round':        round_txt,
                    'minutes':      100,
                    'scraping_date': str(OGGI)
                })
                count += 1
            except:
                continue

    except Exception as e:
        print(f"   ❌ Error: {e}")
        if "no such window" in str(e) or "target window" in str(e):
            print("   🔄 Chrome si e' chiuso. Riavvio...")
            try: driver.quit()
            except: pass
            time.sleep(3)
            driver = uc.Chrome(options=options, version_main=145)
            time.sleep(2)

driver.quit()

# 4. GUARDAR
if all_matches:
    df_new = pd.DataFrame(all_matches)
    # Colonne extra per compatibilita' (tourney_date e' gia' nel dict sopra)
    cols_extra = ['draw_size','tourney_level','match_num','winner_id','winner_seed',
                  'winner_entry','winner_hand','winner_ht','winner_ioc','winner_age',
                  'loser_id','loser_seed','loser_entry','loser_hand','loser_ht',
                  'loser_ioc','loser_age','best_of','winner_rank','winner_rank_points',
                  'loser_rank','loser_rank_points']
    for c in cols_extra:
        if c not in df_new.columns:
            df_new[c] = 0

    if not df_esistente.empty:
        df_finale = pd.concat([df_esistente, df_new], ignore_index=True)
        print(f"\n📎 {len(df_esistente)} mantenute + {len(df_new)} nuove/aggiornate")
    else:
        df_finale = df_new

    df_finale.to_csv(ARCHIVO_SALIDA, index=False)
    print(f"\n🎉 {len(df_finale)} partite totali in {ARCHIVO_SALIDA}")
    print(f"   (di cui {len(df_new)} nuove/aggiornate oggi)")
else:
    if tornei_gia_scaricati or tornei_in_corso:
        print("\n✅ Nessuna nuova partita — CSV gia' aggiornato.")
    else:
        print("\n❌ No se bajaron datos.")