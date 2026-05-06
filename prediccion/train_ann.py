"""
train_ann.py v5.0  —  ANN + GBM Tennis Predictor
=================================================
BUG FIX v5 vs v4:
  ✅ [BUG1] stats_superficie running — win-rate calcolato SOLO su dati passati (no global leak)
  ✅ [BUG2] Temporal split — train/val/test per data cronologica, non random
  ✅ [BUG3] Mirrored pairs — con split temporale sono automaticamente nello stesso fold

NUOVE FEATURE v5:
  ✅ diff_form_volatility   — deviazione std ultimi 10 risultati (giocatore inconsistente)
  ✅ diff_close_match_pct   — % match al 3° set ultimi 10 (vulnerabilità)
  ✅ diff_upset_tendency    — frequenza sconfitte vs rank peggiori (ultimi 20)
  ✅ diff_late_round_wr     — win rate storico in QF/SF/F
  ✅ diff_surface_trend     — trend su superficie: last-5 vs prev-5
  ✅ diff_weeks_load        — n. match nelle ultime 8 settimane (fatica accumulata)

FEATURE RIMOSSE (bassa importanza da letteratura):
  ❌ diff_age, diff_ht      — correlazione bassa, confermato da ricerca
  ❌ tourney_date            — indice temporale, non predittore

QUICK_TEST = False           — True = 3 epoche, 5 trial: debug rapido + feature importance

UTILIZZO:
    cd prediccion/
    python train_ann.py

OUTPUT AGGIUNTIVI v5:
    close_match_hist.pkl / upset_hist.pkl / late_round_hist.pkl / match_load.pkl
"""

import os, warnings, sys
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scraping'))
from glicko2 import Glicko2Store

from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss
warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
    HAS_LGB = True
    print("   ✅ LightGBM disponibile")
except ImportError:
    HAS_LGB = False

try:
    import xgboost as xgb_lib
    HAS_XGB = True
    print("   ✅ XGBoost disponibile")
except ImportError:
    HAS_XGB = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("⚠️  optuna non trovato. pip install optuna")

# ─── Seed + Device ────────────────────────────────────────────────────────────
SEED = 42
import random
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Device: {device}")

# ─── Configurazione globale ───────────────────────────────────────────────────
# QUICK_TEST = True → 2 epoche + 5 trial: per verificare feature / debug rapido
QUICK_TEST  = False
ML_MIN_YEAR = 2015  # Solo partite >= questo anno nel training ML (Glicko-2 usa tutto lo storico)

TRIALS      = 5   if QUICK_TEST else 100
TRIALS_GBM  = 5   if QUICK_TEST else 50
MAX_EPOCHS  = 3   if QUICK_TEST else 120

LEVEL_MULT = {'G': 2.0, 'M': 1.5, 'F': 1.4, 'A': 1.0,
              'D': 1.0, 'C': 0.8, 'S': 0.7, 'E': 0.5}
import platform as _platform
N_WORKERS  = min(4, os.cpu_count() or 1) if (torch.cuda.is_available() and _platform.system() == 'Linux') else 0

print(f"\n⚙️  Configurazione: TRIALS={TRIALS}, MAX_EPOCHS={MAX_EPOCHS}, QUICK_TEST={QUICK_TEST}, n_workers={N_WORKERS}")

def parse_total_games(score_str):
    """Estrae totale game e set da score string. Es: '7-6 6-2' → (21, 2)"""
    if not isinstance(score_str, str) or not score_str.strip():
        return np.nan, np.nan
    score = score_str.strip().upper()
    if any(x in score for x in ['W/O', 'RET', 'DEF', 'WO', 'ABN', 'UNP', 'BYE', 'NA', 'NAN']):
        return np.nan, np.nan
    valid_tokens = [t for t in score.split() if '-' in t]
    sets_played  = len(valid_tokens)
    total_games  = 0
    for token in valid_tokens:
        parts = token.split('(')[0].split('-')
        if len(parts) == 2:
            try:
                total_games += int(parts[0]) + int(parts[1])
            except ValueError:
                continue
    return (float(total_games), float(sets_played)) if total_games > 0 else (np.nan, np.nan)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def carica_e_prepara(csv_path: str):
    print(f"\n📂 Caricamento: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    df['tourney_date'] = pd.to_numeric(df['tourney_date'], errors='coerce')
    df = df.sort_values(by=['tourney_date', 'match_num']).reset_index(drop=True)
    df['minutes'] = pd.to_numeric(df['minutes'], errors='coerce').fillna(90)
    # Converti tutte le colonne statistiche a numerico (il CSV TML può avere stringhe miste)
    _stat_cols = [c for c in df.columns if any(c.startswith(p) for p in
                  ('w_', 'l_', 'winner_rank', 'loser_rank', 'winner_rank_points', 'loser_rank_points'))]
    for c in _stat_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    print(f"   → {len(df):,} partite caricate")

    # ── Encoding ─────────────────────────────────────────────────────────────
    surface_map = {'Hard': 0, 'Clay': 1, 'Grass': 2, 'Carpet': 3}
    level_map   = {'G': 5, 'M': 4, 'A': 3, 'D': 3, 'F': 4, 'C': 2, 'S': 1, 'E': 0}
    round_map   = {'F': 7, 'SF': 6, 'QF': 5, 'R16': 4, 'R32': 3,
                   'R64': 2, 'R128': 1, 'RR': 4, 'BR': 3}
    late_rounds = {'F', 'SF', 'QF'}

    df['surface_enc']       = df['surface'].map(surface_map).fillna(0)
    df['tourney_level_enc'] = df['tourney_level'].map(level_map).fillna(2)
    df['round_enc']         = df['round'].map(round_map).fillna(3)
    df['is_late_round']     = df['round'].isin(late_rounds).astype(int)

    def detect_country(name):
        t = str(name).upper()
        if 'MADRID' in t or 'BARCELONA' in t: return 'ESP'
        if 'PARIS' in t or 'ROLAND' in t:     return 'FRA'
        if 'US OPEN' in t or 'INDIAN' in t:   return 'USA'
        if 'WIMBLEDON' in t or 'LONDON' in t: return 'GBR'
        if 'AUSTRALIAN' in t or 'MELBOURNE' in t: return 'AUS'
        return 'NEUTRAL'
    df['tourney_ioc'] = df['tourney_name'].apply(detect_country)

    # ── Strutture running — tutte aggiornate DOPO la lettura (no leakage) ────
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
    wrate_running     = {}   # {(player, surf): [wins, total]} — FIX BUG1: non più globale
    close_match_t     = {}   # {player: [1/0 se 3° set, last 10]}
    upset_hist_t      = {}   # {player: [(result, rank_diff), last 20]}
    late_round_t      = {}   # {player: [results in QF/SF/F]}
    match_load_t      = {}   # {player: [match days, rolling]}

    # Glicko-2: uno store per ogni superficie + overall
    g2_hard  = Glicko2Store()
    g2_clay  = Glicko2Store()
    g2_grass = Glicko2Store()
    g2_gen   = Glicko2Store()

    SURF_STORE = {'Hard': g2_hard, 'Clay': g2_clay, 'Grass': g2_grass}

    def _date_to_days(d):
        y, rest = divmod(int(d), 10000)
        m, day  = divmod(rest, 100)
        return y * 365 + m * 30 + day

    def _safe_float(v):
        return float(v) if not (isinstance(v, float) and np.isnan(v)) else 0.0

    rows = []
    for idx, row in df.iterrows():
        tid  = row['tourney_id']
        w, l = row['winner_name'], row['loser_name']
        dur  = row['minutes']
        surf = row['surface']

        if not isinstance(w, str) or not isinstance(l, str) or pd.isna(w) or pd.isna(l):
            continue
        if not isinstance(surf, str):
            continue

        total_games, sets_played = parse_total_games(str(row.get('score', '')))
        best_of_val = row.get('best_of', 3)
        is_bo5 = 1.0 if best_of_val == 5 or best_of_val == '5' else 0.0
        if pd.notna(sets_played) and sets_played >= 4:
            is_bo5 = 1.0

        y_sets = np.nan
        if pd.notna(sets_played) and not is_bo5:
            y_sets = 0.0 if sets_played == 2 else (1.0 if sets_played == 3 else np.nan)
        # is_3set: usato per close_match_pct (solo Bo3)
        is_3set = 1.0 if (pd.notna(sets_played) and sets_played == 3 and not is_bo5) else 0.0

        # --- Ranking ---
        rk_w  = float(row['winner_rank'])        if pd.notna(row.get('winner_rank'))        else 500
        rk_l  = float(row['loser_rank'])         if pd.notna(row.get('loser_rank'))         else 500
        pts_w = float(row['winner_rank_points']) if pd.notna(row.get('winner_rank_points')) else 0.0
        pts_l = float(row['loser_rank_points'])  if pd.notna(row.get('loser_rank_points'))  else 0.0

        # ── [BUG1 FIX] Win-rate superficie RUNNING ────────────────────────
        # Legge PRIMA di aggiornare → usa solo storico passato
        def _get_wrate(player, surface):
            rec = wrate_running.get((player, surface), [0, 0])
            return rec[0] / rec[1] if rec[1] >= 5 else 0.5
        sk_w = _get_wrate(w, surf)
        sk_l = _get_wrate(l, surf)

        # --- Fatica torneo ---
        f_w = fatiga_t.get((tid, w), 0.0)
        f_l = fatiga_t.get((tid, l), 0.0)
        # (aggiornamento dopo lettura)

        # --- Momentum + Surface Trend ---
        hw = racha_t.get((w, surf), [])
        hl = racha_t.get((l, surf), [])
        mw = np.mean(hw) if hw else 0.5
        ml = np.mean(hl) if hl else 0.5

        def _surf_trend(hist):
            """Last-5 mean vs prev-5 mean su questa superficie."""
            if len(hist) < 5:
                return 0.0
            recent = float(np.mean(hist[-5:]))
            prev   = float(np.mean(hist[:-5])) if len(hist) > 5 else 0.5
            return recent - prev
        st_w = _surf_trend(hw)
        st_l = _surf_trend(hl)
        # (aggiornamento dopo lettura)

        # --- H2H globale ---
        p1k, p2k = sorted([w, l])
        key = (p1k, p2k)
        rec = h2h_t.get(key, [0, 0])
        h2h_w = (rec[0] - rec[1]) if w == p1k else (rec[1] - rec[0])
        h2h_l = -h2h_w
        # (aggiornamento dopo lettura)

        # --- H2H superficie ---
        key_s = (p1k, p2k, surf)
        rec_s = h2h_surf_t.get(key_s, [0, 0])
        h2h_s_w = (rec_s[0] - rec_s[1]) if w == p1k else (rec_s[1] - rec_s[0])
        h2h_s_l = -h2h_s_w
        # (aggiornamento dopo lettura)

        # --- Days since last match ---
        td      = int(row['tourney_date']) if pd.notna(row.get('tourney_date')) else 20200101
        td_days = _date_to_days(td)
        last_w  = last_match_date_t.get(w)
        last_l  = last_match_date_t.get(l)
        days_since_w = float(np.clip((td_days - _date_to_days(last_w)) if last_w else 14, 0, 180))
        days_since_l = float(np.clip((td_days - _date_to_days(last_l)) if last_l else 14, 0, 180))
        # (aggiornamento dopo lettura)

        # --- Weeks load (match nelle ultime 8 settimane = 56 giorni) ---
        def _weeks_load(player, cur_days):
            dates = match_load_t.get(player, [])
            return float(sum(1 for d in dates if cur_days - d <= 56))
        wload_w = _weeks_load(w, td_days)
        wload_l = _weeks_load(l, td_days)
        # (aggiornamento dopo lettura)

        # --- Glicko-2 superficie (PRIMA dell'aggiornamento, no leakage) ---
        store_s = SURF_STORE.get(surf, g2_gen)
        g2_r_w,  g2_rd_w,  _ = store_s.get(w)
        g2_r_l,  g2_rd_l,  _ = store_s.get(l)
        g2_ov_w, g2_ov_rd_w, _ = g2_gen.get(w)
        g2_ov_l, g2_ov_rd_l, _ = g2_gen.get(l)
        # (aggiornamento dopo lettura)

        # --- Indoor ---
        indoor_val = 0.0 if str(row.get('indoor', 'O')).upper() in ('O', '', 'NAN') else 1.0

        # --- Recent form + Form Volatility ---
        rf_w = recent_form_t.get(w, [])
        rf_l = recent_form_t.get(l, [])
        form_w = float(np.mean(rf_w)) if rf_w else 0.5
        form_l = float(np.mean(rf_l)) if rf_l else 0.5
        fvol_w = float(np.std(rf_w))  if len(rf_w) >= 3 else 0.3
        fvol_l = float(np.std(rf_l))  if len(rf_l) >= 3 else 0.3
        # (aggiornamento dopo lettura)

        # --- Striscia attiva (PRIMA dell'aggiornamento) ---
        str_w = streak_t.get(w, 0)
        str_l = streak_t.get(l, 0)
        # (aggiornamento dopo lettura)

        # --- Close match % ---
        cm_w_hist = close_match_t.get(w, [])
        cm_l_hist = close_match_t.get(l, [])
        cmp_w = float(np.mean(cm_w_hist)) if cm_w_hist else 0.3
        cmp_l = float(np.mean(cm_l_hist)) if cm_l_hist else 0.3
        # (aggiornamento dopo lettura)

        # --- Upset tendency (% partite perse contro rank peggiore) ---
        def _upset_tendency(hist):
            """Frequenza di sconfitte vs avversari con rank peggiore (num più alto)."""
            if not hist: return 0.2
            vs_lower = [(r, rd) for r, rd in hist if rd > 0]   # rd>0 = avversario rank peggiore
            if not vs_lower: return 0.2
            losses = sum(1 for r, _ in vs_lower if r == 0)
            return losses / len(vs_lower)
        uph_w = upset_hist_t.get(w, [])
        uph_l = upset_hist_t.get(l, [])
        up_w = _upset_tendency(uph_w)
        up_l = _upset_tendency(uph_l)
        # (aggiornamento dopo lettura)

        # --- Late round win rate (QF/SF/F) ---
        lr_w = late_round_t.get(w, [])
        lr_l = late_round_t.get(l, [])
        lrwr_w = float(np.mean(lr_w)) if lr_w else 0.5
        lrwr_l = float(np.mean(lr_l)) if lr_l else 0.5
        # (aggiornamento dopo lettura)

        # --- Statistiche servizio ---
        def get_sa(player):
            s = serve_t.get(player, {})
            return {'ace':      float(np.mean(s.get('ace',     [0.0]))),
                    '1st_pct':  float(np.mean(s.get('1st_pct', [0.60]))),
                    '1st_won':  float(np.mean(s.get('1st_won', [0.70]))),
                    '2nd_won':  float(np.mean(s.get('2nd_won', [0.50]))),
                    'bp_saved': float(np.mean(s.get('bp_saved',[0.60])))}
        sa_w = get_sa(w); sa_l = get_sa(l)

        # --- Statistiche ritorno ---
        def get_ra(player):
            s = return_t.get(player, {})
            return {'return_pct': float(np.mean(s.get('return_pct', [0.35]))),
                    'bp_conv':    float(np.mean(s.get('bp_conv',    [0.35]))),
                    'return_1st': float(np.mean(s.get('return_1st', [0.30])))}
        ra_w = get_ra(w); ra_l = get_ra(l)

        # --- Home advantage ---
        home_w = 1 if row.get('winner_ioc') == row.get('tourney_ioc') else 0
        home_l = 1 if row.get('loser_ioc')  == row.get('tourney_ioc') else 0
        lev_w  = LEVEL_MULT.get(str(row.get('tourney_level', '')), 1.0)

        # --- Opponent Quality Score ---
        def _oq_score(history, my_rank):
            if not history: return 0.0
            total = 0.0; r_me = max(1.0, float(my_rank))
            for result, r_opp in history:
                r_opp = max(1.0, float(r_opp))
                contrib = np.log1p(r_opp) / np.log1p(r_me) if result == 1 \
                          else -(np.log1p(r_me) / np.log1p(r_opp))
                total += float(np.clip(contrib, -3.0, 3.0))
            return total / len(history)
        oq_hist_w = opp_quality_t.get(w, [])
        oq_hist_l = opp_quality_t.get(l, [])
        oq_w = _oq_score(oq_hist_w, rk_w)
        oq_l = _oq_score(oq_hist_l, rk_l)
        # (aggiornamento dopo lettura)

        # ── Costruzione vettore feature v6 (32 feature) ──────────────────────
        diffs = {
            'log_rank_ratio':        np.log1p(rk_l)    - np.log1p(rk_w),
            'log_pts_ratio':         np.log1p(pts_w)   - np.log1p(pts_l),
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

        # Chiavi simmetriche: non si negano per d0
        _SYMM_KEYS = ('surface_enc', 'tourney_level', 'round_enc', 'is_best_of_5',
                      'indoor', 'level_weight')

        d1 = diffs.copy()
        d1['target']      = 1
        d1['total_games'] = total_games
        d1['y_sets']      = y_sets
        d1['match_id']    = idx          # per grouping (già risolto da split temporale)
        d1['tourney_date'] = float(td)

        d0 = {k: -v if k not in _SYMM_KEYS else v for k, v in diffs.items()}
        d0['target']      = 0
        d0['total_games'] = total_games
        d0['y_sets']      = y_sets
        d0['match_id']    = idx
        d0['tourney_date'] = float(td)

        rows.append(d1)
        rows.append(d0)

        # ── Aggiornamenti DOPO la lettura (garantisce no-leakage) ─────────
        fatiga_t[(tid, w)] = f_w + dur
        fatiga_t[(tid, l)] = f_l + dur

        hw.append(1); hl.append(0)
        if len(hw) > 10: hw.pop(0)
        if len(hl) > 10: hl.pop(0)
        racha_t[(w, surf)] = hw; racha_t[(l, surf)] = hl

        if w == p1k: rec[0] += 1
        else:        rec[1] += 1
        h2h_t[key] = rec

        if w == p1k: rec_s[0] += 1
        else:        rec_s[1] += 1
        h2h_surf_t[key_s] = rec_s

        last_match_date_t[w] = td; last_match_date_t[l] = td

        ml_w = match_load_t.setdefault(w, []); ml_w.append(td_days)
        if len(ml_w) > 80: ml_w.pop(0)
        ml_l = match_load_t.setdefault(l, []); ml_l.append(td_days)
        if len(ml_l) > 80: ml_l.pop(0)

        # Aggiorna Glicko-2 superficie e overall (post-match, no leakage)
        store_s.update_match(w, l)
        g2_gen.update_match(w, l)

        rf_w.append(1.0); rf_l.append(0.0)
        if len(rf_w) > 10: rf_w.pop(0)
        if len(rf_l) > 10: rf_l.pop(0)
        recent_form_t[w] = rf_w; recent_form_t[l] = rf_l

        streak_t[w] = (max(0, str_w) + 1) if str_w >= 0 else 1
        streak_t[l] = (min(0, str_l) - 1) if str_l <= 0 else -1

        cm_w_hist.append(is_3set); cm_l_hist.append(is_3set)
        if len(cm_w_hist) > 10: cm_w_hist.pop(0)
        if len(cm_l_hist) > 10: cm_l_hist.pop(0)
        close_match_t[w] = cm_w_hist; close_match_t[l] = cm_l_hist

        uph_w.append((1, rk_l - rk_w)); uph_l.append((0, rk_w - rk_l))
        if len(uph_w) > 20: uph_w.pop(0)
        if len(uph_l) > 20: uph_l.pop(0)
        upset_hist_t[w] = uph_w; upset_hist_t[l] = uph_l

        if row.get('is_late_round', 0):
            lr_w.append(1.0); lr_l.append(0.0)
            if len(lr_w) > 30: lr_w.pop(0)
            if len(lr_l) > 30: lr_l.pop(0)
            late_round_t[w] = lr_w; late_round_t[l] = lr_l

        opp_quality_t[w] = (oq_hist_w + [(1, rk_l)])[-5:]
        opp_quality_t[l] = (oq_hist_l + [(0, rk_w)])[-5:]

        # Serve / Return stats
        def upd_serve(player, rd_row, pref):
            svpt = rd_row.get(f'{pref}_svpt', np.nan)
            if not svpt or pd.isna(svpt) or float(svpt) <= 0:
                return
            s    = serve_t.setdefault(player, {})
            fi   = rd_row.get(f'{pref}_1stIn', np.nan)
            fw   = rd_row.get(f'{pref}_1stWon', np.nan)
            sw   = rd_row.get(f'{pref}_2ndWon', np.nan)
            bps  = rd_row.get(f'{pref}_bpSaved', np.nan)
            bpf  = rd_row.get(f'{pref}_bpFaced', np.nan)
            svpt_f = float(svpt)
            for k2, v in [
                    ('ace',      rd_row.get(f'{pref}_ace', np.nan)),
                    ('1st_pct',  float(fi)/svpt_f if fi and not pd.isna(fi) else np.nan),
                    ('1st_won',  float(fw)/float(fi) if fi and float(fi) > 0 and not pd.isna(fw) else np.nan),
                    ('2nd_won',  float(sw)/(svpt_f - float(fi)) if fi and (svpt_f - float(fi)) > 0 and not pd.isna(sw) else np.nan),
                    ('bp_saved', float(bps)/float(bpf) if bpf and float(bpf) > 0 and not pd.isna(bps) else np.nan)]:
                if v is not np.nan and not (isinstance(v, float) and np.isnan(v)):
                    lst = s.setdefault(k2, []); lst.append(float(v))
                    if len(lst) > 10: lst.pop(0)

        def upd_return(player, rd, opp_pref):
            s = return_t.setdefault(player, {})
            svpt    = rd.get(f'{opp_pref}_svpt', np.nan)
            fw      = rd.get(f'{opp_pref}_1stWon', np.nan)
            sw      = rd.get(f'{opp_pref}_2ndWon', np.nan)
            fi      = rd.get(f'{opp_pref}_1stIn', np.nan)
            bpFaced = rd.get(f'{opp_pref}_bpFaced', np.nan)
            bpSaved = rd.get(f'{opp_pref}_bpSaved', np.nan)
            if svpt and svpt > 0 and not np.isnan(float(svpt)):
                rtn = (float(svpt) - _safe_float(fw) - _safe_float(sw)) / float(svpt)
                lst = s.setdefault('return_pct', []); lst.append(rtn)
                if len(lst) > 10: lst.pop(0)
            bpf = _safe_float(bpFaced)
            if bpf > 0:
                lst = s.setdefault('bp_conv', [])
                lst.append((bpf - _safe_float(bpSaved)) / bpf)
                if len(lst) > 10: lst.pop(0)
            fi_v = _safe_float(fi)
            if fi_v > 0:
                lst = s.setdefault('return_1st', [])
                lst.append((fi_v - _safe_float(fw)) / fi_v)
                if len(lst) > 10: lst.pop(0)

        rd = row.to_dict()
        upd_serve(w, rd, 'w'); upd_return(w, rd, 'l')
        upd_serve(l, rd, 'l'); upd_return(l, rd, 'w')

        # [BUG1 FIX] Win-rate aggiornato dopo lettura
        r_w = wrate_running.setdefault((w, surf), [0, 0]); r_w[0] += 1; r_w[1] += 1
        r_l = wrate_running.setdefault((l, surf), [0, 0]); r_l[1] += 1

    df_out = pd.DataFrame(rows)
    print(f"   → Dataset v5: {len(df_out):,} righe | {df_out.shape[1]} colonne")

    # ── Salva stati per predictor ────────────────────────────────────────────
    wrate_final = {k: v[0]/v[1] if v[1] >= 5 else 0.5 for k, v in wrate_running.items()}
    joblib.dump(wrate_final,       'stats_superficie_v2.pkl');  print("   → stats_superficie_v2.pkl")
    glicko2_surf = {
        'Hard': g2_hard.to_dict(), 'Clay': g2_clay.to_dict(),
        'Grass': g2_grass.to_dict(), 'Gen': g2_gen.to_dict()
    }
    joblib.dump(glicko2_surf,  'glicko2_stores.pkl'); print("   → glicko2_stores.pkl")
    joblib.dump(serve_t,       'serve_stats.pkl');    print("   → serve_stats.pkl")
    joblib.dump(return_t,      'return_stats.pkl');   print("   → return_stats.pkl")
    joblib.dump(streak_t,          'streak_players.pkl');        print("   → streak_players.pkl")
    joblib.dump(racha_t,           'momentum_surface.pkl');      print("   → momentum_surface.pkl")
    joblib.dump(recent_form_t,     'recent_form.pkl');           print("   → recent_form.pkl")
    joblib.dump(h2h_surf_t,        'h2h_surface.pkl');           print("   → h2h_surface.pkl")
    joblib.dump(opp_quality_t,     'opp_quality.pkl');           print("   → opp_quality.pkl")
    joblib.dump(last_match_date_t, 'last_match_date.pkl');       print("   → last_match_date.pkl")
    joblib.dump(close_match_t,     'close_match_hist.pkl');      print("   → close_match_hist.pkl")
    joblib.dump(upset_hist_t,      'upset_hist.pkl');            print("   → upset_hist.pkl")
    joblib.dump(late_round_t,      'late_round_hist.pkl');       print("   → late_round_hist.pkl")
    joblib.dump(match_load_t,      'match_load.pkl');            print("   → match_load.pkl")

    return df_out, wrate_final, glicko2_surf, streak_t


# ── Lista feature v5.1 (30 feature) ──────────────────────────────────────────
# Rimosse v4: diff_age, diff_ht, tourney_date (bassa importanza)
# Aggiunte v5: diff_weeks_load, diff_upset_tendency, diff_late_round_wr
# Rimosse v5.1 (correlazione ~0 dal quick test):
#   court_ace_pct, court_speed  → court_speed_helper restituisce 0 per tutti i campi
#   diff_form_volatility        → corr=0.011 (rumore)
#   diff_surface_trend          → corr=0.029 (segnale troppo debole)
#   diff_close_match_pct        → corr=0.030 (segnale troppo debole)
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


# ══════════════════════════════════════════════════════════════════════════════
# 2.  PESI TEMPORALI
# ══════════════════════════════════════════════════════════════════════════════

def calcola_pesi_temporali(date_series: pd.Series, lambda_decay: float = 0.003) -> np.ndarray:
    max_date = date_series.max()
    days_ago = (max_date - date_series) / 10000 * 365
    weights  = np.exp(-lambda_decay * days_ago)
    weights  = weights / weights.sum() * len(weights)
    return np.asarray(weights).astype(np.float32)


def calcola_pesi_combinati(date_series: pd.Series, level_weights: np.ndarray,
                           lambda_decay: float = 0.003) -> np.ndarray:
    temporal = calcola_pesi_temporali(date_series, lambda_decay)
    combined = temporal * level_weights
    combined = combined / combined.sum() * len(combined)
    return combined.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  ARCHITETTURA ANN
# ══════════════════════════════════════════════════════════════════════════════

ARCH_OPTIONS = {
    '1L_64':   [64],
    '1L_128':  [128],
    '1L_256':  [256],
    '2L_s':    [128, 64],
    '2L_m':    [256, 128],
    '2L_l':    [512, 256],
    '3L_m':    [256, 128, 64],
    '3L_l':    [512, 256, 128],
    '4L_l':    [512, 256, 128, 64],
    '2L_eq_s': [128, 128],
    '2L_eq_m': [256, 256],
    '3L_s':    [128, 64, 32],
    '4L_m':    [256, 128, 64, 32],
}

# Indici v6.0 (32 feature):
# 0=log_rank, 1=log_pts, 2=glicko, 3=glicko_ov, 4=glicko_rd, 5=streak, 6=form
# 7=surface, 8=level, 9=round, 10=bo5, 11=indoor, 12=h2h, 13=h2h_surf
# 14=skill, 15=momentum, 16=fatigue, 17=days_since, 18=weeks_load
# 19=ace, 20=1st_pct, 21=1st_won, 22=2nd_won, 23=bp_saved
# 24=return_pct, 25=bp_conv, 26=return_1st, 27=home, 28=opp_quality
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


class TennisANNv3(nn.Module):
    """Wide & Deep con Feature Interaction e Residual Connections."""
    def __init__(self, input_dim: int, hidden_layers: list, dropout: float = 0.3,
                 n_interactions: int = N_INTERACTIONS, interaction_pairs: list = None):
        super().__init__()
        self.interaction_pairs = interaction_pairs or DEFAULT_INTERACTION_PAIRS
        self.n_interactions = min(n_interactions, len(self.interaction_pairs))

        self.wide = nn.Linear(input_dim, 1)

        interaction_dim = input_dim + self.n_interactions
        self.deep_layers    = nn.ModuleList()
        self.deep_norms     = nn.ModuleList()
        self.deep_drops     = nn.ModuleList()
        self.residual_projs = nn.ModuleList()

        prev = interaction_dim
        for h in hidden_layers:
            self.deep_layers.append(nn.Linear(prev, h))
            self.deep_norms.append(nn.BatchNorm1d(h))
            self.deep_drops.append(nn.Dropout(dropout))
            self.residual_projs.append(nn.Linear(prev, h) if prev != h else nn.Identity())
            prev = h

        self.deep_out = nn.Linear(prev, 1)

    def forward(self, x):
        wide_out = self.wide(x)

        interactions = []
        for i, j in self.interaction_pairs[:self.n_interactions]:
            if i < x.shape[1] and j < x.shape[1]:
                interactions.append(x[:, i] * x[:, j])
            else:
                interactions.append(torch.zeros(x.shape[0], device=x.device))
        inter_t = torch.stack(interactions, dim=1)
        deep_in = torch.cat([x, inter_t], dim=1)

        h = deep_in
        for layer, norm, drop, res_proj in zip(
                self.deep_layers, self.deep_norms, self.deep_drops, self.residual_projs):
            identity = res_proj(h)
            h = layer(h); h = norm(h); h = torch.relu(h); h = drop(h)
            h = h + identity

        deep_out = self.deep_out(h)
        return (wide_out + deep_out).squeeze(1)


def label_smoothed_bce(logits, targets, smoothing=0.05):
    targets_smooth = targets * (1.0 - smoothing) + 0.5 * smoothing
    return nn.functional.binary_cross_entropy_with_logits(logits, targets_smooth)


def train_model(model, loader_train, loader_val, epochs=80, lr=1e-3, patience=10, smoothing=0.05):
    model.to(device)
    optimizer  = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=4)
    criterion_val = nn.BCEWithLogitsLoss()
    best_val = float('inf'); best_state = None; no_imp = 0

    for _ in range(epochs):
        model.train()
        for batch in loader_train:
            Xb, yb = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            loss = label_smoothed_bce(model(Xb), yb, smoothing)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval(); vl = []
        with torch.no_grad():
            for batch in loader_val:
                Xb, yb = batch[0].to(device), batch[1].to(device)
                vl.append(criterion_val(model(Xb), yb).item())
        vl_mean = float(np.mean(vl)); scheduler.step(vl_mean)
        if vl_mean < best_val - 1e-5:
            best_val = vl_mean
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience: break
    if best_state: model.load_state_dict(best_state)
    return model, best_val


def valuta(model, Xsc, y_np):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(Xsc.astype(np.float32)).to(device)).cpu().numpy()
    probs = 1 / (1 + np.exp(-logits))
    acc = accuracy_score(y_np, (probs >= 0.5).astype(int))
    ll  = log_loss(y_np, probs)
    return acc, ll


# ══════════════════════════════════════════════════════════════════════════════
# 4.  OPTUNA SEARCH
# ══════════════════════════════════════════════════════════════════════════════

def optuna_search(X_tr, y_tr, X_val_sc, X_te_sc,
                  y_val, y_test, dates_tr, level_w_tr,
                  n_trials=TRIALS, input_dim=len(FEATURES), label="Global"):

    X_val_t   = torch.tensor(X_val_sc.astype(np.float32))
    y_val_np  = y_val.values  if hasattr(y_val,  'values') else np.array(y_val)
    y_test_np = y_test.values if hasattr(y_test, 'values') else np.array(y_test)

    X_tr_t = torch.tensor(X_tr.astype(np.float32))
    y_tr_t = torch.tensor(y_tr.values.astype(np.float32) if hasattr(y_tr, 'values')
                          else y_tr.astype(np.float32))

    dataset_tr  = TensorDataset(X_tr_t, y_tr_t)
    dataset_val = TensorDataset(X_val_t, torch.tensor(y_val_np.astype(np.float32)))

    USE_CUDA = device.type == 'cuda'; PIN_MEM = USE_CUDA
    risultati = []

    if HAS_OPTUNA:
        print(f"\n🔬 Optuna search [{label}]: {n_trials} trial...\n")

        def objective(trial):
            arch_name = trial.suggest_categorical('arch', list(ARCH_OPTIONS.keys()))
            hl   = ARCH_OPTIONS[arch_name]
            dr   = trial.suggest_float('dropout', 0.1, 0.5, step=0.05)
            lr   = trial.suggest_float('lr', 1e-4, 3e-3, log=True)
            bs   = trial.suggest_categorical('batch_size', [256, 512, 1024, 2048])
            ep   = min(MAX_EPOCHS, trial.suggest_categorical('epochs', [60, 80, 100, 120]))
            lam  = trial.suggest_categorical('lambda_decay', [0.0001, 0.001, 0.003, 0.007, 0.015])
            sm   = trial.suggest_float('smoothing', 0.0, 0.1, step=0.01)
            iset = trial.suggest_categorical('interaction_set', list(INTERACTION_SETS.keys()))
            i_pairs = INTERACTION_SETS[iset]; n_inter = len(i_pairs)

            print(f"  [{trial.number+1:3d}/{n_trials}] arch={arch_name:12s} "
                  f"dr={dr:.2f} lr={lr:.0e} λ={lam:.4f} sm={sm:.2f} int={iset:14s}  "
                  f"▶ training...", flush=True)

            bs_eff = bs * 2 if USE_CUDA and bs < 2048 else bs
            w_combined = calcola_pesi_combinati(dates_tr, level_w_tr, lam)
            w_t = torch.tensor(w_combined)
            sampler = WeightedRandomSampler(w_t, len(w_t), replacement=True)
            ldr_tr  = DataLoader(dataset_tr,  batch_size=bs_eff, sampler=sampler,
                                 pin_memory=PIN_MEM, num_workers=N_WORKERS)
            ldr_val = DataLoader(dataset_val, batch_size=4096, shuffle=False,
                                 pin_memory=PIN_MEM, num_workers=N_WORKERS)

            model = TennisANNv3(input_dim, hl, dr, n_inter, i_pairs)
            model, _ = train_model(model, ldr_tr, ldr_val, epochs=ep, lr=lr, smoothing=sm)
            _, ll_v = valuta(model, X_val_sc, y_val_np)
            trial.set_user_attr('model', model)
            trial.set_user_attr('hl', hl)
            trial.set_user_attr('hp', {'hidden_layers': hl, 'dropout': dr, 'lr': lr,
                                       'batch_size': bs, 'epochs': ep, 'lambda_decay': lam,
                                       'label_smoothing': sm, 'interaction_set': iset,
                                       'interaction_pairs': i_pairs, 'n_interactions': n_inter})
            return ll_v

        study = optuna.create_study(direction='minimize',
                                    sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(objective, n_trials=n_trials,
                       callbacks=[lambda s, t: print(
                           f"     └─ val_score={t.value:.4f}  "
                           f"{'BEST! ⭐' if t.number == s.best_trial.number else ''}",
                           flush=True) if t.value is not None else None])

        for t in study.trials:
            if t.value is None: continue
            model_t = t.user_attrs.get('model')
            if model_t is None: continue
            acc_t, ll = valuta(model_t, X_te_sc, y_test_np)
            risultati.append({
                'trial':           t.number + 1,
                'hidden_layers':   str(t.user_attrs.get('hl', [])),
                'dropout':         t.params.get('dropout'),
                'lr':              t.params.get('lr'),
                'batch_size':      t.params.get('batch_size'),
                'epochs':          t.params.get('epochs'),
                'lambda_decay':    t.params.get('lambda_decay'),
                'smoothing':       t.params.get('smoothing'),
                'interaction_set': t.params.get('interaction_set'),
                'val_acc':         t.value,
                'test_acc':        acc_t,
                'log_loss':        ll,
                '_model':          model_t,
                '_config':         t.user_attrs.get('hp', {}),
            })

    else:
        print(f"\n🔍 Random search [{label}]: {n_trials} trial...\n")
        for i in range(n_trials):
            arch_name = random.choice(list(ARCH_OPTIONS.keys()))
            hl   = ARCH_OPTIONS[arch_name]
            dr   = random.choice([0.1, 0.2, 0.3, 0.4])
            lr   = random.choice([1e-4, 3e-4, 1e-3, 3e-3])
            bs   = random.choice([256, 512, 1024, 2048])
            ep   = min(MAX_EPOCHS, random.choice([60, 80, 100]))
            lam  = random.choice([0.0001, 0.001, 0.003, 0.007, 0.015])
            sm   = random.choice([0.0, 0.02, 0.05, 0.08, 0.1])
            iset = random.choice(list(INTERACTION_SETS.keys()))
            i_pairs = INTERACTION_SETS[iset]; n_inter = len(i_pairs)
            hp = {'hidden_layers': hl, 'dropout': dr, 'lr': lr, 'batch_size': bs,
                  'epochs': ep, 'lambda_decay': lam, 'label_smoothing': sm,
                  'interaction_set': iset, 'interaction_pairs': i_pairs, 'n_interactions': n_inter}
            print(f"  [{i+1:3d}/{n_trials}] {str(hl):22s} drop={dr} lr={lr:.0e} λ={lam:.4f} sm={sm:.2f} int={iset}  ▶", flush=True)
            w_combined = calcola_pesi_combinati(dates_tr, level_w_tr, lam)
            w_t = torch.tensor(w_combined)
            sampler = WeightedRandomSampler(w_t, len(w_t), replacement=True)
            ldr_tr  = DataLoader(dataset_tr,  batch_size=bs, sampler=sampler)
            ldr_val = DataLoader(dataset_val, batch_size=2048, shuffle=False)
            model = TennisANNv3(input_dim, hl, dr, n_inter, i_pairs)
            model, _ = train_model(model, ldr_tr, ldr_val, epochs=ep, lr=lr, smoothing=sm)
            acc_v, ll_v = valuta(model, X_val_sc, y_val_np)
            acc_t, ll = valuta(model, X_te_sc, y_test_np)
            score_v = acc_v - ll_v
            print(f"     └─ val_score={score_v:.3f} test={acc_t:.3f}", flush=True)
            risultati.append({'trial': i+1, 'hidden_layers': str(hl), 'dropout': dr, 'lr': lr,
                              'batch_size': bs, 'epochs': ep, 'lambda_decay': lam, 'smoothing': sm,
                              'interaction_set': iset, 'val_acc': acc_v, 'test_acc': acc_t,
                              'log_loss': ll, '_model': model, '_config': hp})

    return sorted(risultati, key=lambda x: x['test_acc'], reverse=True)


# ══════════════════════════════════════════════════════════════════════════════
# 4b.  LIGHTGBM
# ══════════════════════════════════════════════════════════════════════════════

def train_lgb_optuna(X_tr, y_tr, X_val, y_val, X_test, y_test,
                     sample_weights_tr=None, n_trials=TRIALS_GBM):
    if not HAS_LGB:
        print("   ⚠️ LightGBM non disponibile, skip")
        return None, 0.0, float('inf')

    print(f"\n🌳 Training LightGBM (Optuna {n_trials} trials)...")
    y_tr_np   = y_tr.values  if hasattr(y_tr,   'values') else np.array(y_tr)
    y_val_np  = y_val.values if hasattr(y_val,  'values') else np.array(y_val)
    y_test_np = y_test.values if hasattr(y_test,'values') else np.array(y_test)

    best_model = [None]; best_acc = [float('inf')]

    def objective(trial):
        params = {
            'objective': 'binary', 'metric': 'binary_logloss',
            'boosting_type': 'gbdt', 'verbosity': -1, 'seed': SEED,
            'n_estimators':     trial.suggest_int('n_estimators', 200, 1500),
            'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'max_depth':        trial.suggest_int('max_depth', 3, 10),
            'num_leaves':       trial.suggest_int('num_leaves', 20, 150),
            'min_child_samples':trial.suggest_int('min_child_samples', 5, 100),
            'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha':        trial.suggest_float('reg_alpha', 1e-8, 10, log=True),
            'reg_lambda':       trial.suggest_float('reg_lambda', 1e-8, 10, log=True),
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr, y_tr_np, sample_weight=sample_weights_tr,
                  eval_set=[(X_val, y_val_np)],
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=0)])
        probs = model.predict_proba(X_val)[:, 1]
        ll = log_loss(y_val_np, probs)
        if ll < best_acc[0]: best_acc[0] = ll; best_model[0] = model
        return ll

    if HAS_OPTUNA:
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        model = best_model[0]
        print(f"   LightGBM best val log-loss: {best_acc[0]:.4f}")
    else:
        model = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, seed=SEED)
        model.fit(X_tr, y_tr_np, sample_weight=sample_weights_tr, eval_set=[(X_val, y_val_np)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])

    y_pred = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test_np, (y_pred >= 0.5).astype(int))
    ll  = log_loss(y_test_np, y_pred)
    print(f"   LightGBM test: acc={acc:.4f} | log_loss={ll:.4f}")

    # Feature importance LightGBM
    if hasattr(model, 'feature_importances_'):
        fi = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
        print("\n   📊 LightGBM Feature Importance (top 15):")
        for feat, val in fi.head(15).items():
            print(f"      {feat:35s} {val:8.0f}")

    return model, acc, ll


# ══════════════════════════════════════════════════════════════════════════════
# 4c.  XGBOOST
# ══════════════════════════════════════════════════════════════════════════════

def train_xgb_optuna(X_tr, y_tr, X_val, y_val, X_test, y_test,
                     sample_weights_tr=None, n_trials=TRIALS_GBM):
    if not HAS_XGB:
        print("   ⚠️ XGBoost non disponibile, skip")
        return None, 0.0, float('inf')

    print(f"\n🌲 Training XGBoost (Optuna {n_trials} trials)...")
    y_tr_np   = y_tr.values  if hasattr(y_tr,   'values') else np.array(y_tr)
    y_val_np  = y_val.values if hasattr(y_val,  'values') else np.array(y_val)
    y_test_np = y_test.values if hasattr(y_test,'values') else np.array(y_test)

    best_model = [None]; best_acc = [float('inf')]

    def objective(trial):
        params = {
            'objective': 'binary:logistic', 'eval_metric': 'logloss',
            'seed': SEED, 'verbosity': 0,
            'n_estimators':     trial.suggest_int('n_estimators', 200, 1500),
            'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'max_depth':        trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 50),
            'subsample':        trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha':        trial.suggest_float('reg_alpha', 1e-8, 10, log=True),
            'reg_lambda':       trial.suggest_float('reg_lambda', 1e-8, 10, log=True),
            'gamma':            trial.suggest_float('gamma', 0, 5),
        }
        model = xgb_lib.XGBClassifier(**params)
        model.fit(X_tr, y_tr_np, sample_weight=sample_weights_tr,
                  eval_set=[(X_val, y_val_np)], verbose=False)
        probs = model.predict_proba(X_val)[:, 1]
        ll   = log_loss(y_val_np, probs)
        if ll < best_acc[0]: best_acc[0] = ll; best_model[0] = model
        return ll

    if HAS_OPTUNA:
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        model = best_model[0]
        print(f"   XGBoost best val log-loss: {best_acc[0]:.4f}")
    else:
        model = xgb_lib.XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, seed=SEED)
        model.fit(X_tr, y_tr_np, sample_weight=sample_weights_tr,
                  eval_set=[(X_val, y_val_np)], verbose=False)

    y_pred = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test_np, (y_pred >= 0.5).astype(int))
    ll  = log_loss(y_test_np, y_pred)
    print(f"   XGBoost test: acc={acc:.4f} | log_loss={ll:.4f}")
    return model, acc, ll



def train_lgb_surface_specific(X_tr_df, y_tr, X_val_df, y_val,
                                X_test_df, y_test,
                                sample_weights_tr=None, n_trials=None):
    """Allena un LightGBM separato per Hard, Clay, Grass."""
    if not HAS_LGB:
        print("   LightGBM non disponibile, skip surface models")
        return {}, {}
    if n_trials is None:
        n_trials = TRIALS_GBM

    SURFACES = {0: 'Hard', 1: 'Clay', 2: 'Grass'}
    models = {}
    accs   = {}

    for surf_enc, surf_name in SURFACES.items():
        mask_tr   = X_tr_df['surface_enc'] == surf_enc
        mask_val  = X_val_df['surface_enc'] == surf_enc
        mask_test = X_test_df['surface_enc'] == surf_enc

        n_tr   = mask_tr.sum()
        n_val  = mask_val.sum()
        n_test = mask_test.sum()
        if n_tr < 200:
            print(f"   {surf_name}: solo {n_tr} esempi train — skip")
            continue
        if n_val < 10 or n_test < 10:
            print(f"   {surf_name}: val={n_val} / test={n_test} troppo piccoli — skip")
            continue

        X_tr_s  = X_tr_df[mask_tr][FEATURES].fillna(0).values
        y_tr_s  = y_tr[mask_tr].values if hasattr(y_tr[mask_tr], 'values') else y_tr[mask_tr]
        X_val_s = X_val_df[mask_val][FEATURES].fillna(0).values
        y_val_s = y_val[mask_val].values if hasattr(y_val[mask_val], 'values') else y_val[mask_val]
        X_te_s  = X_test_df[mask_test][FEATURES].fillna(0).values
        y_te_s  = y_test[mask_test].values if hasattr(y_test[mask_test], 'values') else y_test[mask_test]
        w_s     = sample_weights_tr[mask_tr] if sample_weights_tr is not None else None

        print(f"\n  LGB Surface: {surf_name} ({n_tr:,} train | {mask_val.sum():,} val)")
        model, acc, ll = train_lgb_optuna(
            X_tr_s, y_tr_s, X_val_s, y_val_s, X_te_s, y_te_s,
            sample_weights_tr=w_s, n_trials=n_trials
        )
        if model is not None:
            models[surf_name] = model
            accs[surf_name]   = acc
            print(f"    -> {surf_name}: acc={acc:.4f} | log_loss={ll:.4f}")

    return models, accs


def train_ann_surface_specific(df_tr, y_tr, X_tr_sc,
                                df_val, y_val, X_val_sc,
                                df_test, y_test, X_te_sc,
                                n_trials=None):
    """Allena un'ANN separata per Hard, Clay, Grass con Optuna."""
    if n_trials is None:
        n_trials = TRIALS

    SURFACES = {0: 'Hard', 1: 'Clay', 2: 'Grass'}
    ann_surf_results = {}

    y_tr_np   = y_tr.values   if hasattr(y_tr,   'values') else np.array(y_tr)
    y_val_np  = y_val.values  if hasattr(y_val,  'values') else np.array(y_val)
    y_te_np   = y_test.values if hasattr(y_test, 'values') else np.array(y_test)
    d_tr_arr  = df_tr['tourney_date'].values
    lw_tr_arr = df_tr['level_weight'].values

    for surf_enc, surf_name in SURFACES.items():
        mask_tr   = (df_tr['surface_enc'].values   == surf_enc)
        mask_val  = (df_val['surface_enc'].values  == surf_enc)
        mask_test = (df_test['surface_enc'].values == surf_enc)

        n_tr, n_val, n_test = mask_tr.sum(), mask_val.sum(), mask_test.sum()
        # Soglia più alta per evitare overfitting su superfici con pochi dati (es. Grass ~5k)
        min_tr = 8000
        if n_tr < min_tr:
            print(f"   {surf_name}: {n_tr} esempi train < {min_tr} — skip (fallback ANN globale)")
            continue
        if n_val < 20 or n_test < 10:
            print(f"   {surf_name}: val={n_val} / test={n_test} troppo piccoli — skip")
            continue

        print(f"\n  ANN Surface: {surf_name} ({n_tr:,} train | {n_val:,} val | {n_test:,} test)")
        risultati_s = optuna_search(
            X_tr_sc[mask_tr],
            pd.Series(y_tr_np[mask_tr]),
            X_val_sc[mask_val],
            X_te_sc[mask_test],
            pd.Series(y_val_np[mask_val]),
            pd.Series(y_te_np[mask_test]),
            pd.Series(d_tr_arr[mask_tr]),
            lw_tr_arr[mask_tr],
            n_trials=n_trials,
            label=surf_name,
        )
        if risultati_s:
            best_s = risultati_s[0]
            acc_s, ll_s = valuta(best_s['_model'], X_te_sc[mask_test], y_te_np[mask_test])
            print(f"    -> {surf_name}: acc={acc_s:.4f} | log_loss={ll_s:.4f}")
            ann_surf_results[surf_name] = risultati_s

    return ann_surf_results


# ══════════════════════════════════════════════════════════════════════════════
# 5.  ENSEMBLE UTILS
# ══════════════════════════════════════════════════════════════════════════════

def get_ann_probs(model, X_sc):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_sc.astype(np.float32)).to(device)).cpu().numpy()
    return 1 / (1 + np.exp(-logits))


def get_model_probs(strategy, X_sc, ann_model=None, lgb_model=None, xgb_model=None,
                    top5_models=None, meta_model=None):
    if strategy == 'ann_best':
        return get_ann_probs(ann_model, X_sc)
    elif strategy == 'ann_top5' and top5_models:
        return np.mean([get_ann_probs(m, X_sc).flatten() for m in top5_models], axis=0).reshape(-1, 1)
    elif strategy == 'lgb' and lgb_model is not None:
        return lgb_model.predict_proba(X_sc)[:, 1].reshape(-1, 1)
    elif strategy == 'xgb' and xgb_model is not None:
        return xgb_model.predict_proba(X_sc)[:, 1].reshape(-1, 1)
    elif strategy == 'ensemble_avg':
        probs = []
        if ann_model:  probs.append(get_ann_probs(ann_model, X_sc).flatten())
        if lgb_model:  probs.append(lgb_model.predict_proba(X_sc)[:, 1])
        if xgb_model:  probs.append(xgb_model.predict_proba(X_sc)[:, 1])
        return np.mean(probs, axis=0).reshape(-1, 1) if probs else get_ann_probs(ann_model, X_sc)
    elif strategy == 'ensemble_avg_top5':
        probs = []
        if top5_models: probs.append(np.mean([get_ann_probs(m, X_sc).flatten() for m in top5_models], axis=0))
        if lgb_model:   probs.append(lgb_model.predict_proba(X_sc)[:, 1])
        if xgb_model:   probs.append(xgb_model.predict_proba(X_sc)[:, 1])
        return np.mean(probs, axis=0).reshape(-1, 1) if probs else get_ann_probs(top5_models[0], X_sc)
    elif strategy == 'ensemble_stacking' and meta_model is not None:
        parts = {}
        if ann_model:  parts['ANN'] = get_ann_probs(ann_model, X_sc)
        if lgb_model:  parts['LGB'] = lgb_model.predict_proba(X_sc)[:, 1].reshape(-1, 1)
        if xgb_model:  parts['XGB'] = xgb_model.predict_proba(X_sc)[:, 1].reshape(-1, 1)
        if len(parts) >= 2:
            return meta_model.predict_proba(np.column_stack(list(parts.values())))[:, 1].reshape(-1, 1)
    return get_ann_probs(ann_model, X_sc)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  CONFRONTO FINALE
# ══════════════════════════════════════════════════════════════════════════════

def confronto_finale(results_list):
    df_final = pd.DataFrame(results_list)
    df_final['Acc. %'] = (df_final['Accuracy'] * 100).round(2)
    df_final['Score']  = (df_final['Accuracy'] - df_final['Log Loss']).round(4)
    df_final = df_final.sort_values('Acc. %', ascending=False).reset_index(drop=True)
    df_final.to_csv('resultados_comparacion_finale.csv', index=False)

    print("\n" + "="*80)
    print("  🏆  CLASSIFICA FINALE (Score = Accuracy − Log Loss)")
    print("="*80)
    print(df_final[['Modello', 'Acc. %', 'Log Loss', 'Score', 'Note']].to_string(index=False))
    print("="*80)
    vincitore = df_final.iloc[0]
    print(f"\n  🥇  Miglior modello: {vincitore['Modello']}  →  "
          f"Acc {vincitore['Acc. %']:.2f}%  |  Score {vincitore['Score']:.4f}\n")
    return df_final


# ══════════════════════════════════════════════════════════════════════════════
# 7.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    CSV_CANDIDATES = [
        'master_dataset.csv',
        '../scraping/master_dataset.csv',
        '../scraping/historialTenis.csv',
        'historialTenis.csv',
    ]
    csv_path = next((p for p in CSV_CANDIDATES if os.path.exists(p)), None)
    if csv_path is None:
        raise FileNotFoundError("historialTenis.csv non trovato. Esegui da prediccion/")

    if QUICK_TEST:
        print("\n" + "="*60)
        print("  ⚡  QUICK_TEST MODE — 3 epoche, 5 trial Optuna")
        print("="*60)

    # ── 1. Feature engineering ────────────────────────────────────────────────
    df_ml, wrate_final, glicko2_data, streak_t = carica_e_prepara(csv_path)

    # ── 1b. Filtro anno minimo + escludi Challenger dal training ML ──────────
    n_before = len(df_ml)
    if ML_MIN_YEAR:
        df_ml = df_ml[df_ml['tourney_date'] >= ML_MIN_YEAR * 10000].copy()
    df_ml = df_ml[df_ml['level_weight'] != 0.8].copy()
    print(f"   → Filtro ML_MIN_YEAR={ML_MIN_YEAR} + Challenger esclusi: "
          f"{n_before - len(df_ml):,} righe rimosse | {len(df_ml):,} righe rimaste")

    # ── 2. [BUG2+BUG3 FIX] Temporal split per data cronologica ───────────────
    # Mirrored pairs (d1, d0) dello stesso match hanno la stessa data
    # → finiscono automaticamente nello stesso fold (no leakage)
    df_ml = df_ml.sort_values('tourney_date').reset_index(drop=True)

    unique_dates = np.sort(df_ml['tourney_date'].dropna().unique())
    n_dates      = len(unique_dates)
    cutoff_val   = unique_dates[int(n_dates * 0.70)]
    cutoff_test  = unique_dates[int(n_dates * 0.85)]

    print(f"\n📅 Split temporale (nessun leakage futuro):")
    print(f"   Train : date < {int(cutoff_val)}")
    print(f"   Val   : {int(cutoff_val)} ≤ date < {int(cutoff_test)}")
    print(f"   Test  : date ≥ {int(cutoff_test)}")

    mask_tr   = df_ml['tourney_date'] <  cutoff_val
    mask_val  = (df_ml['tourney_date'] >= cutoff_val) & (df_ml['tourney_date'] < cutoff_test)
    mask_test = df_ml['tourney_date'] >= cutoff_test

    df_tr   = df_ml[mask_tr]
    df_val  = df_ml[mask_val]
    df_test = df_ml[mask_test]

    X_tr    = df_tr[FEATURES].fillna(0)
    y_tr    = df_tr['target']
    d_tr    = df_tr['tourney_date']
    lw_tr   = df_tr['level_weight'].values

    X_val   = df_val[FEATURES].fillna(0)
    y_val   = df_val['target']

    X_test  = df_test[FEATURES].fillna(0)
    y_test  = df_test['target']

    print(f"   → Train: {len(X_tr):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    # ── QUICK_TEST: feature correlation con target ────────────────────────────
    if QUICK_TEST:
        print("\n📊 Feature correlation |r| con target (train set):")
        corr = X_tr.copy()
        corr['target'] = y_tr.values
        feat_corr = corr.corr()['target'].drop('target').abs().sort_values(ascending=False)
        for feat, val in feat_corr.items():
            bar = '█' * int(val * 40) if not (val != val) else ''
            print(f"   {feat:35s} {val:.4f}  {bar}")
        print()

    # ── 3. Scaler ──────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_tr.values)
    X_val_sc = scaler.transform(X_val.values)
    X_te_sc  = scaler.transform(X_test.values)
    joblib.dump(scaler, 'scaler_ann.pkl')
    print(f"   → scaler_ann.pkl salvato  |  {len(FEATURES)} feature")

    combined_weights = calcola_pesi_combinati(d_tr, lw_tr, 0.003)

    # Peso 1.25x sulle partite trappola (upset: lower-ranked player wins)
    upset_mask = (X_tr['log_rank_ratio'].values * (2 * y_tr.values - 1)) < 0
    combined_weights[upset_mask] *= 1.25
    combined_weights = (combined_weights / combined_weights.sum() * len(combined_weights)).astype(np.float32)

    # ── 4. Optuna ANN search ──────────────────────────────────────────────────
    risultati = optuna_search(
        X_tr_sc, y_tr, X_val_sc, X_te_sc,
        y_val, y_test, d_tr, lw_tr,
        n_trials=TRIALS, input_dim=len(FEATURES), label="Global",
    )

    df_ris = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith('_')}
                            for r in risultati])
    df_ris.to_csv('resultados_ann.csv', index=False)

    best    = risultati[0]
    best_hp = best['_config']
    acc_test, ll_test = valuta(best['_model'], X_te_sc, y_test.values)

    print(f"\n🏆 Miglior ANN (Optuna):")
    print(f"   Arch: {best['hidden_layers']} | drop={best['dropout']} | lr={best['lr']:.0e} | "
          f"bs={best['batch_size']} | λ={best['lambda_decay']} | sm={best.get('smoothing','?')} | "
          f"int={best.get('interaction_set','?')}")
    print(f"   Test Accuracy: {acc_test:.2%}  |  Log Loss: {ll_test:.4f}")

    # ── 5. GBM models ─────────────────────────────────────────────────────────
    lgb_model, lgb_acc, lgb_ll = train_lgb_optuna(
        X_tr_sc, y_tr, X_val_sc, y_val, X_te_sc, y_test,
        sample_weights_tr=combined_weights, n_trials=TRIALS_GBM)

    xgb_model, xgb_acc, xgb_ll = train_xgb_optuna(
        X_tr_sc, y_tr, X_val_sc, y_val, X_te_sc, y_test,
        sample_weights_tr=combined_weights, n_trials=TRIALS_GBM)

    # ── 6. Ensemble ────────────────────────────────────────────────────────────
    from sklearn.linear_model import LogisticRegression

    y_val_np  = y_val.values  if hasattr(y_val,  'values') else np.array(y_val)
    y_test_np = y_test.values if hasattr(y_test, 'values') else np.array(y_test)

    # ── Surface-specific LGB ──────────────────────────────────────────────────
    print("\n🌍 Training LGB Surface-Specific (Hard / Clay / Grass)...")
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
        if lgb_model is not None:
            no_surf = ~df_test['surface_enc'].isin([0, 1, 2]).values
            if no_surf.sum() > 0:
                surf_preds[no_surf] = lgb_model.predict_proba(X_te_sc[no_surf])[:, 1]
        surf_acc = accuracy_score(y_test_np, (surf_preds >= 0.5).astype(int))
        surf_ll  = log_loss(y_test_np, surf_preds)
        print(f"\n   LGB Surface ensemble: acc={surf_acc:.4f} | log_loss={surf_ll:.4f}")
    else:
        surf_acc, surf_ll = 0.0, float('inf')

    # ── Surface-specific ANN ──────────────────────────────────────────────────
    print("\n🧠 Training ANN Surface-Specific (Hard / Clay / Grass)...")
    ann_surf_models = train_ann_surface_specific(
        df_tr, y_tr, X_tr_sc,
        df_val, y_val, X_val_sc,
        df_test, y_test, X_te_sc,
        n_trials=TRIALS,
    )

    # Calcola accuracy ANN surface routing sul test set
    if ann_surf_models:
        _surf_enc_map = {'Hard': 0, 'Clay': 1, 'Grass': 2}
        ann_surf_preds = get_ann_probs(best['_model'], X_te_sc).flatten()  # fallback globale
        for sname, sresults in ann_surf_models.items():
            smask = (df_test['surface_enc'].values == _surf_enc_map[sname])
            if smask.sum() > 0:
                ann_surf_preds[smask] = get_ann_probs(
                    sresults[0]['_model'], X_te_sc[smask]
                ).flatten()
        ann_surf_acc = accuracy_score(y_test_np, (ann_surf_preds >= 0.5).astype(int))
        ann_surf_ll  = log_loss(y_test_np, ann_surf_preds)
        print(f"\n   ANN Surface ensemble: acc={ann_surf_acc:.4f} | log_loss={ann_surf_ll:.4f}")
    else:
        ann_surf_acc, ann_surf_ll = 0.0, float('inf')

    best_ann = best['_model']
    ann_probs_val  = get_ann_probs(best_ann, X_val_sc)
    ann_probs_test = get_ann_probs(best_ann, X_te_sc)

    top5_ann_probs_val  = np.mean([get_ann_probs(r['_model'], X_val_sc)  for r in risultati[:5]], axis=0)
    # Salva matrice (5 x n_test) per calcolare std per-match
    _top5_matrix_test = np.array([get_ann_probs(r['_model'], X_te_sc).flatten() for r in risultati[:min(5, len(risultati))]])
    top5_ann_probs_test = np.mean(_top5_matrix_test, axis=0)
    top5_ann_std_test   = np.std(_top5_matrix_test, axis=0)   # std per-match (per filter_analysis)
    top5_ann_acc = accuracy_score(y_test_np, (top5_ann_probs_test >= 0.5).astype(int))
    top5_ann_ll  = log_loss(y_test_np, top5_ann_probs_test)
    print(f"\n   ANN Top-5 ensemble: acc={top5_ann_acc:.4f} | log_loss={top5_ann_ll:.4f}")

    # ── Salva top-5 state dicts per uncertainty estimation (sempre, indipendente da strategy) ──
    top5_for_uncertainty = [
        {'state_dict': r['_model'].state_dict(), 'config': r['_config']}
        for r in risultati[:min(5, len(risultati))]
    ]
    print(f"   → {len(top5_for_uncertainty)} modelli salvati per uncertainty estimation")

    # ── Calibrazione Temperature Scaling su VALIDATION SET (no leakage su test) ─
    from scipy.optimize import minimize_scalar
    from sklearn.calibration import calibration_curve as sk_calibration_curve
    from prediction_engine import TemperatureScaler

    raw_probs_val  = np.mean([get_ann_probs(r['_model'], X_val_sc) for r in risultati[:5]], axis=0).flatten()
    raw_probs_test = top5_ann_probs_test.flatten()

    def _nll_temp(T):
        p = np.clip(raw_probs_val, 1e-7, 1 - 1e-7)
        cal_p = 1.0 / (1.0 + np.exp(-np.log(p / (1 - p)) / T))
        return log_loss(y_val_np, cal_p)

    _res   = minimize_scalar(_nll_temp, bounds=(0.1, 10.0), method='bounded')
    T_opt  = float(_res.x)
    calibrator = TemperatureScaler(T_opt)
    calibrated_probs_test = calibrator.predict(raw_probs_test)
    ll_calibrated  = log_loss(y_test_np, calibrated_probs_test)
    acc_calibrated = accuracy_score(y_test_np, (calibrated_probs_test >= 0.5).astype(int))
    print(f"   Calibrated (temperature T={T_opt:.4f}): acc={acc_calibrated:.4f} | log_loss={ll_calibrated:.4f}")

    # Curva di calibrazione (per reliability diagram nel dashboard)
    try:
        frac_pos, mean_pred = sk_calibration_curve(y_test_np, raw_probs_test, n_bins=10, strategy='quantile')
        calibration_curve_data = {
            'fraction_pos': frac_pos.tolist(),
            'mean_pred':    mean_pred.tolist(),
        }
    except Exception:
        calibration_curve_data = {}

    joblib.dump(calibrator, 'calibrator_ann.pkl')
    print(f"   → calibrator_ann.pkl salvato")

    # ── Accuracy per superficie (su calibrated probs del test set) ────────────
    _surf_acc_map = {'Hard': 0, 'Clay': 1, 'Grass': 2}
    surface_accuracy = {}
    for sname, senc in _surf_acc_map.items():
        smask = df_test['surface_enc'].values == senc
        if smask.sum() >= 10:
            surface_accuracy[sname] = float(accuracy_score(
                y_test_np[smask], (calibrated_probs_test[smask] >= 0.5).astype(int)
            ))
    print(f"   Accuracy per superficie: " +
          " | ".join(f"{s}={v:.1%}" for s, v in surface_accuracy.items()))

    val_probs  = {'ANN': ann_probs_val}
    test_probs = {'ANN': ann_probs_test}
    val_probs_top5  = {'ANN5': top5_ann_probs_val}
    test_probs_top5 = {'ANN5': top5_ann_probs_test}

    if lgb_model is not None:
        val_probs['LGB']  = lgb_model.predict_proba(X_val_sc)[:, 1]
        test_probs['LGB'] = lgb_model.predict_proba(X_te_sc)[:, 1]
        val_probs_top5['LGB']  = val_probs['LGB']
        test_probs_top5['LGB'] = test_probs['LGB']

    if xgb_model is not None:
        val_probs['XGB']  = xgb_model.predict_proba(X_val_sc)[:, 1]
        test_probs['XGB'] = xgb_model.predict_proba(X_te_sc)[:, 1]
        val_probs_top5['XGB']  = val_probs['XGB']
        test_probs_top5['XGB'] = test_probs['XGB']

    avg_test     = np.mean(list(test_probs.values()), axis=0)
    avg_acc      = accuracy_score(y_test_np, (avg_test >= 0.5).astype(int))
    avg_ll       = log_loss(y_test_np, avg_test)
    avg_test_top5 = np.mean(list(test_probs_top5.values()), axis=0)
    avg_acc_top5  = accuracy_score(y_test_np, (avg_test_top5 >= 0.5).astype(int))
    avg_ll_top5   = log_loss(y_test_np, avg_test_top5)
    print(f"   Avg ensemble: acc={avg_acc:.4f} | Avg top5: acc={avg_acc_top5:.4f}")

    meta_model = None
    if len(val_probs) >= 2:
        val_meta  = np.column_stack(list(val_probs.values()))
        test_meta = np.column_stack(list(test_probs.values()))
        meta_model = LogisticRegression(C=1.0, random_state=SEED, max_iter=2000)
        meta_model.fit(val_meta, y_val_np)
        stack_probs = meta_model.predict_proba(test_meta)[:, 1]
        stack_acc   = accuracy_score(y_test_np, (stack_probs >= 0.5).astype(int))
        stack_ll    = log_loss(y_test_np, stack_probs)
        print(f"   Stacking ensemble: acc={stack_acc:.4f} | log_loss={stack_ll:.4f}")
        joblib.dump(meta_model, 'modelo_meta_lr.pkl')
    else:
        stack_acc, stack_ll = avg_acc, avg_ll

    # ── 7. Selezione strategia migliore ───────────────────────────────────────
    results_list = [
        {'Modello': 'ANN Best',          'Accuracy': acc_test,      'Log Loss': ll_test,
         'Note': f'arch={best["hidden_layers"]}', '_strategy': 'ann_best'},
        {'Modello': 'ANN Top-5 Avg',     'Accuracy': top5_ann_acc,  'Log Loss': top5_ann_ll,
         'Note': 'Avg top-5 ANN trials', '_strategy': 'ann_top5'},
        {'Modello': 'Ensemble Avg',      'Accuracy': avg_acc,       'Log Loss': avg_ll,
         'Note': 'ANN+LGB+XGB avg', '_strategy': 'ensemble_avg'},
        {'Modello': 'Ensemble Avg Top5', 'Accuracy': avg_acc_top5,  'Log Loss': avg_ll_top5,
         'Note': 'ANN5+LGB+XGB avg', '_strategy': 'ensemble_avg_top5'},
        {'Modello': 'Ensemble Stacking', 'Accuracy': stack_acc,     'Log Loss': stack_ll,
         'Note': 'Meta-LR su ANN+LGB+XGB', '_strategy': 'ensemble_stacking'},
    ]
    if lgb_model:
        results_list.append({'Modello': 'LightGBM', 'Accuracy': lgb_acc, 'Log Loss': lgb_ll,
                              'Note': f'Optuna {TRIALS_GBM} trials', '_strategy': 'lgb'})
    if xgb_model:
        results_list.append({'Modello': 'XGBoost',  'Accuracy': xgb_acc, 'Log Loss': xgb_ll,
                              'Note': f'Optuna {TRIALS_GBM} trials', '_strategy': 'xgb'})
    if surf_lgb_models:
        results_list.append({'Modello': 'LGB Surface', 'Accuracy': surf_acc, 'Log Loss': surf_ll,
                              'Note': 'LGB Hard/Clay/Grass routing', '_strategy': 'lgb_surface'})
    if ann_surf_models:
        results_list.append({'Modello': 'ANN Surface', 'Accuracy': ann_surf_acc, 'Log Loss': ann_surf_ll,
                              'Note': 'ANN Hard/Clay/Grass routing', '_strategy': 'ann_surface'})

    for r in results_list:
        r['_score'] = r['Accuracy'] - r['Log Loss']
    winner        = max(results_list, key=lambda r: r['Accuracy'])
    best_strategy = winner['_strategy']
    print(f"\n🏆 Strategia migliore: {winner['Modello']} "
          f"(acc={winner['Accuracy']:.2%}, ll={winner['Log Loss']:.4f})")

    # ── Persisti test set con predizioni per error analysis ───────────────────
    os.makedirs('error_analysis_output', exist_ok=True)
    test_analysis_df = df_test[FEATURES + ['target', 'tourney_date']].copy()
    test_analysis_df['p_raw']        = raw_probs_test
    test_analysis_df['p_calibrated'] = calibrated_probs_test
    test_analysis_df['ann_std']      = top5_ann_std_test   # std top-5 per-match
    test_analysis_df['predicted']    = (calibrated_probs_test >= 0.5).astype(int)
    test_analysis_df['correct']      = (test_analysis_df['predicted'] == test_analysis_df['target']).astype(int)
    test_predictions_path = os.path.join('error_analysis_output', 'test_predictions.csv')
    test_analysis_df.to_csv(test_predictions_path, index=False)
    print(f"   → {test_predictions_path} salvato ({len(test_analysis_df):,} righe)")

    # ── 8. Re-training finale su tutti i dati ─────────────────────────────────
    print(f"\n🚀 Re-training modello finale su TUTTI i dati...")

    X_all  = df_ml[FEATURES].fillna(0)
    y_all  = df_ml['target']
    d_all  = df_ml['tourney_date']
    lw_all = df_ml['level_weight'].values

    scaler_final = StandardScaler()
    scaler_final.fit(X_all.values)
    joblib.dump(scaler_final, 'scaler_ann.pkl')

    # 85% train / 15% val per early stopping (split temporale)
    ud_all    = np.sort(d_all.dropna().unique())
    cut_fin   = ud_all[int(len(ud_all) * 0.85)]
    mask_fin_tr  = d_all <  cut_fin
    mask_fin_val = d_all >= cut_fin

    X_tr_fin_sc  = scaler_final.transform(X_all[mask_fin_tr].values)
    X_val_fin_sc = scaler_final.transform(X_all[mask_fin_val].values)
    y_tr_fin     = y_all[mask_fin_tr]
    y_val_fin    = y_all[mask_fin_val]
    d_tr_fin     = d_all[mask_fin_tr]
    lw_tr_fin    = lw_all[mask_fin_tr]

    hl   = best_hp['hidden_layers'];  dr  = best_hp['dropout']
    lr   = best_hp['lr'];             bs  = best_hp['batch_size']
    ep   = best_hp['epochs'];         lam = best_hp['lambda_decay']
    sm   = best_hp.get('label_smoothing', 0.05)
    best_iset   = best_hp.get('interaction_set', 'core')
    best_ipairs = best_hp.get('interaction_pairs', DEFAULT_INTERACTION_PAIRS)
    best_ninter = best_hp.get('n_interactions', len(best_ipairs))

    USE_CUDA = device.type == 'cuda'; PIN_MEM = USE_CUDA
    bs_eff   = bs * 2 if USE_CUDA and bs < 2048 else bs

    X_tr_ft = torch.tensor(X_tr_fin_sc.astype(np.float32))
    y_tr_ft = torch.tensor(y_tr_fin.values.astype(np.float32) if hasattr(y_tr_fin, 'values') else y_tr_fin.astype(np.float32))
    X_vf_t  = torch.tensor(X_val_fin_sc.astype(np.float32))
    y_vf_t  = torch.tensor(y_val_fin.values.astype(np.float32) if hasattr(y_val_fin, 'values') else y_val_fin.astype(np.float32))

    w_fin = torch.tensor(calcola_pesi_combinati(d_tr_fin, lw_tr_fin, lam))
    ldr_tr_fin  = DataLoader(TensorDataset(X_tr_ft, y_tr_ft), batch_size=bs_eff,
                              sampler=WeightedRandomSampler(w_fin, len(w_fin), replacement=True),
                              pin_memory=PIN_MEM, num_workers=N_WORKERS)
    ldr_val_fin = DataLoader(TensorDataset(X_vf_t, y_vf_t), batch_size=4096,
                              shuffle=False, pin_memory=PIN_MEM, num_workers=N_WORKERS)

    model_final = TennisANNv3(len(FEATURES), hl, dr, best_ninter, best_ipairs)
    model_final, _ = train_model(model_final, ldr_tr_fin, ldr_val_fin, epochs=ep, lr=lr, smoothing=sm)

    # Re-train GBM se la strategia vincente li richiede
    needs_gbm  = best_strategy in ('lgb', 'lgb_surface', 'xgb', 'ensemble_avg', 'ensemble_avg_top5', 'ensemble_stacking')
    needs_ann_surf = best_strategy == 'ann_surface'
    needs_top5 = best_strategy in ('ann_top5', 'ensemble_avg_top5')

    X_all_sc = scaler_final.transform(X_all.values)
    y_all_np = y_all.values if hasattr(y_all, 'values') else np.array(y_all)
    all_weights = calcola_pesi_combinati(d_all, lw_all, 0.003)

    lgb_final = None; xgb_final = None
    if needs_gbm and lgb_model is not None:
        print("   Re-training LightGBM su tutti i dati...")
        lgb_final = lgb.LGBMClassifier(**lgb_model.get_params())
        lgb_final.fit(X_all_sc, y_all_np, sample_weight=all_weights)
    if needs_gbm and xgb_model is not None:
        print("   Re-training XGBoost su tutti i dati...")
        xgb_final = xgb_lib.XGBClassifier(**xgb_model.get_params())
        xgb_final.set_params(early_stopping_rounds=None)
        xgb_final.fit(X_all_sc, y_all_np, sample_weight=all_weights)

    top5_state_dicts = [model_final.state_dict()]
    top5_configs     = [best_hp]
    if needs_top5:
        for idx_i in range(1, min(5, len(risultati))):
            hp_i = risultati[idx_i]['_config']
            hl_i = hp_i['hidden_layers']; dr_i = hp_i['dropout']; lr_i = hp_i['lr']
            bs_i = hp_i['batch_size'];    ep_i = hp_i['epochs'];   sm_i = hp_i.get('label_smoothing', 0.05)
            ip_i = hp_i.get('interaction_pairs', DEFAULT_INTERACTION_PAIRS)
            ni_i = hp_i.get('n_interactions', len(ip_i))
            print(f"   Re-training ANN #{idx_i+1} (arch={hl_i})...")
            bs_ei = bs_i * 2 if USE_CUDA and bs_i < 2048 else bs_i
            ldr_i = DataLoader(TensorDataset(X_tr_ft, y_tr_ft), batch_size=bs_ei,
                               sampler=WeightedRandomSampler(w_fin, len(w_fin), replacement=True),
                               pin_memory=PIN_MEM, num_workers=N_WORKERS)
            m_i = TennisANNv3(len(FEATURES), hl_i, dr_i, ni_i, ip_i)
            m_i, _ = train_model(m_i, ldr_i, ldr_val_fin, epochs=ep_i, lr=lr_i, smoothing=sm_i)
            top5_state_dicts.append(m_i.state_dict())
            top5_configs.append(hp_i)

    # ── 9. Costruzione modelo_finale.pkl ──────────────────────────────────────
    from datetime import datetime
    modelo_finale = {
        'strategy':             best_strategy,
        'model_name':           winner['Modello'],
        'accuracy':             winner['Accuracy'],
        'score':                winner['_score'],
        'features':             FEATURES,
        'scaler':               scaler_final,
        'trained_date':         datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'ann':                  {'state_dict': model_final.state_dict(), 'config': best_hp},
        # Moduli di supporto (sempre presenti, indipendenti da strategy)
        'calibrator':           calibrator,
        'calibration_curve':    calibration_curve_data,
        'ann_top5_uncertainty': top5_for_uncertainty,
        'lgb_surface_models':   {},
        'ann_surface_models':   {},
        'surface_accuracy':     surface_accuracy,
    }
    if best_strategy == 'ann_top5':
        modelo_finale['ann_top5'] = [{'state_dict': sd, 'config': c}
                                     for sd, c in zip(top5_state_dicts, top5_configs)]
    elif best_strategy == 'ann_surface':
        print("   Re-training ANN Surface models su tutti i dati...")
        surf_final_ann = {}
        for sname, sresults in ann_surf_models.items():
            senc = {'Hard': 0, 'Clay': 1, 'Grass': 2}[sname]
            mask_all_s = df_ml['surface_enc'].values == senc
            if mask_all_s.sum() < 100:
                continue
            X_all_s  = X_all_sc[mask_all_s]
            y_all_s  = y_all_np[mask_all_s]
            d_all_s  = d_all.values[mask_all_s]
            lw_all_s = lw_all[mask_all_s]
            ud_s  = np.sort(np.unique(d_all_s[~np.isnan(d_all_s.astype(float))]))
            cut_s = ud_s[int(len(ud_s) * 0.85)] if len(ud_s) > 1 else d_all_s[0]
            mtr_s = d_all_s <  cut_s
            mva_s = d_all_s >= cut_s
            w_s   = torch.tensor(calcola_pesi_combinati(d_all_s[mtr_s], lw_all_s[mtr_s], 0.003))
            Xtr_s_t = torch.tensor(X_all_s[mtr_s].astype(np.float32))
            ytr_s_t = torch.tensor(y_all_s[mtr_s].astype(np.float32))
            Xva_s_t = torch.tensor(X_all_s[mva_s].astype(np.float32))
            yva_s_t = torch.tensor(y_all_s[mva_s].astype(np.float32))
            hp_s  = sresults[0]['_config']
            hl_s  = hp_s['hidden_layers']; dr_s = hp_s['dropout']
            lr_s  = hp_s['lr'];             bs_s = hp_s['batch_size']
            ep_s  = hp_s['epochs'];          sm_s = hp_s.get('label_smoothing', 0.05)
            ip_s  = hp_s.get('interaction_pairs', DEFAULT_INTERACTION_PAIRS)
            ni_s  = hp_s.get('n_interactions', len(ip_s))
            bs_eff_s = bs_s * 2 if USE_CUDA and bs_s < 2048 else bs_s
            ldr_s_tr  = DataLoader(TensorDataset(Xtr_s_t, ytr_s_t), batch_size=bs_eff_s,
                                   sampler=WeightedRandomSampler(w_s, len(w_s), replacement=True),
                                   pin_memory=PIN_MEM, num_workers=N_WORKERS)
            ldr_s_val = DataLoader(TensorDataset(Xva_s_t, yva_s_t), batch_size=4096,
                                   shuffle=False, pin_memory=PIN_MEM, num_workers=N_WORKERS)
            print(f"   Re-training ANN {sname} ({mask_all_s.sum():,} match)...")
            m_s = TennisANNv3(len(FEATURES), hl_s, dr_s, ni_s, ip_s)
            m_s, _ = train_model(m_s, ldr_s_tr, ldr_s_val, epochs=ep_s, lr=lr_s, smoothing=sm_s)
            surf_final_ann[sname] = {'state_dict': m_s.state_dict(), 'config': hp_s}
        modelo_finale['ann_surface_models'] = surf_final_ann

    elif best_strategy == 'lgb_surface':
        print("   Re-training LGB Surface models su tutti i dati...")
        surf_final_models = {}
        for sname, sm_model in surf_lgb_models.items():
            senc = {'Hard': 0, 'Clay': 1, 'Grass': 2}[sname]
            mask_all_s = df_ml['surface_enc'] == senc
            if mask_all_s.sum() > 0:
                X_all_s = X_all_sc[mask_all_s.values]
                y_all_s = y_all_np[mask_all_s.values]
                w_all_s = all_weights[mask_all_s.values]
                m_s = lgb.LGBMClassifier(**sm_model.get_params())
                m_s.fit(X_all_s, y_all_s, sample_weight=w_all_s)
                surf_final_models[sname] = m_s
        modelo_finale['lgb_surface_models'] = surf_final_models
        modelo_finale['lgb_model'] = lgb_final
    elif best_strategy == 'lgb':
        modelo_finale['lgb_model'] = lgb_final
    elif best_strategy == 'xgb':
        modelo_finale['xgb_model'] = xgb_final
    elif best_strategy in ('ensemble_avg', 'ensemble_stacking'):
        modelo_finale['lgb_model'] = lgb_final; modelo_finale['xgb_model'] = xgb_final
        if best_strategy == 'ensemble_stacking': modelo_finale['meta_model'] = meta_model
    elif best_strategy == 'ensemble_avg_top5':
        modelo_finale['ann_top5']  = [{'state_dict': sd, 'config': c}
                                      for sd, c in zip(top5_state_dicts, top5_configs)]
        modelo_finale['lgb_model'] = lgb_final; modelo_finale['xgb_model'] = xgb_final

    joblib.dump(modelo_finale, 'modelo_finale.pkl')
    print(f"\n   → modelo_finale.pkl salvato (strategia: {best_strategy})")

    # ── 10. Confronto finale ───────────────────────────────────────────────────
    results_clean = [{k: v for k, v in r.items() if not k.startswith('_')} for r in results_list]
    confronto_finale(results_clean)

    print("\n✅ File salvati:")
    for f in ['modelo_finale.pkl', 'scaler_ann.pkl', 'glicko2_stores.pkl', 'serve_stats.pkl', 'return_stats.pkl',
              'streak_players.pkl', 'momentum_surface.pkl', 'recent_form.pkl',
              'close_match_hist.pkl', 'upset_hist.pkl', 'late_round_hist.pkl', 'match_load.pkl',
              'resultados_comparacion_finale.csv']:
        stato = "✅" if os.path.exists(f) else "—"
        print(f"  {stato}  {f}")
