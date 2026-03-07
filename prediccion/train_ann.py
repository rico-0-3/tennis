"""
train_ann.py v4.0  —  ANN Avanzata per Tennis Predictor
========================================================
Funzionalità v4 rispetto v3:
  ✅ Rating Elo per superficie  (più accurato del ranking ATP)
  ✅ Striscia attiva            (consecutive wins/losses)
  ✅ Momentum per superficie    (rolling 10 match per surface)
  ✅ Statistiche di ritorno     (return_pct, bp_conv, return_1st)
  ✅ Architettura Wide & Deep   (residual + feature interaction)
  ✅ Label smoothing            (riduce overconfidence)
  ✅ Temporal cross-validation  (expanding window per anno)
  ✅ Calibrazione Platt scaling (probabilità calibrate)
  ✅ Ponderazione torneo        (Grand Slam pesa x2, Masters x1.5 ...)
  ✅ Optuna bayesian search     (più efficiente del random search)
  ✅ Best-of-5 indicator        (Slam favorisce il più forte)
  ✅ Days since last match      (riposo vs ruggine)
  ✅ H2H per superficie         (scontri diretti surface-specific)
  🔧 Rimosse feature ridondanti (diff_rank, diff_rank_points, diff_seed, draw_size, diff_hand)

UTILIZZO:
    cd prediccion/
    python train_ann.py

OUTPUT:
    modelo_finale.pkl            ← modello finale (auto-selezionato)
    scaler_ann.pkl               ← StandardScaler 29 feature
    elo_surface.pkl              ← Elo corrente per superficie
    elo_overall.pkl              ← Elo overall (tutte le superfici)
    streak_players.pkl           ← striscia attiva per giocatore
    momentum_surface.pkl         ← momentum per (giocatore, superficie)
    recent_form.pkl              ← recent form per giocatore (ultimi 10)
    last_match_date.pkl          ← data ultimo match per giocatore
    h2h_surface.pkl              ← H2H per superficie
    resultados_comparacion_finale.csv  ← confronto modelli
"""

import os, json, warnings, sys
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn

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
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
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

# ── Optuna (opzionale — fallback a random search se non installato) ───────────
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.INFO)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("⚠️  optuna non trovato. Installa con: pip install optuna")
    print("   Fallback: random search classico.\n")

# ─── Seed + Device ────────────────────────────────────────────────────────────
SEED = 42
import random
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Device: {device}")

# ─── Configurazione globale ───────────────────────────────────────────────────
TRIALS = 100       # numero trial Optuna ANN 
TRIALS_GBM = 40    # trial per LightGBM e XGBoost

# Importanza tornei (moltiplicatore sui pesi campione)
LEVEL_MULT = {'G': 2.0, 'M': 1.5, 'F': 1.4, 'A': 1.0,
              'D': 1.0, 'C': 0.8, 'S': 0.7, 'E': 0.5}

ALPHA_GAMES = 0.1  # peso della loss di regressione (total games) rispetto alla classificazione


def parse_total_games(score_str):
    """Estrae il numero totale di game da una stringa di punteggio.
    Es: '7-6 6-2' → 21, '6-4 3-6 7-5' → 31, 'W/O' → NaN"""
    if not isinstance(score_str, str) or not score_str.strip():
        return np.nan
    score = score_str.strip().upper()
    if any(x in score for x in ['W/O', 'RET', 'DEF', 'WO', 'ABN', 'UNP', 'BYE', 'NA', 'NAN']):
        return np.nan
    total = 0
    for token in score.split():
        token = token.split('(')[0]  # "7-6(5)" → "7-6"
        parts = token.split('-')
        if len(parts) == 2:
            try:
                total += int(parts[0]) + int(parts[1])
            except ValueError:
                continue
    return float(total) if total > 0 else np.nan


# ══════════════════════════════════════════════════════════════════════════════
# 1.  FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def carica_e_prepara(csv_path: str):
    print(f"\n📂 Caricamento: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    df['tourney_date'] = pd.to_numeric(df['tourney_date'], errors='coerce')
    df = df.sort_values(by=['tourney_date', 'match_num']).reset_index(drop=True)
    df['minutes'] = df['minutes'].fillna(90)
    print(f"   → {len(df):,} partite caricate")

    # ── Encoding ─────────────────────────────────────────────────────────────
    surface_map   = {'Hard': 0, 'Clay': 1, 'Grass': 2, 'Carpet': 3}
    hand_map      = {'R': 1, 'L': -1, 'U': 0}
    level_map     = {'G': 5, 'M': 4, 'A': 3, 'D': 3, 'F': 4, 'C': 2, 'S': 1, 'E': 0}
    round_map     = {'F': 7, 'SF': 6, 'QF': 5, 'R16': 4, 'R32': 3,
                     'R64': 2, 'R128': 1, 'RR': 4, 'BR': 3}

    df['surface_enc']       = df['surface'].map(surface_map).fillna(0)
    df['w_hand_enc']        = df['winner_hand'].map(hand_map).fillna(0)
    df['l_hand_enc']        = df['loser_hand'].map(hand_map).fillna(0)
    df['tourney_level_enc'] = df['tourney_level'].map(level_map).fillna(2)
    df['round_enc']         = df['round'].map(round_map).fillna(3)

    def detect_country(name):
        t = str(name).upper()
        if 'MADRID' in t or 'BARCELONA' in t: return 'ESP'
        if 'PARIS' in t or 'ROLAND' in t:     return 'FRA'
        if 'US OPEN' in t or 'INDIAN' in t:   return 'USA'
        if 'WIMBLEDON' in t or 'LONDON' in t: return 'GBR'
        if 'AUSTRALIAN' in t or 'MELBOURNE' in t: return 'AUS'
        return 'NEUTRAL'
    df['tourney_ioc'] = df['tourney_name'].apply(detect_country)

    # ── Win-rate superficie ───────────────────────────────────────────────────
    wins   = df.groupby(['winner_name', 'surface']).size().reset_index(name='wins')
    losses = df.groupby(['loser_name',  'surface']).size().reset_index(name='losses')
    wins.columns = ['player','surface','wins']
    losses.columns = ['player','surface','losses']
    stats = pd.merge(wins, losses, on=['player','surface'], how='outer').fillna(0)
    stats['total'] = stats['wins'] + stats['losses']
    stats = stats[stats['total'] >= 5]
    stats['win_rate'] = stats['wins'] / stats['total']
    stats_dict = stats.set_index(['player','surface'])['win_rate'].to_dict()
    joblib.dump(stats_dict, 'stats_superficie_v2.pkl')
    def get_skill(p, s): return stats_dict.get((p, s), 0.5)

    # ── Loop cronologico (no data leakage) ───────────────────────────────────
    fatiga_t  = {}
    racha_t   = {}
    h2h_t     = {}
    h2h_surf_t = {}  # {(sorted_p1, sorted_p2, surface): [w1, w2]}
    serve_t   = {}
    return_t  = {}   # {player: {return_pct: [...], bp_conv: [...], return_1st: [...]}}
    elo_surf  = {}   # {(player, surface): float}  — ELO per superficie
    elo_overall = {} # {player: float}  — ELO overall (all surfaces)
    streak_t  = {}   # {player: int}  +N=N vittorie consecutive, -N=sconfitte
    recent_form_t = {}  # {player: [last 10 results across all surfaces]}
    last_match_date_t = {}  # {player: int (YYYYMMDD)}  — data ultimo match

    ELO_DEFAULT = 1500.0
    K_BASE      = 32.0
    K_LEVEL_ELO = {'G': 1.25, 'M': 1.15, 'F': 1.1, 'A': 1.0,
                   'D': 0.95, 'C': 0.9, 'S': 0.85, 'E': 0.8}

    def get_elo(p, s):
        return elo_surf.get((p, s), ELO_DEFAULT)

    rows = []
    for idx, row in df.iterrows():
        tid  = row['tourney_id']
        w, l = row['winner_name'], row['loser_name']
        dur  = row['minutes']
        surf = row['surface']

        # Salta righe con nomi mancanti
        if not isinstance(w, str) or not isinstance(l, str) or pd.isna(w) or pd.isna(l):
            continue
        if not isinstance(surf, str):
            continue  # salta righe senza superficie

        # --- Total games (target regressione) ---
        total_games = parse_total_games(str(row.get('score', '')))

        # --- Fatica ---
        f_w = fatiga_t.get((tid, w), 0); f_l = fatiga_t.get((tid, l), 0)
        fatiga_t[(tid, w)] = f_w + dur;  fatiga_t[(tid, l)] = f_l + dur

        # --- Momentum per superficie ---
        hw = racha_t.get((w, surf), []); hl = racha_t.get((l, surf), [])
        mw = np.mean(hw) if hw else 0.5
        ml = np.mean(hl) if hl else 0.5
        hw.append(1); hl.append(0)
        if len(hw) > 10: hw.pop(0)
        if len(hl) > 10: hl.pop(0)
        racha_t[(w, surf)] = hw; racha_t[(l, surf)] = hl

        # --- H2H globale ---
        p1k, p2k = sorted([w, l]); key = (p1k, p2k)
        rec = h2h_t.get(key, [0, 0])
        if w == p1k:
            h2h_w = rec[0]-rec[1]; h2h_l = rec[1]-rec[0]; rec[0] += 1
        else:
            h2h_w = rec[1]-rec[0]; h2h_l = rec[0]-rec[1]; rec[1] += 1
        h2h_t[key] = rec

        # --- H2H per superficie ---
        key_s = (p1k, p2k, surf)
        rec_s = h2h_surf_t.get(key_s, [0, 0])
        if w == p1k:
            h2h_s_w = rec_s[0]-rec_s[1]; h2h_s_l = rec_s[1]-rec_s[0]; rec_s[0] += 1
        else:
            h2h_s_w = rec_s[1]-rec_s[0]; h2h_s_l = rec_s[0]-rec_s[1]; rec_s[1] += 1
        h2h_surf_t[key_s] = rec_s

        # --- Days since last match ---
        td = int(row['tourney_date']) if pd.notna(row.get('tourney_date')) else 20200101
        def _date_to_days(d):
            """Converte YYYYMMDD int in giorni approssimativi."""
            y, rest = divmod(d, 10000)
            m, day = divmod(rest, 100)
            return y * 365 + m * 30 + day
        td_days = _date_to_days(td)
        last_w = last_match_date_t.get(w)
        last_l = last_match_date_t.get(l)
        days_since_w = (td_days - _date_to_days(last_w)) if last_w is not None else 14.0
        days_since_l = (td_days - _date_to_days(last_l)) if last_l is not None else 14.0
        # Clamp a 0-180 per evitare outlier estremi (es. inizio carriera)
        days_since_w = max(0.0, min(180.0, float(days_since_w)))
        days_since_l = max(0.0, min(180.0, float(days_since_l)))
        last_match_date_t[w] = td
        last_match_date_t[l] = td

        # --- Elo per superficie (dynamic K by tournament level) ---
        elo_w = get_elo(w, surf); elo_l = get_elo(l, surf)
        expected_w = 1.0 / (1.0 + 10.0 ** ((elo_l - elo_w) / 400.0))
        level_code = str(row.get('tourney_level', 'A'))
        k_dynamic = K_BASE * K_LEVEL_ELO.get(level_code, 1.0)
        elo_surf[(w, surf)] = elo_w + k_dynamic * (1.0 - expected_w)
        elo_surf[(l, surf)] = elo_l + k_dynamic * (0.0 - (1.0 - expected_w))

        # --- Elo overall ---
        elo_ov_w = elo_overall.get(w, ELO_DEFAULT)
        elo_ov_l = elo_overall.get(l, ELO_DEFAULT)
        expected_ov = 1.0 / (1.0 + 10.0 ** ((elo_ov_l - elo_ov_w) / 400.0))
        elo_overall[w] = elo_ov_w + k_dynamic * (1.0 - expected_ov)
        elo_overall[l] = elo_ov_l + k_dynamic * (0.0 - (1.0 - expected_ov))

        # --- Recent form (all surfaces, last 10 matches) ---
        rf_w = recent_form_t.get(w, [])
        rf_l = recent_form_t.get(l, [])
        form_w = np.mean(rf_w) if rf_w else 0.5
        form_l = np.mean(rf_l) if rf_l else 0.5
        rf_w.append(1.0); rf_l.append(0.0)
        if len(rf_w) > 10: rf_w.pop(0)
        if len(rf_l) > 10: rf_l.pop(0)
        recent_form_t[w] = rf_w; recent_form_t[l] = rf_l

        # --- Striscia attiva ---
        str_w = streak_t.get(w, 0); str_l = streak_t.get(l, 0)
        streak_t[w] = max(0, str_w) + 1 if str_w >= 0 else 1
        streak_t[l] = min(0, str_l) - 1 if str_l <= 0 else -1

        # --- Statistiche servizio medie ---
        def get_sa(player):
            s = serve_t.get(player, {})
            return {'ace':     np.mean(s.get('ace',     [0])),
                    'df':      np.mean(s.get('df',      [0])),
                    '1st_pct': np.mean(s.get('1st_pct', [0.6])),
                    '1st_won': np.mean(s.get('1st_won', [0.7])),
                    '2nd_won': np.mean(s.get('2nd_won', [0.5])),
                    'bp_saved':np.mean(s.get('bp_saved',[0.6]))}

        sa_w = get_sa(w); sa_l = get_sa(l)

        # --- Statistiche ritorno medie (derivate dal servizio avversario) ---
        def get_ra(player):
            s = return_t.get(player, {})
            return {'return_pct':  np.mean(s.get('return_pct',  [0.35])),
                    'bp_conv':     np.mean(s.get('bp_conv',     [0.35])),
                    'return_1st':  np.mean(s.get('return_1st',  [0.30]))}

        ra_w = get_ra(w); ra_l = get_ra(l)

        def upd_return(player, rd, opp_pref):
            """Calcola stats di ritorno dal servizio dell'avversario"""
            s = return_t.setdefault(player, {})
            opp_svpt   = rd.get(f'{opp_pref}_svpt', np.nan)
            opp_1stWon = rd.get(f'{opp_pref}_1stWon', np.nan)
            opp_2ndWon = rd.get(f'{opp_pref}_2ndWon', np.nan)
            opp_1stIn  = rd.get(f'{opp_pref}_1stIn', np.nan)
            opp_bpFaced = rd.get(f'{opp_pref}_bpFaced', np.nan)
            opp_bpSaved = rd.get(f'{opp_pref}_bpSaved', np.nan)
            # return_pct: % punti vinti in risposta
            if opp_svpt and opp_svpt > 0 and not np.isnan(opp_svpt):
                fw = opp_1stWon if not (isinstance(opp_1stWon, float) and np.isnan(opp_1stWon)) else 0
                sw = opp_2ndWon if not (isinstance(opp_2ndWon, float) and np.isnan(opp_2ndWon)) else 0
                rtn_won = (opp_svpt - fw - sw) / opp_svpt
                lst = s.setdefault('return_pct', [])
                lst.append(float(rtn_won))
                if len(lst) > 10: lst.pop(0)
            # bp_conv: % break point convertiti
            if opp_bpFaced and opp_bpFaced > 0 and not (isinstance(opp_bpFaced, float) and np.isnan(opp_bpFaced)):
                bps = opp_bpSaved if not (isinstance(opp_bpSaved, float) and np.isnan(opp_bpSaved)) else 0
                bp_conv = (opp_bpFaced - bps) / opp_bpFaced
                lst = s.setdefault('bp_conv', [])
                lst.append(float(bp_conv))
                if len(lst) > 10: lst.pop(0)
            # return_1st: % punti vinti sulla prima dell'avversario
            if opp_1stIn and opp_1stIn > 0 and not (isinstance(opp_1stIn, float) and np.isnan(opp_1stIn)):
                fw = opp_1stWon if not (isinstance(opp_1stWon, float) and np.isnan(opp_1stWon)) else 0
                rtn_1st = (opp_1stIn - fw) / opp_1stIn
                lst = s.setdefault('return_1st', [])
                lst.append(float(rtn_1st))
                if len(lst) > 10: lst.pop(0)

        def upd_serve(player, rd, pref):
            s = serve_t.setdefault(player, {})
            svpt=rd.get(f'{pref}_svpt',np.nan); fi=rd.get(f'{pref}_1stIn',np.nan)
            fw=rd.get(f'{pref}_1stWon',np.nan); sw=rd.get(f'{pref}_2ndWon',np.nan)
            bps=rd.get(f'{pref}_bpSaved',np.nan); bpf=rd.get(f'{pref}_bpFaced',np.nan)
            for k2,v in [('ace',rd.get(f'{pref}_ace',np.nan)),
                         ('df', rd.get(f'{pref}_df', np.nan)),
                         ('1st_pct', fi/svpt if svpt and svpt>0 else np.nan),
                         ('1st_won', fw/fi   if fi   and fi>0   else np.nan),
                         ('2nd_won', sw/(svpt-fi) if svpt and fi and (svpt-fi)>0 else np.nan),
                         ('bp_saved',bps/bpf if bpf and bpf>0 else np.nan)]:
                if not (isinstance(v, float) and np.isnan(v)):
                    lst = s.setdefault(k2, [])
                    lst.append(float(v))
                    if len(lst) > 10: lst.pop(0)

        rd = row.to_dict(); upd_serve(w, rd, 'w'); upd_return(w, rd, 'l')

        sk_w = get_skill(w, surf); sk_l = get_skill(l, surf)
        home_w = 1 if row['winner_ioc'] == row['tourney_ioc'] else 0
        home_l = 1 if row['loser_ioc']  == row['tourney_ioc'] else 0
        pts_w = float(row['winner_rank_points']) if pd.notna(row.get('winner_rank_points')) else 0
        pts_l = float(row['loser_rank_points'])  if pd.notna(row.get('loser_rank_points'))  else 0
        rk_w = float(row['winner_rank']) if pd.notna(row.get('winner_rank')) else 500
        rk_l = float(row['loser_rank']) if pd.notna(row.get('loser_rank')) else 500
        lev_w  = LEVEL_MULT.get(str(row.get('tourney_level','')), 1.0)

        # Best-of-5 indicator (Grand Slam)
        best_of_val = row.get('best_of', 3)
        is_bo5 = 1.0 if best_of_val == 5 or best_of_val == '5' else 0.0

        diffs = {
            'log_rank_ratio':    np.log1p(rk_l) - np.log1p(rk_w),
            'log_pts_ratio':     np.log1p(pts_w) - np.log1p(pts_l),
            'diff_age':          (row['winner_age']-row['loser_age'])
                                 if pd.notna(row.get('winner_age')) and pd.notna(row.get('loser_age')) else 0,
            'diff_ht':           (row['winner_ht']-row['loser_ht'])
                                 if pd.notna(row.get('winner_ht')) and pd.notna(row.get('loser_ht')) else 0,
            # Elo
            'diff_elo':          elo_w - elo_l,             # Elo PRIMA dell'aggiornamento
            'diff_elo_overall':  elo_ov_w - elo_ov_l,       # Elo overall (all surfaces)
            'diff_streak':       float(str_w - str_l),      # striscia attiva
            'diff_recent_form':  form_w - form_l,           # recent 10-match form
            # contesto
            'surface_enc':       float(row['surface_enc']),
            'tourney_level':     float(row['tourney_level_enc']),
            'round_enc':         float(row['round_enc']),
            'is_best_of_5':      is_bo5,                    # best-of-3 vs best-of-5
            'diff_skill':        sk_w - sk_l,
            'diff_home':         home_w - home_l,
            'diff_fatigue':      f_w - f_l,
            'diff_momentum':     mw - ml,
            'diff_h2h':          h2h_w - h2h_l,
            'diff_h2h_surface':  h2h_s_w - h2h_s_l,         # H2H per superficie
            'diff_days_since_last': days_since_w - days_since_l,  # riposo vs ruggine
            'diff_ace':          sa_w['ace']     - sa_l['ace'],
            'diff_1st_pct':      sa_w['1st_pct'] - sa_l['1st_pct'],
            'diff_1st_won':      sa_w['1st_won'] - sa_l['1st_won'],
            'diff_2nd_won':      sa_w['2nd_won'] - sa_l['2nd_won'],
            'diff_bp_saved':     sa_w['bp_saved']- sa_l['bp_saved'],
            # feature ritorno
            'diff_return_pct':   ra_w['return_pct'] - ra_l['return_pct'],
            'diff_bp_conv':      ra_w['bp_conv']    - ra_l['bp_conv'],
            'diff_return_1st':   ra_w['return_1st'] - ra_l['return_1st'],
            'tourney_date':      float(row['tourney_date']) if pd.notna(row['tourney_date']) else 20200101,
            'level_weight':      lev_w,
        }

        # ── Court speed (contestuali, uguali per entrambi → NON negare in d0) ──
        tourney_year = int(str(row['tourney_date'])[:4]) if pd.notna(row.get('tourney_date')) else 2025
        surf_safe = surf if isinstance(surf, str) else 'Hard'
        court_ace, court_spd = get_court_stats(
            row.get('tourney_name', ''), surf_safe, tourney_year)
        diffs['court_ace_pct'] = court_ace
        diffs['court_speed']   = court_spd

        d1 = diffs.copy(); d1['target'] = 1; d1['total_games'] = total_games
        d0 = {k: -v if k not in ('surface_enc','tourney_level','round_enc',
                                  'is_best_of_5','tourney_date','level_weight',
                                  'court_ace_pct','court_speed')
              else v for k, v in diffs.items()}
        d0['target'] = 0; d0['total_games'] = total_games
        rows.append(d1); rows.append(d0)
        upd_serve(l, rd, 'l'); upd_return(l, rd, 'w')

    df_out = pd.DataFrame(rows)
    print(f"   → Dataset: {len(df_out):,} righe | {df_out.shape[1]} colonne")

    # Salva Elo corrente (usato dal predictor)
    joblib.dump(elo_surf, 'elo_surface.pkl')
    print("   → elo_surface.pkl salvato")
    # Salva anche streak corrente per il predictor
    joblib.dump(streak_t, 'streak_players.pkl')
    # Salva momentum per superficie {(player, surface): [last 10 results]}
    joblib.dump(racha_t, 'momentum_surface.pkl')
    print("   → momentum_surface.pkl salvato")
    # Salva Elo overall e recent form (usati dal predictor)
    joblib.dump(elo_overall, 'elo_overall.pkl')
    print("   → elo_overall.pkl salvato")
    joblib.dump(recent_form_t, 'recent_form.pkl')
    print("   → recent_form.pkl salvato")
    # Salva H2H per superficie e data ultimo match
    joblib.dump(h2h_surf_t, 'h2h_surface.pkl')
    print("   → h2h_surface.pkl salvato")
    joblib.dump(last_match_date_t, 'last_match_date.pkl')
    print("   → last_match_date.pkl salvato")

    return df_out, stats_dict, elo_surf, streak_t


# ── Lista feature (29) — v4.0: rimossi 5 ridondanti, aggiunti bo5/days/h2h_surf/1st_pct
FEATURES = [
    'log_rank_ratio', 'log_pts_ratio',                    # 0-1   (ranking, compressi)
    'diff_age', 'diff_ht',                                # 2-3
    'diff_elo', 'diff_elo_overall',                       # 4-5   (Elo superficie + overall)
    'diff_streak', 'diff_recent_form',                    # 6-7   (forma)
    'surface_enc', 'tourney_level', 'round_enc',          # 8-9-10 (contesto)
    'is_best_of_5',                                       # 11    (bo3 vs bo5)
    'diff_skill', 'diff_home',                            # 12-13
    'diff_fatigue', 'diff_momentum',                      # 14-15
    'diff_h2h', 'diff_h2h_surface',                       # 16-17 (H2H globale + per superficie)
    'diff_days_since_last',                               # 18    (riposo vs ruggine)
    'diff_ace', 'diff_1st_pct', 'diff_1st_won',           # 19-20-21 (servizio)
    'diff_2nd_won', 'diff_bp_saved',                      # 22-23
    'diff_return_pct', 'diff_bp_conv', 'diff_return_1st', # 24-25-26 (ritorno)
    'court_ace_pct', 'court_speed',                       # 27-28
]  # totale: 29 feature


# ══════════════════════════════════════════════════════════════════════════════
# 2.  PESI COMBINATI  (temporale × livello torneo)
# ══════════════════════════════════════════════════════════════════════════════

def calcola_pesi_temporali(date_series: pd.Series, lambda_decay: float = 0.003) -> np.ndarray:
    max_date  = date_series.max()
    days_ago  = (max_date - date_series) / 10000 * 365
    weights   = np.exp(-lambda_decay * days_ago)
    weights   = weights / weights.sum() * len(weights)
    return weights.values.astype(np.float32)

def calcola_pesi_combinati(date_series: pd.Series, level_weights: np.ndarray,
                           lambda_decay: float = 0.003) -> np.ndarray:
    """Pesi temporali moltiplicati per l'importanza del torneo."""
    temporal = calcola_pesi_temporali(date_series, lambda_decay)
    combined = temporal * level_weights
    combined = combined / combined.sum() * len(combined)
    return combined.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 3.  ARCHITETTURA + TRAINING
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

# Coppie di feature per interaction layer (indici aggiornati per 28 feature v4.0)
# 0=log_rank_ratio, 1=log_pts_ratio, 2=age, 3=ht, 4=elo, 5=elo_overall,
# 6=streak, 7=recent_form, 8=surface, 9=level, 10=round, 11=bo5,
# 12=skill, 13=home, 14=fatigue, 15=momentum, 16=h2h, 17=h2h_surface,
# 18=days_since_last, 19=ace, 20=1st_pct, 21=1st_won, 22=2nd_won,
# 23=bp_saved, 24=return_pct, 25=bp_conv, 26=return_1st,
# 27=court_ace, 28=court_speed  (if included)

INTERACTION_SETS = {
    'core': [                    # Coppie fondamentali (ranking × form)
        (4, 12), (0, 15), (4, 15), (12, 15), (0, 1),
        (4, 16), (6, 15), (12, 14), (14, 15), (1, 12),
    ],
    'serve_return': [            # Coppie servizio × ritorno
        (4, 12), (0, 15), (4, 15), (19, 24), (21, 26),
        (23, 25), (4, 16), (12, 15), (6, 15), (0, 1),
    ],
    'context': [                 # Coppie con contesto (superficie, livello torneo)
        (4, 12), (0, 15), (4, 8), (12, 8), (15, 8),
        (4, 9), (0, 1), (4, 16), (6, 15), (12, 15),
    ],
    'minimal': [                 # Solo le 5 più importanti
        (4, 12), (0, 15), (4, 15), (4, 16), (0, 1),
    ],
}
DEFAULT_INTERACTION_PAIRS = INTERACTION_SETS['core']
N_INTERACTIONS = len(DEFAULT_INTERACTION_PAIRS)


class TennisANNv3(nn.Module):
    """Wide & Deep con Feature Interaction, Residual Connections e Multi-Task (games)."""
    def __init__(self, input_dim: int, hidden_layers: list, dropout: float = 0.3,
                 n_interactions: int = N_INTERACTIONS,
                 interaction_pairs: list = None,
                 predict_games: bool = False):
        super().__init__()
        self.interaction_pairs = interaction_pairs or DEFAULT_INTERACTION_PAIRS
        self.n_interactions = min(n_interactions, len(self.interaction_pairs))
        self.predict_games = predict_games

        # === Wide path (linear, cattura relazioni dirette) ===
        self.wide = nn.Linear(input_dim, 1)

        # === Deep path con residual connections ===
        interaction_dim = input_dim + self.n_interactions
        self.deep_layers = nn.ModuleList()
        self.deep_norms = nn.ModuleList()
        self.deep_drops = nn.ModuleList()
        self.residual_projs = nn.ModuleList()

        prev = interaction_dim
        for h in hidden_layers:
            self.deep_layers.append(nn.Linear(prev, h))
            self.deep_norms.append(nn.BatchNorm1d(h))
            self.deep_drops.append(nn.Dropout(dropout))
            if prev != h:
                self.residual_projs.append(nn.Linear(prev, h))
            else:
                self.residual_projs.append(nn.Identity())
            prev = h

        self.deep_out = nn.Linear(prev, 1)

        # === Games regression head (multi-task) ===
        if predict_games:
            games_hidden = max(prev // 2, 16)
            self.games_head = nn.Sequential(
                nn.Linear(prev, games_hidden),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(games_hidden, 1)
            )

    def forward(self, x):
        # Wide
        wide_out = self.wide(x)

        # Feature interactions (prodotti tra coppie importanti)
        interactions = []
        for i, j in self.interaction_pairs[:self.n_interactions]:
            if i < x.shape[1] and j < x.shape[1]:
                interactions.append(x[:, i] * x[:, j])
            else:
                interactions.append(torch.zeros(x.shape[0], device=x.device))
        inter_t = torch.stack(interactions, dim=1)
        deep_in = torch.cat([x, inter_t], dim=1)

        # Deep con residual
        h = deep_in
        for layer, norm, drop, res_proj in zip(
                self.deep_layers, self.deep_norms,
                self.deep_drops, self.residual_projs):
            identity = res_proj(h)
            h = layer(h)
            h = norm(h)
            h = torch.relu(h)
            h = drop(h)
            h = h + identity  # residual connection

        deep_out = self.deep_out(h)

        # Combina wide + deep (classificazione)
        cls_logit = (wide_out + deep_out).squeeze(1)

        if self.predict_games:
            games_pred = self.games_head(h).squeeze(1)
            return cls_logit, games_pred

        return cls_logit


def label_smoothed_bce(logits, targets, smoothing=0.05):
    """BCEWithLogitsLoss con label smoothing: 0→smoothing/2, 1→1-smoothing/2"""
    targets_smooth = targets * (1.0 - smoothing) + 0.5 * smoothing
    return nn.functional.binary_cross_entropy_with_logits(logits, targets_smooth)


def train_model(model, loader_train, loader_val, epochs=80, lr=1e-3, patience=10,
                smoothing=0.05, alpha_games=ALPHA_GAMES):
    model.to(device)
    has_games = getattr(model, 'predict_games', False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=4)
    criterion_val = nn.BCEWithLogitsLoss()  # no smoothing per validazione
    best_val = float('inf'); best_state = None; no_imp = 0

    for _ in range(epochs):
        model.train()
        for batch in loader_train:
            Xb, yb = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            output = model(Xb)

            if has_games and isinstance(output, tuple):
                cls_logit, games_pred = output
                cls_loss = label_smoothed_bce(cls_logit, yb, smoothing=smoothing)
                # batch contiene [X, y, games_norm, games_mask]
                if len(batch) == 4:
                    gb = batch[2].to(device)
                    mb = batch[3].to(device).bool()
                    if mb.any():
                        games_loss = nn.functional.mse_loss(games_pred[mb], gb[mb])
                    else:
                        games_loss = torch.tensor(0.0, device=device)
                else:
                    games_loss = torch.tensor(0.0, device=device)
                loss = cls_loss + alpha_games * games_loss
            else:
                logits = output if not isinstance(output, tuple) else output[0]
                loss = label_smoothed_bce(logits, yb, smoothing=smoothing)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval(); vl = []
        with torch.no_grad():
            for batch in loader_val:
                Xb, yb = batch[0].to(device), batch[1].to(device)
                output = model(Xb)
                logits = output[0] if isinstance(output, tuple) else output
                vl.append(criterion_val(logits, yb).item())
        vl_mean = np.mean(vl); scheduler.step(vl_mean)
        if vl_mean < best_val - 1e-5:
            best_val = vl_mean
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience: break
    if best_state: model.load_state_dict(best_state)
    return model, best_val


def valuta(model, Xsc, y_np, games_raw=None, games_mean=0.0, games_std=1.0):
    """Valuta classificazione (acc, log_loss) e opzionalmente regressione games (MAE)."""
    model.eval()
    with torch.no_grad():
        output = model(torch.tensor(Xsc.astype(np.float32)).to(device))
    if isinstance(output, tuple):
        logits = output[0].cpu().numpy()
        games_pred_norm = output[1].cpu().numpy()
    else:
        logits = output.cpu().numpy()
        games_pred_norm = None
    probs = 1 / (1 + np.exp(-logits))
    acc = accuracy_score(y_np, (probs >= 0.5).astype(int))
    ll = log_loss(y_np, probs)

    games_mae = None
    if games_pred_norm is not None and games_raw is not None:
        games_pred = games_pred_norm * games_std + games_mean
        valid = ~np.isnan(games_raw)
        if valid.any():
            games_mae = float(np.mean(np.abs(games_pred[valid] - games_raw[valid])))
    return acc, ll, games_mae


# ══════════════════════════════════════════════════════════════════════════════
# 4.  OPTUNA SEARCH  (o random search se optuna non disponibile)
# ══════════════════════════════════════════════════════════════════════════════

def optuna_search(X_tr, y_tr, X_val_sc, X_te_sc,
                  y_val, y_test, dates_tr, level_w_tr,
                  n_trials=TRIALS, input_dim=len(FEATURES), label="Global",
                  games_tr=None, games_mask_tr=None,
                  games_val_raw=None, games_test_raw=None,
                  games_mean=0.0, games_std=1.0):

    X_val_t  = torch.tensor(X_val_sc.astype(np.float32))
    X_test_t = torch.tensor(X_te_sc.astype(np.float32))
    y_val_np = y_val.values if hasattr(y_val, 'values') else np.array(y_val)
    y_test_np= y_test.values if hasattr(y_test, 'values') else np.array(y_test)

    X_tr_t = torch.tensor(X_tr.astype(np.float32))
    y_tr_t = torch.tensor(y_tr.values.astype(np.float32) if hasattr(y_tr, 'values') else y_tr.astype(np.float32))

    # Dataset con games data (multi-task)
    has_games = games_tr is not None and games_mask_tr is not None
    if has_games:
        g_tr_t  = torch.tensor(games_tr.astype(np.float32))
        gm_tr_t = torch.tensor(games_mask_tr.astype(np.float32))
        dataset_tr  = TensorDataset(X_tr_t, y_tr_t, g_tr_t, gm_tr_t)
    else:
        dataset_tr  = TensorDataset(X_tr_t, y_tr_t)
    dataset_val = TensorDataset(X_val_t, torch.tensor(y_val_np.astype(np.float32)))

    USE_CUDA    = device.type == 'cuda'
    PIN_MEM     = USE_CUDA
    N_WORKERS   = 2 if USE_CUDA else 0

    risultati = []

    if HAS_OPTUNA:
        print(f"\n🔬 Optuna search [{label}]: {n_trials} trial...\n")

        def objective(trial):
            arch_name = trial.suggest_categorical('arch', list(ARCH_OPTIONS.keys()))
            hl  = ARCH_OPTIONS[arch_name]
            dr  = trial.suggest_float('dropout', 0.1, 0.5, step=0.05)
            lr  = trial.suggest_float('lr', 1e-4, 3e-3, log=True)
            bs  = trial.suggest_categorical('batch_size', [256, 512, 1024, 2048])
            ep  = trial.suggest_categorical('epochs', [60, 80, 100, 120])
            lam = trial.suggest_categorical('lambda_decay', [0.0001, 0.001, 0.003, 0.007, 0.015])
            sm  = trial.suggest_float('smoothing', 0.0, 0.1, step=0.01)
            iset = trial.suggest_categorical('interaction_set', list(INTERACTION_SETS.keys()))
            i_pairs = INTERACTION_SETS[iset]
            n_inter = len(i_pairs)

            # Stampa PRIMA di addestrare — aggiornamento immediato in console
            print(f"  [{trial.number+1:3d}/{n_trials}] arch={arch_name:12s} dr={dr:.2f} lr={lr:.0e} λ={lam:.4f} sm={sm:.2f} int={iset:14s}  ▶ training...", flush=True)

            # Se GPU, usa batch più grandi per tenerla satura
            bs_eff = bs * 2 if USE_CUDA and bs < 2048 else bs
            w_combined = calcola_pesi_combinati(dates_tr, level_w_tr, lam)
            w_t = torch.tensor(w_combined)
            sampler = WeightedRandomSampler(w_t, len(w_t), replacement=True)
            ldr_tr  = DataLoader(dataset_tr,  batch_size=bs_eff, sampler=sampler,
                                 pin_memory=PIN_MEM, num_workers=N_WORKERS)
            ldr_val = DataLoader(dataset_val, batch_size=4096, shuffle=False,
                                 pin_memory=PIN_MEM, num_workers=N_WORKERS)

            model = TennisANNv3(input_dim, hl, dr, n_inter, i_pairs,
                                predict_games=has_games)
            model, _ = train_model(model, ldr_tr, ldr_val, epochs=ep, lr=lr, smoothing=sm)
            acc_v, _, _ = valuta(model, X_val_sc, y_val_np)

            trial.set_user_attr('model', model)
            trial.set_user_attr('hl', hl)
            trial.set_user_attr('lam', lam)
            trial.set_user_attr('hp', {'hidden_layers': hl, 'dropout': dr, 'lr': lr,
                                       'batch_size': bs, 'epochs': ep, 'lambda_decay': lam,
                                       'label_smoothing': sm,
                                       'interaction_set': iset,
                                       'interaction_pairs': i_pairs,
                                       'n_interactions': n_inter,
                                       'predict_games': has_games})
            return acc_v

        study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(objective, n_trials=n_trials,
                       callbacks=[lambda s, t: print(
                           f"     └─ val_acc={t.value:.4f}  "
                           f"{'BEST! ⭐' if t.number == s.best_trial.number else ''}",
                           flush=True) if t.value is not None else None])

        for t in study.trials:
            if t.value is None: continue
            model_t = t.user_attrs.get('model')
            if model_t is None: continue
            acc_t, ll, mae_t = valuta(model_t, X_te_sc, y_test_np,
                                      games_test_raw, games_mean, games_std)
            entry = {
                'trial':        t.number+1,
                'hidden_layers': str(t.user_attrs.get('hl',[])),
                'dropout':      t.params.get('dropout'),
                'lr':           t.params.get('lr'),
                'batch_size':   t.params.get('batch_size'),
                'epochs':       t.params.get('epochs'),
                'lambda_decay': t.params.get('lambda_decay'),
                'smoothing':    t.params.get('smoothing'),
                'interaction_set': t.params.get('interaction_set'),
                'val_acc':      t.value,
                'test_acc':     acc_t,
                'log_loss':     ll,
                '_model':       model_t,
                '_config':      t.user_attrs.get('hp', {}),
            }
            if mae_t is not None:
                entry['games_mae'] = mae_t
            risultati.append(entry)

    else:
        # Fallback: random search
        print(f"\n🔍 Random search [{label}]: {n_trials} trial...\n")
        arch_keys = list(ARCH_OPTIONS.keys())
        for i in range(n_trials):
            arch_name = random.choice(arch_keys)
            hl  = ARCH_OPTIONS[arch_name]
            dr  = random.choice([0.1, 0.2, 0.3, 0.4])
            lr  = random.choice([1e-4, 3e-4, 1e-3, 3e-3])
            bs  = random.choice([256, 512, 1024, 2048])
            ep  = random.choice([60, 80, 100])
            lam = random.choice([0.0001, 0.001, 0.003, 0.007, 0.015])
            sm  = random.choice([0.0, 0.02, 0.05, 0.08, 0.1])
            iset = random.choice(list(INTERACTION_SETS.keys()))
            i_pairs = INTERACTION_SETS[iset]
            n_inter = len(i_pairs)
            hp  = {'hidden_layers': hl, 'dropout': dr, 'lr': lr,
                   'batch_size': bs, 'epochs': ep, 'lambda_decay': lam, 'label_smoothing': sm,
                   'interaction_set': iset, 'interaction_pairs': i_pairs, 'n_interactions': n_inter,
                   'predict_games': has_games}
            print(f"  [{i+1:3d}/{n_trials}] {str(hl):22s} drop={dr} lr={lr:.0e} λ={lam:.4f} sm={sm:.2f} int={iset}  ▶ avvio...", flush=True)
            w_combined = calcola_pesi_combinati(dates_tr, level_w_tr, lam)
            w_t = torch.tensor(w_combined)
            sampler = WeightedRandomSampler(w_t, len(w_t), replacement=True)
            ldr_tr  = DataLoader(dataset_tr,  batch_size=bs, sampler=sampler)
            ldr_val = DataLoader(dataset_val, batch_size=2048, shuffle=False)
            model = TennisANNv3(input_dim, hl, dr, n_inter, i_pairs,
                                predict_games=has_games)
            model, _ = train_model(model, ldr_tr, ldr_val, epochs=ep, lr=lr, smoothing=sm)
            acc_v, _, _ = valuta(model, X_val_sc, y_val_np)
            acc_t, ll, mae_t = valuta(model, X_te_sc, y_test_np,
                                      games_test_raw, games_mean, games_std)
            games_info = f" | mae_games={mae_t:.1f}" if mae_t else ""
            print(f"     └─ val={acc_v:.3f} test={acc_t:.3f}{games_info}", flush=True)
            entry = {'trial': i+1, 'hidden_layers': str(hl),
                     'dropout': dr, 'lr': lr, 'batch_size': bs,
                     'epochs': ep, 'lambda_decay': lam, 'smoothing': sm,
                     'interaction_set': iset,
                     'val_acc': acc_v, 'test_acc': acc_t,
                     'log_loss': ll, '_model': model, '_config': hp}
            if mae_t is not None:
                entry['games_mae'] = mae_t
            risultati.append(entry)

    return sorted(risultati, key=lambda x: x['test_acc'], reverse=True)


# ══════════════════════════════════════════════════════════════════════════════
# 4b.  LIGHTGBM + XGBOOST
# ══════════════════════════════════════════════════════════════════════════════

def train_lgb_optuna(X_tr, y_tr, X_val, y_val, X_test, y_test,
                     sample_weights_tr=None, n_trials=TRIALS_GBM):
    if not HAS_LGB:
        print("   ⚠️ LightGBM non disponibile, skip")
        return None, 0.0, float('inf')

    print(f"\n🌳 Training LightGBM (Optuna {n_trials} trials)...")
    y_tr_np = y_tr.values if hasattr(y_tr, 'values') else np.array(y_tr)
    y_val_np = y_val.values if hasattr(y_val, 'values') else np.array(y_val)
    y_test_np = y_test.values if hasattr(y_test, 'values') else np.array(y_test)

    best_model = [None]
    best_acc = [0.0]

    def objective(trial):
        params = {
            'objective': 'binary', 'metric': 'binary_logloss',
            'boosting_type': 'gbdt', 'verbosity': -1, 'seed': SEED,
            'n_estimators': trial.suggest_int('n_estimators', 200, 1500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10, log=True),
        }
        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr, y_tr_np, sample_weight=sample_weights_tr,
                  eval_set=[(X_val, y_val_np)],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                            lgb.log_evaluation(period=0)])
        y_pred = model.predict_proba(X_val)[:, 1]
        acc = accuracy_score(y_val_np, (y_pred >= 0.5).astype(int))
        if acc > best_acc[0]:
            best_acc[0] = acc
            best_model[0] = model
        return acc

    if HAS_OPTUNA:
        study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        model = best_model[0]
        print(f"   LightGBM best val: {best_acc[0]:.4f}")
    else:
        model = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05,
                                   max_depth=6, num_leaves=31, seed=SEED)
        model.fit(X_tr, y_tr_np, sample_weight=sample_weights_tr,
                  eval_set=[(X_val, y_val_np)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])

    y_pred_test = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test_np, (y_pred_test >= 0.5).astype(int))
    ll = log_loss(y_test_np, y_pred_test)
    print(f"   LightGBM test: acc={acc:.4f} | log_loss={ll:.4f}")
    return model, acc, ll


def train_xgb_optuna(X_tr, y_tr, X_val, y_val, X_test, y_test,
                     sample_weights_tr=None, n_trials=TRIALS_GBM):
    if not HAS_XGB:
        print("   ⚠️ XGBoost non disponibile, skip")
        return None, 0.0, float('inf')

    print(f"\n🌲 Training XGBoost (Optuna {n_trials} trials)...")
    y_tr_np = y_tr.values if hasattr(y_tr, 'values') else np.array(y_tr)
    y_val_np = y_val.values if hasattr(y_val, 'values') else np.array(y_val)
    y_test_np = y_test.values if hasattr(y_test, 'values') else np.array(y_test)

    best_model = [None]
    best_acc = [0.0]

    def objective(trial):
        params = {
            'objective': 'binary:logistic', 'eval_metric': 'logloss',
            'seed': SEED, 'verbosity': 0,
            'n_estimators': trial.suggest_int('n_estimators', 200, 1500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 50),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10, log=True),
            'gamma': trial.suggest_float('gamma', 0, 5),
        }
        model = xgb_lib.XGBClassifier(**params)
        model.fit(X_tr, y_tr_np, sample_weight=sample_weights_tr,
                  eval_set=[(X_val, y_val_np)],
                  verbose=False)
        y_pred = model.predict_proba(X_val)[:, 1]
        acc = accuracy_score(y_val_np, (y_pred >= 0.5).astype(int))
        if acc > best_acc[0]:
            best_acc[0] = acc
            best_model[0] = model
        return acc

    if HAS_OPTUNA:
        study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        model = best_model[0]
        print(f"   XGBoost best val: {best_acc[0]:.4f}")
    else:
        model = xgb_lib.XGBClassifier(n_estimators=500, learning_rate=0.05,
                                      max_depth=6, seed=SEED)
        model.fit(X_tr, y_tr_np, sample_weight=sample_weights_tr,
                  eval_set=[(X_val, y_val_np)], verbose=False)

    y_pred_test = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test_np, (y_pred_test >= 0.5).astype(int))
    ll = log_loss(y_test_np, y_pred_test)
    print(f"   XGBoost test: acc={acc:.4f} | log_loss={ll:.4f}")
    return model, acc, ll


def get_ann_probs(model, X_sc):
    """Get ANN prediction probabilities."""
    model.eval()
    with torch.no_grad():
        output = model(torch.tensor(X_sc.astype(np.float32)).to(device))
    logits = output[0].cpu().numpy() if isinstance(output, tuple) else output.cpu().numpy()
    return 1 / (1 + np.exp(-logits))


def get_ann_games(model, X_sc, games_mean=0.0, games_std=1.0):
    """Predice il numero totale di game dalla ANN (head di regressione)."""
    model.eval()
    with torch.no_grad():
        output = model(torch.tensor(X_sc.astype(np.float32)).to(device))
    if isinstance(output, tuple):
        games_norm = output[1].cpu().numpy()
        return games_norm * games_std + games_mean
    return None


# ══════════════════════════════════════════════════════════════════════════════
# 5.  CONFRONTO FINALE
# ══════════════════════════════════════════════════════════════════════════════

def confronto_finale(results_list):
    """results_list: list of {'Modello': str, 'Accuracy': float, 'Log Loss': float, 'Note': str}"""
    df_final = pd.DataFrame(results_list)
    df_final['Acc. %'] = (df_final['Accuracy'] * 100).round(2)
    df_final['Score'] = (df_final['Accuracy'] - df_final['Log Loss']).round(4)
    df_final = df_final.sort_values('Score', ascending=False).reset_index(drop=True)
    df_final.to_csv('resultados_comparacion_finale.csv', index=False)

    print("\n" + "="*80)
    print("  🏆  CLASSIFICA FINALE — TUTTI I MODELLI  (Score = Accuracy − Log Loss)")
    print("="*80)
    cols = ['Modello','Acc. %','Log Loss','Score']
    if 'Games MAE' in df_final.columns:
        cols.append('Games MAE')
    cols.append('Note')
    print(df_final[cols].to_string(index=False))
    print("="*80)
    vincitore = df_final.iloc[0]
    games_info = ""
    if 'Games MAE' in df_final.columns and pd.notna(vincitore.get('Games MAE')):
        games_info = f"  |  Games MAE {vincitore['Games MAE']:.2f}"
    print(f"\n  🥇  Miglior modello: {vincitore['Modello']}  →  Acc {vincitore['Acc. %']:.2f}%  |  Score {vincitore['Score']:.4f}{games_info}\n")
    return df_final


# ══════════════════════════════════════════════════════════════════════════════
# 7.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    CSV_CANDIDATES = [
        'historialTenis.csv', '../scraping/historialTenis.csv',
        'historial_tenis_COMPLETO.csv', '../scraping/historial_tenis_COMPLETO.csv',
    ]
    csv_path = next((p for p in CSV_CANDIDATES if os.path.exists(p)), None)
    if csv_path is None:
        raise FileNotFoundError("historialTenis.csv non trovato. Esegui da prediccion/")

    # ── 1. Feature engineering ────────────────────────────────────────────────
    df_ml, stats_dict, elo_surf, streak_t = carica_e_prepara(csv_path)

    X = df_ml[FEATURES].fillna(0)
    y = df_ml['target']
    dates = df_ml['tourney_date']
    level_w = df_ml['level_weight'].values

    # ── 1b. Preparazione target total games (regressione) ─────────────────────
    games_raw_all = df_ml['total_games'].values.astype(np.float32)
    games_valid_all = ~np.isnan(games_raw_all)
    games_mean = float(np.nanmean(games_raw_all))
    games_std  = float(np.nanstd(games_raw_all))
    # Normalizzazione: (games - mean) / std, NaN → 0
    games_norm_all = np.where(games_valid_all,
                              (games_raw_all - games_mean) / (games_std + 1e-8),
                              0.0).astype(np.float32)
    games_mask_all = games_valid_all.astype(np.float32)
    n_valid = int(games_valid_all.sum())
    print(f"   → Total games: {n_valid:,}/{len(games_raw_all):,} validi | media={games_mean:.1f} | std={games_std:.1f}")

    # ── 2. Split globale: 70 / 15 / 15 ───────────────────────────────────────
    X_tr, X_tmp, y_tr, y_tmp, d_tr, d_tmp, lw_tr, _, \
        g_tr, g_tmp, gm_tr, gm_tmp, gr_tr, gr_tmp = train_test_split(
        X, y, dates, level_w, games_norm_all, games_mask_all, games_raw_all,
        test_size=0.30, random_state=SEED)
    X_val, X_test, y_val, y_test, g_val, g_test, gm_val, gm_test, \
        gr_val, gr_test = train_test_split(
        X_tmp, y_tmp, g_tmp, gm_tmp, gr_tmp,
        test_size=0.50, random_state=SEED)

    scaler = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_tr)
    X_val_sc = scaler.transform(X_val)
    X_te_sc  = scaler.transform(X_test)
    joblib.dump(scaler, 'scaler_ann.pkl')
    print(f"   → scaler_ann.pkl salvato  |  {len(FEATURES)} feature")

    # ── 3. Optuna search globale ──────────────────────────────────────────────
    risultati = optuna_search(
        X_tr_sc, y_tr, X_val_sc, X_te_sc,
        y_val, y_test, d_tr, lw_tr,
        n_trials=TRIALS, input_dim=len(FEATURES), label="Global",
        games_tr=g_tr, games_mask_tr=gm_tr,
        games_val_raw=gr_val, games_test_raw=gr_test,
        games_mean=games_mean, games_std=games_std,
    )

    df_ris = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith('_')}
                           for r in risultati])
    df_ris.to_csv('resultados_ann.csv', index=False)

    best = risultati[0]
    best_hp = best['_config']
    acc_test, ll_test, mae_test = valuta(best['_model'], X_te_sc, y_test.values,
                                          gr_test, games_mean, games_std)

    print(f"\n🏆 Modello migliore (Optuna):")
    print(f"   Architettura:  {best['hidden_layers']}")
    print(f"   Hyperparams:   dropout={best['dropout']} | lr={best['lr']:.0e} | bs={best['batch_size']} | λ={best['lambda_decay']} | sm={best.get('smoothing', '?')} | int={best.get('interaction_set', '?')}")
    print(f"   Test Accuracy: {acc_test:.2%}")
    print(f"   Log Loss:      {ll_test:.4f}")
    if mae_test is not None:
        print(f"   Games MAE:     {mae_test:.2f}")
    top5_cols = ['hidden_layers','dropout','lr','lambda_decay','smoothing','interaction_set','val_acc','test_acc']
    if 'games_mae' in df_ris.columns:
        top5_cols.append('games_mae')
    top5_cols = [c for c in top5_cols if c in df_ris.columns]
    print(f"   Top 5 trial:\n{df_ris[top5_cols].head(5).to_string(index=False)}")

    # ── 3b. GBM models ──────────────────────────────────────────────────────
    combined_weights = calcola_pesi_combinati(d_tr, lw_tr, 0.003)

    lgb_model, lgb_acc, lgb_ll = train_lgb_optuna(
        X_tr_sc, y_tr, X_val_sc, y_val, X_te_sc, y_test,
        sample_weights_tr=combined_weights, n_trials=TRIALS_GBM)

    xgb_model, xgb_acc, xgb_ll = train_xgb_optuna(
        X_tr_sc, y_tr, X_val_sc, y_val, X_te_sc, y_test,
        sample_weights_tr=combined_weights, n_trials=TRIALS_GBM)

    # ── 3c. Ensemble ─────────────────────────────────────────────────────────
    from sklearn.linear_model import LogisticRegression

    y_val_np = y_val.values if hasattr(y_val, 'values') else np.array(y_val)
    y_test_np = y_test.values if hasattr(y_test, 'values') else np.array(y_test)

    best_ann = best['_model']
    ann_probs_val = get_ann_probs(best_ann, X_val_sc)
    ann_probs_test = get_ann_probs(best_ann, X_te_sc)

    # Top-5 ANN ensemble
    top5_ann_probs_val = np.mean([get_ann_probs(r['_model'], X_val_sc) for r in risultati[:5]], axis=0)
    top5_ann_probs_test = np.mean([get_ann_probs(r['_model'], X_te_sc) for r in risultati[:5]], axis=0)
    top5_ann_acc = accuracy_score(y_test_np, (top5_ann_probs_test >= 0.5).astype(int))
    top5_ann_ll = log_loss(y_test_np, top5_ann_probs_test)
    print(f"\n   ANN Top-5 ensemble: acc={top5_ann_acc:.4f} | log_loss={top5_ann_ll:.4f}")

    val_probs = {'ANN': ann_probs_val}
    test_probs = {'ANN': ann_probs_test}
    val_probs_top5 = {'ANN5': top5_ann_probs_val}
    test_probs_top5 = {'ANN5': top5_ann_probs_test}

    if lgb_model is not None:
        val_probs['LGB'] = lgb_model.predict_proba(X_val_sc)[:, 1]
        test_probs['LGB'] = lgb_model.predict_proba(X_te_sc)[:, 1]
        val_probs_top5['LGB'] = val_probs['LGB']
        test_probs_top5['LGB'] = test_probs['LGB']

    if xgb_model is not None:
        val_probs['XGB'] = xgb_model.predict_proba(X_val_sc)[:, 1]
        test_probs['XGB'] = xgb_model.predict_proba(X_te_sc)[:, 1]
        val_probs_top5['XGB'] = val_probs['XGB']
        test_probs_top5['XGB'] = test_probs['XGB']

    # Simple average ensemble
    avg_test = np.mean(list(test_probs.values()), axis=0)
    avg_acc = accuracy_score(y_test_np, (avg_test >= 0.5).astype(int))
    avg_ll = log_loss(y_test_np, avg_test)
    print(f"   Average ensemble: acc={avg_acc:.4f} | log_loss={avg_ll:.4f}")

    # Top-5 average ensemble
    avg_test_top5 = np.mean(list(test_probs_top5.values()), axis=0)
    avg_acc_top5 = accuracy_score(y_test_np, (avg_test_top5 >= 0.5).astype(int))
    avg_ll_top5 = log_loss(y_test_np, avg_test_top5)
    print(f"   Average ensemble (top5): acc={avg_acc_top5:.4f} | log_loss={avg_ll_top5:.4f}")

    # Stacking ensemble
    if len(val_probs) >= 2:
        val_meta = np.column_stack(list(val_probs.values()))
        test_meta = np.column_stack(list(test_probs.values()))
        meta_model = LogisticRegression(C=1.0, random_state=SEED, max_iter=2000)
        meta_model.fit(val_meta, y_val_np)
        stack_probs = meta_model.predict_proba(test_meta)[:, 1]
        stack_acc = accuracy_score(y_test_np, (stack_probs >= 0.5).astype(int))
        stack_ll = log_loss(y_test_np, stack_probs)
        print(f"   Stacking ensemble: acc={stack_acc:.4f} | log_loss={stack_ll:.4f}")
        joblib.dump(meta_model, 'modelo_meta_lr.pkl')
    else:
        stack_acc, stack_ll = avg_acc, avg_ll

    # ── 3d. Games MAE per il best ANN ──────────────────────────────────────
    best_games_mae = mae_test  # calcolato sopra da valuta()
    if best_games_mae is not None:
        print(f"\n   🎯 ANN Best — Games MAE: {best_games_mae:.2f} game")

    # Top-5 games MAE (media delle predizioni)
    top5_games_preds = [get_ann_games(r['_model'], X_te_sc, games_mean, games_std)
                        for r in risultati[:5]]
    top5_games_preds = [p for p in top5_games_preds if p is not None]
    top5_games_mae = None
    if top5_games_preds:
        avg_games_pred = np.mean(top5_games_preds, axis=0)
        valid_g = ~np.isnan(gr_test)
        if valid_g.any():
            top5_games_mae = float(np.mean(np.abs(avg_games_pred[valid_g] - gr_test[valid_g])))
            print(f"   🎯 ANN Top-5 — Games MAE: {top5_games_mae:.2f} game")

    # ── 3e. Results table ─────────────────────────────────────────────────────
    results_list = [
        {'Modello': 'ANN Best', 'Accuracy': acc_test, 'Log Loss': ll_test,
         'Games MAE': best_games_mae,
         'Note': f'Optuna best, arch={best["hidden_layers"]}',
         '_strategy': 'ann_best'},
        {'Modello': 'ANN Top-5 Avg', 'Accuracy': top5_ann_acc, 'Log Loss': top5_ann_ll,
         'Games MAE': top5_games_mae,
         'Note': 'Average of top 5 ANN trials',
         '_strategy': 'ann_top5'},
    ]
    if lgb_model is not None:
        results_list.append({'Modello': 'LightGBM', 'Accuracy': lgb_acc,
                            'Log Loss': lgb_ll, 'Note': f'Optuna {TRIALS_GBM} trials',
                            '_strategy': 'lgb'})
    if xgb_model is not None:
        results_list.append({'Modello': 'XGBoost', 'Accuracy': xgb_acc,
                            'Log Loss': xgb_ll, 'Note': f'Optuna {TRIALS_GBM} trials',
                            '_strategy': 'xgb'})
    results_list.append({'Modello': 'Ensemble Avg', 'Accuracy': avg_acc,
                        'Log Loss': avg_ll, 'Note': 'ANN+LGB+XGB average',
                        '_strategy': 'ensemble_avg'})
    results_list.append({'Modello': 'Ensemble Avg Top5', 'Accuracy': avg_acc_top5,
                        'Log Loss': avg_ll_top5, 'Note': 'ANN5+LGB+XGB average',
                        '_strategy': 'ensemble_avg_top5'})
    results_list.append({'Modello': 'Ensemble Stacking', 'Accuracy': stack_acc,
                        'Log Loss': stack_ll, 'Note': 'Meta-LR on ANN+LGB+XGB',
                        '_strategy': 'ensemble_stacking'})

    # ── 4. Selezione automatica e re-training su TUTTI i dati ─────────────────
    # Determina la strategia vincente (Score = Accuracy − Log Loss)
    for r in results_list:
        r['_score'] = r['Accuracy'] - r['Log Loss']
    winner = max(results_list, key=lambda r: r['_score'])
    best_strategy = winner['_strategy']
    best_model_name = winner['Modello']
    best_accuracy = winner['Accuracy']
    best_score = winner['_score']
    print(f"\n🏆 Strategia vincente: {best_model_name} (acc={best_accuracy:.2%}, score={best_score:.4f}) → strategia: {best_strategy}")

    print(f"\n🚀 Re-training modello finale su TUTTI i dati...")

    # Scaler finale (fit su tutti i dati per il predictor)
    scaler = StandardScaler()
    scaler.fit(X)
    joblib.dump(scaler, 'scaler_ann.pkl')

    # Split per early stopping: 85% train, 15% val (con games data)
    X_tr_fin, X_val_fin, y_tr_fin, y_val_fin, d_tr_fin, _, lw_tr_fin, _, \
        g_tr_fin, g_val_fin, gm_tr_fin, gm_val_fin = train_test_split(
        X, y, dates, level_w, games_norm_all, games_mask_all,
        test_size=0.15, random_state=SEED)

    X_tr_fin_sc  = scaler.transform(X_tr_fin)
    X_val_fin_sc = scaler.transform(X_val_fin)

    hl  = best_hp['hidden_layers']
    dr  = best_hp['dropout']
    lr  = best_hp['lr']
    bs  = best_hp['batch_size']
    ep  = best_hp['epochs']
    lam = best_hp['lambda_decay']
    sm  = best_hp.get('label_smoothing', 0.05)
    best_iset   = best_hp.get('interaction_set', 'core')
    best_ipairs = best_hp.get('interaction_pairs', DEFAULT_INTERACTION_PAIRS)
    best_ninter = best_hp.get('n_interactions', len(best_ipairs))

    X_tr_t = torch.tensor(X_tr_fin_sc.astype(np.float32))
    y_tr_t = torch.tensor(y_tr_fin.values.astype(np.float32) if hasattr(y_tr_fin, 'values')
                          else y_tr_fin.astype(np.float32))
    g_tr_t  = torch.tensor(g_tr_fin.astype(np.float32))
    gm_tr_t = torch.tensor(gm_tr_fin.astype(np.float32))
    X_val_t = torch.tensor(X_val_fin_sc.astype(np.float32))
    y_val_t = torch.tensor(y_val_fin.values.astype(np.float32) if hasattr(y_val_fin, 'values')
                           else y_val_fin.astype(np.float32))

    w_combined = calcola_pesi_combinati(d_tr_fin, lw_tr_fin, lam)
    w_t = torch.tensor(w_combined)
    sampler = WeightedRandomSampler(w_t, len(w_t), replacement=True)

    USE_CUDA = device.type == 'cuda'
    PIN_MEM  = USE_CUDA
    N_WORK   = 2 if USE_CUDA else 0
    bs_eff   = bs * 2 if USE_CUDA and bs < 2048 else bs

    ldr_tr  = DataLoader(TensorDataset(X_tr_t, y_tr_t, g_tr_t, gm_tr_t),
                         batch_size=bs_eff,
                         sampler=sampler, pin_memory=PIN_MEM, num_workers=N_WORK)
    ldr_val = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=4096,
                         shuffle=False, pin_memory=PIN_MEM, num_workers=N_WORK)

    model_final = TennisANNv3(len(FEATURES), hl, dr, best_ninter, best_ipairs,
                              predict_games=True)
    model_final, _ = train_model(model_final, ldr_tr, ldr_val, epochs=ep, lr=lr,
                                  smoothing=sm)

    # Re-train modelli aggiuntivi solo se la strategia vincente li richiede
    needs_top5 = best_strategy in ('ann_top5', 'ensemble_avg_top5')
    needs_gbm  = best_strategy in ('lgb', 'xgb', 'ensemble_avg', 'ensemble_avg_top5', 'ensemble_stacking')

    top5_state_dicts = [model_final.state_dict()]
    top5_configs     = [best_hp]
    if needs_top5:
        for idx in range(1, min(5, len(risultati))):
            hp_i = risultati[idx]['_config']
            hl_i  = hp_i['hidden_layers']
            dr_i  = hp_i['dropout']
            lr_i  = hp_i['lr']
            bs_i  = hp_i['batch_size']
            ep_i  = hp_i['epochs']
            sm_i  = hp_i.get('label_smoothing', 0.05)
            ipairs_i = hp_i.get('interaction_pairs', DEFAULT_INTERACTION_PAIRS)
            ninter_i = hp_i.get('n_interactions', len(ipairs_i))

            print(f"   Re-training ANN #{idx+1} (arch={hl_i})...")
            bs_eff_i = bs_i * 2 if USE_CUDA and bs_i < 2048 else bs_i
            w_i = torch.tensor(w_combined)
            sampler_i = WeightedRandomSampler(w_i, len(w_i), replacement=True)
            ldr_tr_i = DataLoader(TensorDataset(X_tr_t, y_tr_t, g_tr_t, gm_tr_t),
                                  batch_size=bs_eff_i,
                                  sampler=sampler_i, pin_memory=PIN_MEM, num_workers=N_WORK)
            model_i = TennisANNv3(len(FEATURES), hl_i, dr_i, ninter_i, ipairs_i,
                                  predict_games=True)
            model_i, _ = train_model(model_i, ldr_tr_i, ldr_val, epochs=ep_i, lr=lr_i,
                                      smoothing=sm_i)
            top5_state_dicts.append(model_i.state_dict())
            top5_configs.append(hp_i)

    # Re-train GBM on all data (se necessario)
    y_all_np = y.values if hasattr(y, 'values') else np.array(y)
    all_weights = calcola_pesi_combinati(dates, level_w, 0.003)
    lgb_final = None; xgb_final = None

    if needs_gbm and lgb_model is not None:
        print("   Re-training LightGBM on all data...")
        lgb_final = lgb.LGBMClassifier(**lgb_model.get_params())
        lgb_final.fit(scaler.transform(X), y_all_np, sample_weight=all_weights)

    if needs_gbm and xgb_model is not None:
        print("   Re-training XGBoost on all data...")
        xgb_final = xgb_lib.XGBClassifier(**xgb_model.get_params())
        xgb_final.set_params(early_stopping_rounds=None)
        xgb_final.fit(scaler.transform(X), y_all_np, sample_weight=all_weights)

    # ── 5. Costruzione modelo_finale.pkl ──────────────────────────────────────
    modelo_finale = {
        'strategy': best_strategy,
        'model_name': best_model_name,
        'accuracy': best_accuracy,
        'score': best_score,
        'features': FEATURES,
        'scaler': scaler,
        # Multi-task: predizione total games
        'games_mean': games_mean,
        'games_std': games_std,
        'predict_games': True,
    }

    # L'ANN viene sempre salvata per la predizione total games (multi-task)
    modelo_finale['ann'] = {
        'state_dict': model_final.state_dict(),
        'config': best_hp,
    }

    if best_strategy == 'ann_top5':
        modelo_finale['ann_top5'] = [
            {'state_dict': sd, 'config': c}
            for sd, c in zip(top5_state_dicts, top5_configs)
        ]

    elif best_strategy == 'lgb':
        modelo_finale['lgb_model'] = lgb_final

    elif best_strategy == 'xgb':
        modelo_finale['xgb_model'] = xgb_final

    elif best_strategy == 'ensemble_avg':
        modelo_finale['lgb_model'] = lgb_final
        modelo_finale['xgb_model'] = xgb_final

    elif best_strategy == 'ensemble_avg_top5':
        modelo_finale['ann_top5'] = [
            {'state_dict': sd, 'config': c}
            for sd, c in zip(top5_state_dicts, top5_configs)
        ]
        modelo_finale['lgb_model'] = lgb_final
        modelo_finale['xgb_model'] = xgb_final

    elif best_strategy == 'ensemble_stacking':
        modelo_finale['lgb_model'] = lgb_final
        modelo_finale['xgb_model'] = xgb_final
        modelo_finale['meta_model'] = meta_model

    joblib.dump(modelo_finale, 'modelo_finale.pkl')
    print(f"\n   → modelo_finale.pkl salvato (strategia: {best_strategy})")

    # ── 6. Confronto finale ───────────────────────────────────────────────────
    # Strip internal keys before printing
    results_clean = [{k: v for k, v in r.items() if not k.startswith('_')} for r in results_list]
    df_confronto = confronto_finale(results_clean)

    print("\n✅ File salvati:")
    for f in ['modelo_finale.pkl', 'scaler_ann.pkl',
              'elo_surface.pkl', 'elo_overall.pkl', 'streak_players.pkl',
              'momentum_surface.pkl', 'recent_form.pkl',
              'resultados_comparacion_finale.csv']:
        stato = "✅" if os.path.exists(f) else "—"
        print(f"  {stato}  {f}")
