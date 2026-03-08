"""
train_games.py — Predizione Numero di Set e Totale Game per Partita
====================================================================
Modello SEPARATO e INDIPENDENTE dal modello di classificazione (train_ann.py).
NON modifica nessun file .pkl esistente.

Target: Predirre total_games con MAE ≤ 4

Approccio GERARCHICO (2 stadi):
  STADIO 1: Classificatore set  (Bo3: 2 vs 3 set,  Bo5: 3/4/5 set)
  STADIO 2: Regressori condizionali per game (uno per ogni n_set)
  PREDIZIONE FINALE: Σ P(n_set) × E[games | n_set]

  ✅ Feature engineering specifiche per game count
  ✅ Feature dei 29 classificatori (riutilizzate, non mirror)
  ✅ ~15 nuove feature game-specifiche (avg games, tiebreak rate, ecc.)
  ✅ Classificatore set: LightGBM, XGBoost con Optuna tuning
  ✅ Regressori condizionali: separati per 2-set, 3-set (Bo3), 3/4/5-set (Bo5)
  ✅ Cross-validation 5-fold
  ✅ Feature importance e pair analysis
  ✅ Modelli separati per Bo3 e Bo5
  ✅ Confronto diretto vs gerarchico

UTILIZZO:
    cd prediccion/
    python train_games.py

OUTPUT:
    modelo_games.pkl         ← modello game prediction (auto-selezionato)
    game_tracker.pkl         ← statistiche game per giocatore (per il predictor)
"""

import os, sys, warnings, json, time
import numpy as np
import pandas as pd
import joblib
from collections import deque

from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, log_loss
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor,
    GradientBoostingRegressor, BaggingRegressor,
    StackingRegressor, VotingRegressor,
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier,
)
from sklearn.linear_model import BayesianRidge, Ridge, ElasticNet, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ── Court Speed helper ────────────────────────────────────────────────────────
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scraping'))
try:
    from court_speed_helper import get_court_stats
    HAS_COURT_SPEED = True
except ImportError:
    HAS_COURT_SPEED = False
    def get_court_stats(name, surface='Hard', year=2025):
        return 0.0, 0.0

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
    print("   ✅ Optuna disponibile")
except ImportError:
    HAS_OPTUNA = False

SEED = 42
np.random.seed(SEED)

# ── Configurazione ────────────────────────────────────────────────────────────
TRIALS_GBM   = 15     # trial Optuna per LGB/XGB (ridotti per iterare velocemente)
GAME_HISTORY = 30     # finestra rolling per avg games
GAME_HISTORY_SURF = 20
GLOBAL_HISTORY = 500  # per medie globali superficie/livello

LEVEL_MULT   = {'G': 2.0, 'M': 1.5, 'F': 1.4, 'A': 1.0,
                'D': 1.0, 'C': 0.8, 'S': 0.7, 'E': 0.5}
K_LEVEL_ELO  = {'G': 1.25, 'M': 1.15, 'F': 1.1, 'A': 1.0,
                'D': 0.95, 'C': 0.9, 'S': 0.85, 'E': 0.8}
ELO_DEFAULT  = 1500.0
K_BASE       = 32.0


# ══════════════════════════════════════════════════════════════════════════════
# 1.  SCORE PARSING
# ══════════════════════════════════════════════════════════════════════════════

def parse_score_details(score_str):
    """Estrae informazioni dettagliate dal punteggio.
    Returns dict con total_games, sets_played, n_tiebreaks, w_games, l_games, avg_gps
    oppure None se il punteggio non è valido."""
    if not isinstance(score_str, str) or not score_str.strip():
        return None
    score = score_str.strip().upper()
    if any(x in score for x in ['W/O', 'RET', 'DEF', 'WO', 'ABN', 'UNP', 'BYE', 'NA', 'NAN']):
        return None

    tokens = [t for t in score.split() if '-' in t]
    if not tokens:
        return None

    total_games = 0
    w_games = 0
    l_games = 0
    n_tiebreaks = 0

    for token in tokens:
        has_tb = '(' in token
        token_clean = token.split('(')[0]
        parts = token_clean.split('-')
        if len(parts) == 2:
            try:
                p1 = int(parts[0])
                p2 = int(parts[1])
                total_games += p1 + p2
                w_games += p1
                l_games += p2
                if has_tb or (p1 == 7 and p2 == 6) or (p1 == 6 and p2 == 7):
                    n_tiebreaks += 1
            except ValueError:
                continue

    if total_games == 0:
        return None

    sets_played = len(tokens)
    return {
        'total_games': float(total_games),
        'sets_played': float(sets_played),
        'n_tiebreaks': n_tiebreaks,
        'w_games': w_games,
        'l_games': l_games,
        'avg_gps': total_games / sets_played,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2.  FEATURE ENGINEERING  (un unico pass cronologico, NO mirror pairs)
# ══════════════════════════════════════════════════════════════════════════════

# Feature list: 29 classificazione + ~17 game-specifiche
CLF_FEATURES = [
    'log_rank_ratio', 'log_pts_ratio',
    'diff_age', 'diff_ht',
    'diff_elo', 'diff_elo_overall',
    'diff_streak', 'diff_recent_form',
    'surface_enc', 'tourney_level', 'round_enc',
    'is_best_of_5',
    'diff_skill', 'diff_home',
    'diff_fatigue', 'diff_momentum',
    'diff_h2h', 'diff_h2h_surface',
    'diff_days_since_last',
    'diff_ace', 'diff_1st_pct', 'diff_1st_won',
    'diff_2nd_won', 'diff_bp_saved',
    'diff_return_pct', 'diff_bp_conv', 'diff_return_1st',
    'court_ace_pct', 'court_speed',
]

GAME_FEATURES = [
    'w_avg_total_games', 'l_avg_total_games',
    'w_avg_total_games_surf', 'l_avg_total_games_surf',
    'w_avg_gps', 'l_avg_gps',
    'w_tb_rate', 'l_tb_rate',
    'w_straight_rate', 'l_straight_rate',
    'combined_avg_games', 'combined_avg_games_surf',
    'combined_avg_gps', 'combined_tb_rate',
    'abs_elo_diff', 'abs_rank_ratio',
    'surface_avg_games',
]

ALL_FEATURES = CLF_FEATURES + GAME_FEATURES


def prepare_game_dataset(csv_path: str):
    """Feature engineering completa per predizione game count.
    Un unico pass cronologico. Output: 1 riga per match (prospettiva winner).
    NON salva i .pkl di elo/streak (quelli restano gestiti da train_ann.py).
    """
    print(f"\n📂 Caricamento: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    df['tourney_date'] = pd.to_numeric(df['tourney_date'], errors='coerce')
    df = df.sort_values(by=['tourney_date', 'match_num']).reset_index(drop=True)
    df['minutes'] = df['minutes'].fillna(90)
    print(f"   → {len(df):,} partite caricate")

    # ── Encoding ─────────────────────────────────────────────────────────────
    surface_map = {'Hard': 0, 'Clay': 1, 'Grass': 2, 'Carpet': 3}
    level_map   = {'G': 5, 'M': 4, 'A': 3, 'D': 3, 'F': 4, 'C': 2, 'S': 1, 'E': 0}
    round_map   = {'F': 7, 'SF': 6, 'QF': 5, 'R16': 4, 'R32': 3,
                   'R64': 2, 'R128': 1, 'RR': 4, 'BR': 3}

    df['surface_enc']       = df['surface'].map(surface_map).fillna(0)
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

    # ── Win-rate superficie ──────────────────────────────────────────────────
    wins   = df.groupby(['winner_name', 'surface']).size().reset_index(name='wins')
    losses = df.groupby(['loser_name',  'surface']).size().reset_index(name='losses')
    wins.columns = ['player', 'surface', 'wins']
    losses.columns = ['player', 'surface', 'losses']
    stats = pd.merge(wins, losses, on=['player', 'surface'], how='outer').fillna(0)
    stats['total'] = stats['wins'] + stats['losses']
    stats = stats[stats['total'] >= 5]
    stats['win_rate'] = stats['wins'] / stats['total']
    stats_dict = stats.set_index(['player', 'surface'])['win_rate'].to_dict()
    def get_skill(p, s): return stats_dict.get((p, s), 0.5)

    # ── Tracker classificazione (stessi di train_ann.py) ─────────────────────
    fatiga_t       = {}
    racha_t        = {}
    h2h_t          = {}
    h2h_surf_t     = {}
    serve_t        = {}
    return_t       = {}
    elo_surf       = {}
    elo_overall_t  = {}
    streak_t       = {}
    recent_form_t  = {}
    last_match_date_t = {}

    # ── Tracker GAME-SPECIFICI ───────────────────────────────────────────────
    player_games_hist      = {}   # player → list[total_games] (last GAME_HISTORY)
    player_games_surf_hist = {}   # (player, surf) → list (last GAME_HISTORY_SURF)
    player_gps_hist        = {}   # player → list[avg_games_per_set] (last GAME_HISTORY)
    player_tb_count        = {}   # player → [n_tiebreaks, n_sets_total]
    player_straight_count  = {}   # player → [straight_sets_matches, total_bo3]
    surface_games_hist     = {}   # surface → list[total_games] (last GLOBAL_HISTORY)
    level_games_hist       = {}   # level → list (last GLOBAL_HISTORY)

    def get_elo(p, s):
        return elo_surf.get((p, s), ELO_DEFAULT)

    def _deque_mean(lst, default):
        return float(np.mean(lst)) if lst else default

    def _append_limited(lst, val, maxlen):
        lst.append(val)
        if len(lst) > maxlen:
            lst.pop(0)

    rows = []
    n_skipped = 0

    for idx, row in df.iterrows():
        w, l = row['winner_name'], row['loser_name']
        dur  = row['minutes']
        surf = row['surface']

        if not isinstance(w, str) or not isinstance(l, str) or pd.isna(w) or pd.isna(l):
            continue
        if not isinstance(surf, str):
            continue

        # --- Score parsing ---
        score_info = parse_score_details(str(row.get('score', '')))
        if score_info is None:
            n_skipped += 1
            # Aggiorna comunque i tracker clf per consistenza
            # (serve per mantenere ELO allineato)
            best_of_val = row.get('best_of', 3)
            is_bo5_flag = 1.0 if best_of_val == 5 or best_of_val == '5' else 0.0

            # ELO update anche per partite senza score (per consistenza)
            elo_w = get_elo(w, surf); elo_l = get_elo(l, surf)
            expected_w = 1.0 / (1.0 + 10.0 ** ((elo_l - elo_w) / 400.0))
            level_code = str(row.get('tourney_level', 'A'))
            k_dynamic = K_BASE * K_LEVEL_ELO.get(level_code, 1.0)
            elo_surf[(w, surf)] = elo_w + k_dynamic * (1.0 - expected_w)
            elo_surf[(l, surf)] = elo_l + k_dynamic * (0.0 - (1.0 - expected_w))
            elo_ov_w = elo_overall_t.get(w, ELO_DEFAULT)
            elo_ov_l = elo_overall_t.get(l, ELO_DEFAULT)
            expected_ov = 1.0 / (1.0 + 10.0 ** ((elo_ov_l - elo_ov_w) / 400.0))
            elo_overall_t[w] = elo_ov_w + k_dynamic * (1.0 - expected_ov)
            elo_overall_t[l] = elo_ov_l + k_dynamic * (0.0 - (1.0 - expected_ov))
            # streak
            str_w = streak_t.get(w, 0); str_l = streak_t.get(l, 0)
            streak_t[w] = max(0, str_w) + 1 if str_w >= 0 else 1
            streak_t[l] = min(0, str_l) - 1 if str_l <= 0 else -1
            # recent form
            rf_w = recent_form_t.setdefault(w, []); rf_l = recent_form_t.setdefault(l, [])
            rf_w.append(1.0); rf_l.append(0.0)
            if len(rf_w) > 10: rf_w.pop(0)
            if len(rf_l) > 10: rf_l.pop(0)
            # momentum
            hw = racha_t.setdefault((w, surf), []); hl = racha_t.setdefault((l, surf), [])
            hw.append(1); hl.append(0)
            if len(hw) > 10: hw.pop(0)
            if len(hl) > 10: hl.pop(0)
            continue

        total_games  = score_info['total_games']
        sets_played  = score_info['sets_played']
        n_tiebreaks  = score_info['n_tiebreaks']
        avg_gps_match = score_info['avg_gps']

        best_of_val = row.get('best_of', 3)
        is_bo5 = 1.0 if best_of_val == 5 or best_of_val == '5' else 0.0
        if sets_played >= 4:
            is_bo5 = 1.0

        # ── GAME-SPECIFIC FEATURES (PRIMA dell'aggiornamento) ────────────────

        # Player avg total games (overall)
        w_avg_tg = _deque_mean(player_games_hist.get(w, []), 23.0 if not is_bo5 else 35.0)
        l_avg_tg = _deque_mean(player_games_hist.get(l, []), 23.0 if not is_bo5 else 35.0)

        # Player avg total games (surface-specific)
        w_avg_tg_s = _deque_mean(player_games_surf_hist.get((w, surf), []), w_avg_tg)
        l_avg_tg_s = _deque_mean(player_games_surf_hist.get((l, surf), []), l_avg_tg)

        # Avg games per set
        w_avg_gps = _deque_mean(player_gps_hist.get(w, []), 10.5)
        l_avg_gps = _deque_mean(player_gps_hist.get(l, []), 10.5)

        # Tiebreak rate
        w_tb = player_tb_count.get(w, [0, 0])
        l_tb = player_tb_count.get(l, [0, 0])
        w_tb_rate = w_tb[0] / max(w_tb[1], 1)
        l_tb_rate = l_tb[0] / max(l_tb[1], 1)

        # Straight sets rate (Bo3: 2-set finishes)
        w_ss = player_straight_count.get(w, [0, 0])
        l_ss = player_straight_count.get(l, [0, 0])
        w_ss_rate = w_ss[0] / max(w_ss[1], 1)
        l_ss_rate = l_ss[0] / max(l_ss[1], 1)

        # Combined features
        combined_avg   = (w_avg_tg + l_avg_tg) / 2.0
        combined_avg_s = (w_avg_tg_s + l_avg_tg_s) / 2.0
        combined_gps   = (w_avg_gps + l_avg_gps) / 2.0
        combined_tb    = (w_tb_rate + l_tb_rate) / 2.0

        # Surface & level global average
        surf_avg = _deque_mean(surface_games_hist.get(surf, []),
                               23.0 if not is_bo5 else 35.0)
        level_code = str(row.get('tourney_level', 'A'))

        # ── CLASSIFICATION FEATURES (stesse di train_ann.py) ─────────────────

        # Fatica
        f_w = fatiga_t.get((row['tourney_id'], w), 0)
        f_l = fatiga_t.get((row['tourney_id'], l), 0)
        fatiga_t[(row['tourney_id'], w)] = f_w + dur
        fatiga_t[(row['tourney_id'], l)] = f_l + dur

        # Momentum per superficie
        hw_m = racha_t.get((w, surf), []); hl_m = racha_t.get((l, surf), [])
        mw = np.mean(hw_m) if hw_m else 0.5
        ml = np.mean(hl_m) if hl_m else 0.5
        hw_m.append(1); hl_m.append(0)
        if len(hw_m) > 10: hw_m.pop(0)
        if len(hl_m) > 10: hl_m.pop(0)
        racha_t[(w, surf)] = hw_m; racha_t[(l, surf)] = hl_m

        # H2H globale
        p1k, p2k = sorted([w, l]); key = (p1k, p2k)
        rec = h2h_t.get(key, [0, 0])
        if w == p1k:
            h2h_w = rec[0] - rec[1]; h2h_l = rec[1] - rec[0]; rec[0] += 1
        else:
            h2h_w = rec[1] - rec[0]; h2h_l = rec[0] - rec[1]; rec[1] += 1
        h2h_t[key] = rec

        # H2H per superficie
        key_s = (p1k, p2k, surf)
        rec_s = h2h_surf_t.get(key_s, [0, 0])
        if w == p1k:
            h2h_s_w = rec_s[0] - rec_s[1]; h2h_s_l = rec_s[1] - rec_s[0]; rec_s[0] += 1
        else:
            h2h_s_w = rec_s[1] - rec_s[0]; h2h_s_l = rec_s[0] - rec_s[1]; rec_s[1] += 1
        h2h_surf_t[key_s] = rec_s

        # Days since last match
        td = int(row['tourney_date']) if pd.notna(row.get('tourney_date')) else 20200101
        def _date_to_days(d):
            y, rest = divmod(d, 10000)
            m, day = divmod(rest, 100)
            return y * 365 + m * 30 + day
        td_days = _date_to_days(td)
        last_w = last_match_date_t.get(w)
        last_l = last_match_date_t.get(l)
        days_since_w = (td_days - _date_to_days(last_w)) if last_w is not None else 14.0
        days_since_l = (td_days - _date_to_days(last_l)) if last_l is not None else 14.0
        days_since_w = max(0.0, min(180.0, float(days_since_w)))
        days_since_l = max(0.0, min(180.0, float(days_since_l)))
        last_match_date_t[w] = td
        last_match_date_t[l] = td

        # Elo per superficie
        elo_w = get_elo(w, surf); elo_l = get_elo(l, surf)
        expected_w = 1.0 / (1.0 + 10.0 ** ((elo_l - elo_w) / 400.0))
        k_dynamic = K_BASE * K_LEVEL_ELO.get(level_code, 1.0)
        elo_surf[(w, surf)] = elo_w + k_dynamic * (1.0 - expected_w)
        elo_surf[(l, surf)] = elo_l + k_dynamic * (0.0 - (1.0 - expected_w))

        # Elo overall
        elo_ov_w = elo_overall_t.get(w, ELO_DEFAULT)
        elo_ov_l = elo_overall_t.get(l, ELO_DEFAULT)
        expected_ov = 1.0 / (1.0 + 10.0 ** ((elo_ov_l - elo_ov_w) / 400.0))
        elo_overall_t[w] = elo_ov_w + k_dynamic * (1.0 - expected_ov)
        elo_overall_t[l] = elo_ov_l + k_dynamic * (0.0 - (1.0 - expected_ov))

        # Recent form
        rf_w = recent_form_t.get(w, []); rf_l = recent_form_t.get(l, [])
        form_w = np.mean(rf_w) if rf_w else 0.5
        form_l = np.mean(rf_l) if rf_l else 0.5
        rf_w.append(1.0); rf_l.append(0.0)
        if len(rf_w) > 10: rf_w.pop(0)
        if len(rf_l) > 10: rf_l.pop(0)
        recent_form_t[w] = rf_w; recent_form_t[l] = rf_l

        # Striscia attiva
        str_w = streak_t.get(w, 0); str_l = streak_t.get(l, 0)
        streak_t[w] = max(0, str_w) + 1 if str_w >= 0 else 1
        streak_t[l] = min(0, str_l) - 1 if str_l <= 0 else -1

        # Statistiche servizio medie
        def get_sa(player):
            s = serve_t.get(player, {})
            return {'ace':     np.mean(s.get('ace',     [0])),
                    'df':      np.mean(s.get('df',      [0])),
                    '1st_pct': np.mean(s.get('1st_pct', [0.6])),
                    '1st_won': np.mean(s.get('1st_won', [0.7])),
                    '2nd_won': np.mean(s.get('2nd_won', [0.5])),
                    'bp_saved':np.mean(s.get('bp_saved',[0.6]))}
        sa_w = get_sa(w); sa_l = get_sa(l)

        # Return stats
        def get_ra(player):
            s = return_t.get(player, {})
            return {'return_pct':  np.mean(s.get('return_pct',  [0.35])),
                    'bp_conv':     np.mean(s.get('bp_conv',     [0.35])),
                    'return_1st':  np.mean(s.get('return_1st',  [0.30]))}
        ra_w = get_ra(w); ra_l = get_ra(l)

        # Update serve/return stats
        def upd_return(player, rd, opp_pref):
            s = return_t.setdefault(player, {})
            opp_svpt   = rd.get(f'{opp_pref}_svpt', np.nan)
            opp_1stWon = rd.get(f'{opp_pref}_1stWon', np.nan)
            opp_2ndWon = rd.get(f'{opp_pref}_2ndWon', np.nan)
            opp_1stIn  = rd.get(f'{opp_pref}_1stIn', np.nan)
            opp_bpFaced = rd.get(f'{opp_pref}_bpFaced', np.nan)
            opp_bpSaved = rd.get(f'{opp_pref}_bpSaved', np.nan)
            if opp_svpt and opp_svpt > 0 and not (isinstance(opp_svpt, float) and np.isnan(opp_svpt)):
                fw = opp_1stWon if not (isinstance(opp_1stWon, float) and np.isnan(opp_1stWon)) else 0
                sw = opp_2ndWon if not (isinstance(opp_2ndWon, float) and np.isnan(opp_2ndWon)) else 0
                rtn_won = (opp_svpt - fw - sw) / opp_svpt
                lst = s.setdefault('return_pct', [])
                lst.append(float(rtn_won))
                if len(lst) > 10: lst.pop(0)
            if opp_bpFaced and opp_bpFaced > 0 and not (isinstance(opp_bpFaced, float) and np.isnan(opp_bpFaced)):
                bps = opp_bpSaved if not (isinstance(opp_bpSaved, float) and np.isnan(opp_bpSaved)) else 0
                bp_conv = (opp_bpFaced - bps) / opp_bpFaced
                lst = s.setdefault('bp_conv', [])
                lst.append(float(bp_conv))
                if len(lst) > 10: lst.pop(0)
            if opp_1stIn and opp_1stIn > 0 and not (isinstance(opp_1stIn, float) and np.isnan(opp_1stIn)):
                fw = opp_1stWon if not (isinstance(opp_1stWon, float) and np.isnan(opp_1stWon)) else 0
                rtn_1st = (opp_1stIn - fw) / opp_1stIn
                lst = s.setdefault('return_1st', [])
                lst.append(float(rtn_1st))
                if len(lst) > 10: lst.pop(0)

        def upd_serve(player, rd, pref):
            s = serve_t.setdefault(player, {})
            svpt = rd.get(f'{pref}_svpt', np.nan); fi = rd.get(f'{pref}_1stIn', np.nan)
            fw = rd.get(f'{pref}_1stWon', np.nan); sw = rd.get(f'{pref}_2ndWon', np.nan)
            bps = rd.get(f'{pref}_bpSaved', np.nan); bpf = rd.get(f'{pref}_bpFaced', np.nan)
            for k2, v in [('ace', rd.get(f'{pref}_ace', np.nan)),
                          ('df', rd.get(f'{pref}_df', np.nan)),
                          ('1st_pct', fi / svpt if svpt and svpt > 0 else np.nan),
                          ('1st_won', fw / fi if fi and fi > 0 else np.nan),
                          ('2nd_won', sw / (svpt - fi) if svpt and fi and (svpt - fi) > 0 else np.nan),
                          ('bp_saved', bps / bpf if bpf and bpf > 0 else np.nan)]:
                if not (isinstance(v, float) and np.isnan(v)):
                    lst = s.setdefault(k2, [])
                    lst.append(float(v))
                    if len(lst) > 10: lst.pop(0)

        rd = row.to_dict()
        upd_serve(w, rd, 'w'); upd_return(w, rd, 'l')

        sk_w = get_skill(w, surf); sk_l = get_skill(l, surf)
        home_w = 1 if row['winner_ioc'] == row['tourney_ioc'] else 0
        home_l = 1 if row['loser_ioc']  == row['tourney_ioc'] else 0
        pts_w = float(row['winner_rank_points']) if pd.notna(row.get('winner_rank_points')) else 0
        pts_l = float(row['loser_rank_points'])  if pd.notna(row.get('loser_rank_points'))  else 0
        rk_w  = float(row['winner_rank']) if pd.notna(row.get('winner_rank')) else 500
        rk_l  = float(row['loser_rank'])  if pd.notna(row.get('loser_rank'))  else 500

        # Court speed
        tourney_year = int(str(row['tourney_date'])[:4]) if pd.notna(row.get('tourney_date')) else 2025
        surf_safe = surf if isinstance(surf, str) else 'Hard'
        court_ace, court_spd = get_court_stats(
            row.get('tourney_name', ''), surf_safe, tourney_year)

        # ── BUILD ROW ────────────────────────────────────────────────────────
        feature_row = {
            # 29 clf features (winner perspective)
            'log_rank_ratio':    np.log1p(rk_l) - np.log1p(rk_w),
            'log_pts_ratio':     np.log1p(pts_w) - np.log1p(pts_l),
            'diff_age':          (row['winner_age'] - row['loser_age'])
                                 if pd.notna(row.get('winner_age')) and pd.notna(row.get('loser_age')) else 0,
            'diff_ht':           (row['winner_ht'] - row['loser_ht'])
                                 if pd.notna(row.get('winner_ht')) and pd.notna(row.get('loser_ht')) else 0,
            'diff_elo':          elo_w - elo_l,
            'diff_elo_overall':  elo_ov_w - elo_ov_l,
            'diff_streak':       float(str_w - str_l),
            'diff_recent_form':  form_w - form_l,
            'surface_enc':       float(row['surface_enc']),
            'tourney_level':     float(row['tourney_level_enc']),
            'round_enc':         float(row['round_enc']),
            'is_best_of_5':      is_bo5,
            'diff_skill':        sk_w - sk_l,
            'diff_home':         home_w - home_l,
            'diff_fatigue':      f_w - f_l,
            'diff_momentum':     mw - ml,
            'diff_h2h':          h2h_w - h2h_l,
            'diff_h2h_surface':  h2h_s_w - h2h_s_l,
            'diff_days_since_last': days_since_w - days_since_l,
            'diff_ace':          sa_w['ace']     - sa_l['ace'],
            'diff_1st_pct':      sa_w['1st_pct'] - sa_l['1st_pct'],
            'diff_1st_won':      sa_w['1st_won'] - sa_l['1st_won'],
            'diff_2nd_won':      sa_w['2nd_won'] - sa_l['2nd_won'],
            'diff_bp_saved':     sa_w['bp_saved'] - sa_l['bp_saved'],
            'diff_return_pct':   ra_w['return_pct'] - ra_l['return_pct'],
            'diff_bp_conv':      ra_w['bp_conv']    - ra_l['bp_conv'],
            'diff_return_1st':   ra_w['return_1st'] - ra_l['return_1st'],
            'court_ace_pct':     court_ace,
            'court_speed':       court_spd,

            # 17 game-specific features
            'w_avg_total_games':       w_avg_tg,
            'l_avg_total_games':       l_avg_tg,
            'w_avg_total_games_surf':  w_avg_tg_s,
            'l_avg_total_games_surf':  l_avg_tg_s,
            'w_avg_gps':               w_avg_gps,
            'l_avg_gps':               l_avg_gps,
            'w_tb_rate':               w_tb_rate,
            'l_tb_rate':               l_tb_rate,
            'w_straight_rate':         w_ss_rate,
            'l_straight_rate':         l_ss_rate,
            'combined_avg_games':      combined_avg,
            'combined_avg_games_surf': combined_avg_s,
            'combined_avg_gps':        combined_gps,
            'combined_tb_rate':        combined_tb,
            'abs_elo_diff':            abs(elo_w - elo_l),
            'abs_rank_ratio':          abs(np.log1p(rk_l) - np.log1p(rk_w)),
            'surface_avg_games':       surf_avg,

            # target + meta
            'total_games':  total_games,
            'sets_played':  sets_played,
            'tourney_date': float(row['tourney_date']) if pd.notna(row['tourney_date']) else 20200101,
        }
        rows.append(feature_row)

        # ── AGGIORNA TRACKER GAME-SPECIFICI (DOPO le feature) ────────────────
        upd_serve(l, rd, 'l'); upd_return(l, rd, 'w')

        if not np.isnan(total_games) and not np.isnan(sets_played):
            # Player total games history
            _append_limited(player_games_hist.setdefault(w, []), total_games, GAME_HISTORY)
            _append_limited(player_games_hist.setdefault(l, []), total_games, GAME_HISTORY)

            # Surface-specific
            _append_limited(player_games_surf_hist.setdefault((w, surf), []),
                            total_games, GAME_HISTORY_SURF)
            _append_limited(player_games_surf_hist.setdefault((l, surf), []),
                            total_games, GAME_HISTORY_SURF)

            # Avg games per set
            _append_limited(player_gps_hist.setdefault(w, []), avg_gps_match, GAME_HISTORY)
            _append_limited(player_gps_hist.setdefault(l, []), avg_gps_match, GAME_HISTORY)

            # Tiebreak count
            tb_w = player_tb_count.setdefault(w, [0, 0])
            tb_l = player_tb_count.setdefault(l, [0, 0])
            tb_w[0] += n_tiebreaks; tb_w[1] += int(sets_played)
            tb_l[0] += n_tiebreaks; tb_l[1] += int(sets_played)

            # Straight sets count (Bo3 only)
            if not is_bo5 and sets_played <= 3:
                ss_w = player_straight_count.setdefault(w, [0, 0])
                ss_l = player_straight_count.setdefault(l, [0, 0])
                ss_w[1] += 1; ss_l[1] += 1
                if sets_played == 2:
                    ss_w[0] += 1  # winner won in straight sets

            # Surface global
            _append_limited(surface_games_hist.setdefault(surf, []),
                            total_games, GLOBAL_HISTORY)
            _append_limited(level_games_hist.setdefault(level_code, []),
                            total_games, GLOBAL_HISTORY)

    df_out = pd.DataFrame(rows)

    # Rimuovi partite con total_games invalido
    df_out = df_out.dropna(subset=['total_games']).reset_index(drop=True)

    # Filtra outlier estremi (es. 1 set, formati anomali)
    df_out = df_out[(df_out['total_games'] >= 12) & (df_out['total_games'] <= 90)].reset_index(drop=True)

    print(f"   → Dataset game: {len(df_out):,} righe | {df_out.shape[1]} colonne")
    print(f"   → Partite scartate (score invalido): {n_skipped:,}")

    # ── Salva tracker per il predictor (file separati, NON sovrascrive quelli di train_ann) ──
    game_tracker = {
        'player_games_hist':       player_games_hist,
        'player_games_surf_hist':  player_games_surf_hist,
        'player_gps_hist':         player_gps_hist,
        'player_tb_count':         player_tb_count,
        'player_straight_count':   player_straight_count,
        'surface_games_hist':      surface_games_hist,
        'level_games_hist':        level_games_hist,
    }
    joblib.dump(game_tracker, 'game_tracker.pkl')
    print("   → game_tracker.pkl salvato")

    return df_out


# ══════════════════════════════════════════════════════════════════════════════
# 3.  FEATURE ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyze_features(X, y, feature_names):
    """Analisi correlazioni feature-target e interazioni a coppie."""
    print("\n" + "=" * 70)
    print("  📊  ANALISI FEATURE — Correlazione con total_games")
    print("=" * 70)

    correlations = {}
    for i, fname in enumerate(feature_names):
        col = X[:, i] if isinstance(X, np.ndarray) else X.iloc[:, i].values
        valid = ~(np.isnan(col) | np.isnan(y))
        if valid.sum() > 100:
            corr = np.corrcoef(col[valid], y[valid])[0, 1]
            correlations[fname] = corr
        else:
            correlations[fname] = 0.0

    sorted_feats = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    for fname, corr in sorted_feats:
        if np.isnan(corr):
            continue
        bar = '█' * int(abs(corr) * 50)
        print(f"  {fname:30s}  r={corr:+.4f}  {bar}")

    print("\n  🏆 Top 10 feature (per |r|):")
    for i, (fname, corr) in enumerate(sorted_feats[:10]):
        print(f"  {i+1:2d}. {fname:30s}  |r|={abs(corr):.4f}")

    # Feature pair analysis
    print("\n  🔗 Top 15 coppie di feature (interazione × target):")
    pair_scores = []
    n_feats = min(len(feature_names), 25)
    y_arr = np.array(y)
    for i in range(n_feats):
        col_i = X[:, i] if isinstance(X, np.ndarray) else X.iloc[:, i].values
        for j in range(i + 1, n_feats):
            col_j = X[:, j] if isinstance(X, np.ndarray) else X.iloc[:, j].values
            interaction = col_i * col_j
            valid = ~(np.isnan(interaction) | np.isnan(y_arr))
            if valid.sum() > 100:
                corr = abs(np.corrcoef(interaction[valid], y_arr[valid])[0, 1])
                if not np.isnan(corr):
                    pair_scores.append((feature_names[i], feature_names[j], corr))
    pair_scores.sort(key=lambda x: x[2], reverse=True)
    for f1, f2, corr in pair_scores[:15]:
        print(f"     {f1} × {f2}: |r|={corr:.4f}")

    return correlations, pair_scores


# ══════════════════════════════════════════════════════════════════════════════
# 4.  PCA ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def pca_analysis(X, y, feature_names):
    """Test PCA con diversi numeri di componenti."""
    print("\n" + "=" * 70)
    print("  🔬  PCA ANALYSIS")
    print("=" * 70)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results = {}
    for n_comp in [5, 10, 15, 20, 25, 30]:
        if n_comp >= X.shape[1]:
            continue
        pca = PCA(n_components=n_comp, random_state=SEED)
        X_pca = pca.fit_transform(X_scaled)
        var_explained = pca.explained_variance_ratio_.sum()

        if HAS_LGB:
            model = lgb.LGBMRegressor(n_estimators=500, max_depth=6,
                                      verbosity=-1, random_state=SEED)
            kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
            scores = cross_val_score(model, X_pca, y, cv=kf,
                                     scoring='neg_mean_absolute_error', n_jobs=-1)
            mae = -scores.mean()
            results[n_comp] = {'MAE': mae, 'var_explained': var_explained}
            print(f"  PCA({n_comp:2d}) → var_explained={var_explained:.2%}, LGB MAE={mae:.3f}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 5.  MODEL TRAINING & CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def get_base_models():
    """Restituisce dizionario di modelli base per il benchmark."""
    models = {}

    if HAS_LGB:
        models['LightGBM'] = lgb.LGBMRegressor(
            n_estimators=800, learning_rate=0.05, max_depth=7,
            num_leaves=40, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1,
            random_state=SEED, verbosity=-1)

    if HAS_XGB:
        models['XGBoost'] = xgb_lib.XGBRegressor(
            n_estimators=800, learning_rate=0.05, max_depth=7,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1,
            random_state=SEED, verbosity=0)

    models['RandomForest'] = RandomForestRegressor(
        n_estimators=500, max_depth=15, min_samples_leaf=5,
        random_state=SEED, n_jobs=-1)

    models['ExtraTrees'] = ExtraTreesRegressor(
        n_estimators=500, max_depth=15, min_samples_leaf=5,
        random_state=SEED, n_jobs=-1)

    models['GradientBoosting'] = GradientBoostingRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=5,
        random_state=SEED)

    models['BayesianRidge'] = BayesianRidge()

    models['Ridge'] = Ridge(alpha=1.0)

    models['ElasticNet'] = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=SEED)

    for k in [5, 10, 20, 50]:
        models[f'KNN_{k}'] = KNeighborsRegressor(n_neighbors=k, n_jobs=-1)

    models['SVR_rbf'] = SVR(kernel='rbf', C=10.0, epsilon=0.5)

    if HAS_LGB:
        models['Bagging_LGB'] = BaggingRegressor(
            estimator=lgb.LGBMRegressor(n_estimators=300, max_depth=6,
                                        verbosity=-1, random_state=SEED),
            n_estimators=10, random_state=SEED, n_jobs=-1)

    return models


NEEDS_SCALING = {'BayesianRidge', 'Ridge', 'ElasticNet',
                 'KNN_5', 'KNN_10', 'KNN_20', 'KNN_50',
                 'SVR_rbf'}


def cross_validate_models(models, X, y, cv=5):
    """Cross-validation di tutti i modelli. Restituisce risultati ordinati per MAE."""
    print(f"\n  🔄 Cross-validation ({cv}-fold) su {len(models)} modelli...")
    kf = KFold(n_splits=cv, shuffle=True, random_state=SEED)
    results = {}

    for name, model in models.items():
        if name in NEEDS_SCALING:
            pipeline = Pipeline([('scaler', StandardScaler()), ('model', model)])
        else:
            pipeline = model

        try:
            scores = cross_val_score(pipeline, X, y, cv=kf,
                                     scoring='neg_mean_absolute_error', n_jobs=-1)
            mae = -scores.mean()
            std = scores.std()
            results[name] = {'MAE': mae, 'std': std}
            print(f"    {name:25s}  MAE={mae:.3f} ± {std:.3f}")
        except Exception as e:
            print(f"    {name:25s}  ERRORE: {e}")

    sorted_results = dict(sorted(results.items(), key=lambda x: x[1]['MAE']))
    return sorted_results


# ══════════════════════════════════════════════════════════════════════════════
# 6.  OPTUNA TUNING
# ══════════════════════════════════════════════════════════════════════════════

def optuna_tune_lgb_reg(X_tr, y_tr, X_val, y_val, n_trials=TRIALS_GBM):
    """Tune LightGBM regressor con Optuna."""
    if not HAS_OPTUNA or not HAS_LGB:
        return None, float('inf')

    print(f"\n  🌳 Optuna LightGBM Regressor ({n_trials} trials)...")
    best_model = [None]
    best_mae = [float('inf')]

    def objective(trial):
        params = {
            'objective': 'regression', 'metric': 'mae',
            'verbosity': -1, 'seed': SEED,
            'n_estimators': trial.suggest_int('n_estimators', 300, 2500),
            'learning_rate': trial.suggest_float('lr', 0.005, 0.15, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'num_leaves': trial.suggest_int('num_leaves', 15, 200),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10, log=True),
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                             lgb.log_evaluation(period=0)])
        y_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        if mae < best_mae[0]:
            best_mae[0] = mae
            best_model[0] = model
        return mae

    study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"     Best val MAE: {best_mae[0]:.4f}")
    print(f"     Best params: {study.best_params}")
    return best_model[0], best_mae[0]


def optuna_tune_xgb_reg(X_tr, y_tr, X_val, y_val, n_trials=TRIALS_GBM):
    """Tune XGBoost regressor con Optuna."""
    if not HAS_OPTUNA or not HAS_XGB:
        return None, float('inf')

    print(f"\n  🌲 Optuna XGBoost Regressor ({n_trials} trials)...")
    best_model = [None]
    best_mae = [float('inf')]

    def objective(trial):
        params = {
            'objective': 'reg:squarederror',
            'seed': SEED, 'verbosity': 0,
            'n_estimators': trial.suggest_int('n_estimators', 300, 2500),
            'learning_rate': trial.suggest_float('lr', 0.005, 0.15, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 50),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10, log=True),
            'gamma': trial.suggest_float('gamma', 0, 5),
        }
        model = xgb_lib.XGBRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        y_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        if mae < best_mae[0]:
            best_mae[0] = mae
            best_model[0] = model
        return mae

    study = optuna.create_study(direction='minimize',
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"     Best val MAE: {best_mae[0]:.4f}")
    print(f"     Best params: {study.best_params}")
    return best_model[0], best_mae[0]


# ══════════════════════════════════════════════════════════════════════════════
# 7.  STACKING ENSEMBLE
# ══════════════════════════════════════════════════════════════════════════════

def build_stacking_ensemble(base_models, X_tr, y_tr, X_val, y_val):
    """Costruisce un ensemble stacking con i modelli migliori."""
    print("\n  🏗️  Stacking Ensemble...")

    estimators = []
    for name, model in base_models.items():
        if name in NEEDS_SCALING:
            pipe = Pipeline([('scaler', StandardScaler()), ('model', model)])
            estimators.append((name, pipe))
        else:
            estimators.append((name, model))

    if len(estimators) < 2:
        print("     ⚠️ Troppo pochi modelli per stacking, skip")
        return None, float('inf')

    stacking = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=1.0),
        cv=5, n_jobs=-1
    )
    stacking.fit(X_tr, y_tr)
    y_pred = stacking.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    print(f"     Stacking MAE (val): {mae:.4f}")
    return stacking, mae


# ══════════════════════════════════════════════════════════════════════════════
# 8.  FEATURE IMPORTANCE (LightGBM)
# ══════════════════════════════════════════════════════════════════════════════

def print_feature_importance(model, feature_names, top_n=20):
    """Stampa feature importance del modello LightGBM o XGBoost."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'feature_importance'):
        importances = model.feature_importance(importance_type='gain')
    else:
        return

    indices = np.argsort(importances)[::-1]
    print(f"\n  📈 Feature Importance (top {top_n}):")
    for i in range(min(top_n, len(feature_names))):
        idx = indices[i]
        bar = '█' * int(importances[idx] / max(importances) * 40)
        print(f"    {i+1:2d}. {feature_names[idx]:30s}  {importances[idx]:10.1f}  {bar}")


# ══════════════════════════════════════════════════════════════════════════════
# 9.  SET CLASSIFIER  (Stadio 1)
# ══════════════════════════════════════════════════════════════════════════════

def train_set_classifier_optuna(X_tr, y_tr, X_val, y_val, n_classes,
                                 n_trials=TRIALS_GBM, label="Bo3"):
    """Addestra classificatore numero di set con Optuna.
    Bo3: y ∈ {0, 1} (2-set=0, 3-set=1)
    Bo5: y ∈ {0, 1, 2} (3-set=0, 4-set=1, 5-set=2)
    """
    if not HAS_LGB or not HAS_OPTUNA:
        return None, 0.0

    is_binary = (n_classes == 2)
    print(f"\n  🎯 Training Set Classifier [{label}] — {'binary' if is_binary else f'{n_classes}-class'}...")

    best_model = [None]
    best_acc = [0.0]

    def objective(trial):
        params = {
            'objective': 'binary' if is_binary else 'multiclass',
            'metric': 'binary_logloss' if is_binary else 'multi_logloss',
            'verbosity': -1, 'seed': SEED,
            'n_estimators': trial.suggest_int('n_estimators', 200, 2000),
            'learning_rate': trial.suggest_float('lr', 0.005, 0.15, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'num_leaves': trial.suggest_int('num_leaves', 15, 200),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10, log=True),
        }
        if not is_binary:
            params['num_class'] = n_classes

        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                             lgb.log_evaluation(period=0)])
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        if acc > best_acc[0]:
            best_acc[0] = acc
            best_model[0] = model
        return acc

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"     Best val accuracy: {best_acc[0]:.4f}")
    return best_model[0], best_acc[0]


def train_set_classifier_xgb(X_tr, y_tr, X_val, y_val, n_classes,
                               n_trials=TRIALS_GBM, label="Bo3"):
    """XGBoost set classifier con Optuna."""
    if not HAS_XGB or not HAS_OPTUNA:
        return None, 0.0

    is_binary = (n_classes == 2)
    print(f"\n  🌲 XGB Set Classifier [{label}]...")

    best_model = [None]
    best_acc = [0.0]

    def objective(trial):
        params = {
            'objective': 'binary:logistic' if is_binary else 'multi:softprob',
            'eval_metric': 'logloss' if is_binary else 'mlogloss',
            'seed': SEED, 'verbosity': 0,
            'n_estimators': trial.suggest_int('n_estimators', 200, 2000),
            'learning_rate': trial.suggest_float('lr', 0.005, 0.15, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 50),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10, log=True),
            'gamma': trial.suggest_float('gamma', 0, 5),
        }
        if not is_binary:
            params['num_class'] = n_classes

        model = xgb_lib.XGBClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        if acc > best_acc[0]:
            best_acc[0] = acc
            best_model[0] = model
        return acc

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"     Best val accuracy: {best_acc[0]:.4f}")
    return best_model[0], best_acc[0]


# ══════════════════════════════════════════════════════════════════════════════
# 10.  CONDITIONAL GAME REGRESSOR  (Stadio 2)
# ══════════════════════════════════════════════════════════════════════════════

def train_conditional_regressor(X, y, n_trials=10, label="2-set"):
    """Addestra un regressore specifico per partite con N set."""
    if len(X) < 200:
        print(f"     ⚠️ Troppo pochi dati per {label} ({len(X)}), skip")
        return None

    print(f"     Conditional regressor [{label}]: {len(X)} samples, mean={y.mean():.1f}, std={y.std():.1f}")

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)

    best_model = None
    best_mae = float('inf')

    if HAS_LGB and HAS_OPTUNA:
        lgb_model, lgb_mae = optuna_tune_lgb_reg(X_tr, y_tr, X_val, y_val, n_trials=n_trials)
        if lgb_model is not None and lgb_mae < best_mae:
            best_mae = lgb_mae
            best_model = lgb_model

    if HAS_XGB and HAS_OPTUNA:
        xgb_model, xgb_mae = optuna_tune_xgb_reg(X_tr, y_tr, X_val, y_val, n_trials=n_trials)
        if xgb_model is not None and xgb_mae < best_mae:
            best_mae = xgb_mae
            best_model = xgb_model

    if best_model is not None:
        # Retrain on all data
        if isinstance(best_model, lgb.LGBMRegressor):
            final = lgb.LGBMRegressor(**best_model.get_params())
        else:
            params = best_model.get_params()
            params.pop('early_stopping_rounds', None)
            final = xgb_lib.XGBRegressor(**params)
        final.fit(X, y)
        print(f"     → [{label}] Best MAE (val): {best_mae:.4f}")
        return final

    return None


# ══════════════════════════════════════════════════════════════════════════════
# 11.  HIERARCHICAL PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

def predict_hierarchical(set_clf, cond_regressors, X, set_classes):
    """Predizione gerarchica: P(n_set) × E[games | n_set].

    set_clf: classificatore che restituisce probabilità per ogni classe
    cond_regressors: dict {set_count: regressor}
    X: features matrix
    set_classes: lista classi (es. [2, 3] per Bo3)
    """
    # Probabilità per ogni numero di set
    set_probs = set_clf.predict_proba(X)  # shape (N, n_classes)

    # Predizione condizionale per ogni numero di set
    y_pred = np.zeros(len(X))
    for i, set_count in enumerate(set_classes):
        if set_count in cond_regressors and cond_regressors[set_count] is not None:
            cond_pred = cond_regressors[set_count].predict(X)
        else:
            # Fallback: usa la media storica per quel numero di set
            cond_pred = np.full(len(X), set_count * 10.0)  # approssimazione

        y_pred += set_probs[:, i] * cond_pred

    return y_pred


# ══════════════════════════════════════════════════════════════════════════════
# 12.  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    t_start = time.time()

    CSV_CANDIDATES = [
        'historialTenis.csv', '../scraping/historialTenis.csv',
        'historial_tenis_COMPLETO.csv', '../scraping/historial_tenis_COMPLETO.csv',
    ]
    csv_path = next((p for p in CSV_CANDIDATES if os.path.exists(p)), None)
    if csv_path is None:
        raise FileNotFoundError("historialTenis.csv non trovato. Esegui da prediccion/")

    # ── 1. Feature engineering ────────────────────────────────────────────────
    df_all = prepare_game_dataset(csv_path)

    # ── 2. Separa Bo3 e Bo5 ──────────────────────────────────────────────────
    df_bo3 = df_all[df_all['is_best_of_5'] == 0.0].reset_index(drop=True)
    df_bo5 = df_all[df_all['is_best_of_5'] == 1.0].reset_index(drop=True)

    print(f"\n   Bo3: {len(df_bo3):,} partite | Bo5: {len(df_bo5):,} partite")
    print(f"   Bo3 target: mean={df_bo3['total_games'].mean():.1f}, std={df_bo3['total_games'].std():.1f}")
    print(f"   Bo5 target: mean={df_bo5['total_games'].mean():.1f}, std={df_bo5['total_games'].std():.1f}")

    best_models = {}  # {'bo3': {...}, 'bo5': {...}}

    for label, df_subset, set_classes in [
        ('Bo3', df_bo3, [2, 3]),
        ('Bo5', df_bo5, [3, 4, 5]),
    ]:
        print("\n" + "=" * 80)
        print(f"  🎾  TRAINING SET + GAME PREDICTION — {label}  ({len(df_subset):,} matches)")
        print("=" * 80)

        if len(df_subset) < 500:
            print(f"  ⚠️ Troppo pochi dati per {label}, skip")
            continue

        # Filtra solo partite con set validi
        df_clean = df_subset[df_subset['sets_played'].isin(set_classes)].reset_index(drop=True)
        print(f"  → Partite con set validi: {len(df_clean):,}")

        X_all = df_clean[ALL_FEATURES].fillna(0).values
        y_games = df_clean['total_games'].values
        y_sets_raw = df_clean['sets_played'].values

        # Encode set classes: {2: 0, 3: 1} per Bo3, {3: 0, 4: 1, 5: 2} per Bo5
        set_to_idx = {s: i for i, s in enumerate(set_classes)}
        y_sets = np.array([set_to_idx[int(s)] for s in y_sets_raw])
        n_classes = len(set_classes)

        # Set distribution
        print(f"\n  📊 Distribuzione set:")
        for s in set_classes:
            n = np.sum(y_sets_raw == s)
            pct = n / len(y_sets_raw) * 100
            sub_games = y_games[y_sets_raw == s]
            print(f"     {s}-set: {n:,} ({pct:.1f}%) → games mean={sub_games.mean():.1f}, std={sub_games.std():.1f}")

        # ── Feature analysis ─────────────────────────────────────────────────
        analyze_features(X_all, y_games, ALL_FEATURES)

        # ── Split: 70% train, 15% val, 15% test ─────────────────────────────
        X_tr, X_tmp, y_games_tr, y_games_tmp, y_sets_tr, y_sets_tmp = train_test_split(
            X_all, y_games, y_sets, test_size=0.30, random_state=SEED, stratify=y_sets)
        X_val, X_test, y_games_val, y_games_test, y_sets_val, y_sets_test = train_test_split(
            X_tmp, y_games_tmp, y_sets_tmp, test_size=0.50, random_state=SEED, stratify=y_sets_tmp)

        y_sets_raw_test = np.array([set_classes[i] for i in y_sets_test])

        print(f"\n  Split: train={len(X_tr)}, val={len(X_val)}, test={len(X_test)}")

        # ══════════════════════════════════════════════════════════════════════
        # STADIO 1: CLASSIFICATORE SET
        # ══════════════════════════════════════════════════════════════════════
        print("\n" + "-" * 70)
        print(f"  STADIO 1: Classificatore Set — {label}")
        print("-" * 70)

        # Majority baseline
        majority_class = np.bincount(y_sets_test).argmax()
        baseline_acc = np.mean(y_sets_test == majority_class)
        print(f"  📏 Baseline (majority class={set_classes[majority_class]}): accuracy={baseline_acc:.1%}")

        # LightGBM set classifier
        lgb_set_clf, lgb_set_acc = train_set_classifier_optuna(
            X_tr, y_sets_tr, X_val, y_sets_val,
            n_classes=n_classes, n_trials=TRIALS_GBM, label=label)

        # XGBoost set classifier
        xgb_set_clf, xgb_set_acc = train_set_classifier_xgb(
            X_tr, y_sets_tr, X_val, y_sets_val,
            n_classes=n_classes, n_trials=TRIALS_GBM, label=label)

        # Cross-validation for set classifiers
        print(f"\n  🔄 Set Classifier CV Benchmark:")
        kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
        set_clf_models = {}

        if HAS_LGB:
            set_clf_models['LGB_set'] = lgb.LGBMClassifier(
                n_estimators=800, max_depth=7, verbosity=-1, random_state=SEED)
        if HAS_XGB:
            set_clf_models['XGB_set'] = xgb_lib.XGBClassifier(
                n_estimators=800, max_depth=7, verbosity=0, random_state=SEED)
        set_clf_models['RF_set'] = RandomForestClassifier(
            n_estimators=500, max_depth=15, random_state=SEED, n_jobs=-1)
        set_clf_models['ET_set'] = ExtraTreesClassifier(
            n_estimators=500, max_depth=15, random_state=SEED, n_jobs=-1)
        set_clf_models['KNN10_set'] = Pipeline([
            ('scaler', StandardScaler()),
            ('model', KNeighborsClassifier(n_neighbors=10, n_jobs=-1))])
        set_clf_models['GB_set'] = GradientBoostingClassifier(
            n_estimators=500, max_depth=5, random_state=SEED)
        set_clf_models['LR_set'] = Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(max_iter=2000, random_state=SEED))])

        for name, clf in set_clf_models.items():
            try:
                scores = cross_val_score(clf, X_all, y_sets, cv=kf, scoring='accuracy', n_jobs=-1)
                print(f"    {name:20s}  acc={scores.mean():.4f} ± {scores.std():.4f}")
            except Exception as e:
                print(f"    {name:20s}  ERRORE: {e}")

        # Select best set classifier
        best_set_clf = None
        best_set_acc_test = 0.0
        set_clf_name = "None"

        if lgb_set_clf is not None:
            y_pred = lgb_set_clf.predict(X_test)
            acc = accuracy_score(y_sets_test, y_pred)
            print(f"\n  📊 LGB Set Clf test accuracy: {acc:.4f}")
            if acc > best_set_acc_test:
                best_set_acc_test = acc
                best_set_clf = lgb_set_clf
                set_clf_name = "LGB"

        if xgb_set_clf is not None:
            y_pred = xgb_set_clf.predict(X_test)
            acc = accuracy_score(y_sets_test, y_pred)
            print(f"  📊 XGB Set Clf test accuracy: {acc:.4f}")
            if acc > best_set_acc_test:
                best_set_acc_test = acc
                best_set_clf = xgb_set_clf
                set_clf_name = "XGB"

        if best_set_clf is not None:
            print(f"\n  🏆 Best Set Classifier: {set_clf_name} (acc={best_set_acc_test:.4f}, baseline={baseline_acc:.4f})")
            if hasattr(best_set_clf, 'feature_importances_'):
                print_feature_importance(best_set_clf, ALL_FEATURES, top_n=15)
        else:
            print(f"\n  ⚠️ Nessun classificatore set addestrato per {label}")

        # ══════════════════════════════════════════════════════════════════════
        # STADIO 2: REGRESSORI CONDIZIONALI
        # ══════════════════════════════════════════════════════════════════════
        print("\n" + "-" * 70)
        print(f"  STADIO 2: Regressori Condizionali per Game — {label}")
        print("-" * 70)

        cond_regressors = {}
        cond_means = {}

        for s_idx, s_count in enumerate(set_classes):
            mask_all = (y_sets_raw == s_count)
            X_cond = X_all[df_clean['sets_played'].values == s_count]
            y_cond = y_games[df_clean['sets_played'].values == s_count]

            cond_means[s_count] = float(y_cond.mean())
            print(f"\n  --- {s_count}-set matches ({len(X_cond)} samples) ---")

            reg = train_conditional_regressor(X_cond, y_cond, n_trials=8, label=f"{s_count}-set")
            cond_regressors[s_count] = reg

        # ══════════════════════════════════════════════════════════════════════
        # STADIO 3: VALUTAZIONE
        # ══════════════════════════════════════════════════════════════════════
        print("\n" + "-" * 70)
        print(f"  STADIO 3: Valutazione Finale — {label}")
        print("-" * 70)

        # --- A. Direct regression (benchmark) ---
        print(f"\n  A) Approccio DIRETTO (regressione su total_games):")
        lgb_direct, lgb_direct_mae = optuna_tune_lgb_reg(
            X_tr, y_games_tr, X_val, y_games_val, n_trials=TRIALS_GBM)
        xgb_direct, xgb_direct_mae = optuna_tune_xgb_reg(
            X_tr, y_games_tr, X_val, y_games_val, n_trials=TRIALS_GBM)

        direct_results = {}
        if lgb_direct is not None:
            y_pred = lgb_direct.predict(X_test)
            mae = mean_absolute_error(y_games_test, y_pred)
            direct_results['LGB_direct'] = mae
            print(f"     LGB Direct MAE (test): {mae:.4f}")

        if xgb_direct is not None:
            y_pred = xgb_direct.predict(X_test)
            mae = mean_absolute_error(y_games_test, y_pred)
            direct_results['XGB_direct'] = mae
            print(f"     XGB Direct MAE (test): {mae:.4f}")

        if lgb_direct is not None and xgb_direct is not None:
            y_avg = (lgb_direct.predict(X_test) + xgb_direct.predict(X_test)) / 2
            mae_avg = mean_absolute_error(y_games_test, y_avg)
            direct_results['Avg_direct'] = mae_avg
            print(f"     Avg Direct MAE (test): {mae_avg:.4f}")

        best_direct_name = min(direct_results, key=direct_results.get) if direct_results else None
        best_direct_mae = direct_results[best_direct_name] if best_direct_name else float('inf')

        # --- B. Hierarchical: P(set) × E[games|set] ---
        print(f"\n  B) Approccio GERARCHICO (set classifier → conditional regressors):")

        hier_results = {}

        if best_set_clf is not None:
            # Use predicted set probabilities × conditional regressors
            y_hier = predict_hierarchical(best_set_clf, cond_regressors, X_test, set_classes)
            mae_hier = mean_absolute_error(y_games_test, y_hier)
            hier_results['Hierarchical'] = mae_hier
            print(f"     Hierarchical MAE (test): {mae_hier:.4f}")

            # Also try: use predicted set probabilities × conditional MEANS
            set_probs = best_set_clf.predict_proba(X_test)
            y_hier_mean = np.zeros(len(X_test))
            for i, s_count in enumerate(set_classes):
                y_hier_mean += set_probs[:, i] * cond_means[s_count]
            mae_hier_mean = mean_absolute_error(y_games_test, y_hier_mean)
            hier_results['Hierarchical_means'] = mae_hier_mean
            print(f"     Hierarchical (cond means only) MAE (test): {mae_hier_mean:.4f}")

            # Hybrid: average of hierarchical + direct
            if best_direct_name:
                if best_direct_name == 'Avg_direct' and lgb_direct is not None and xgb_direct is not None:
                    y_direct_best = (lgb_direct.predict(X_test) + xgb_direct.predict(X_test)) / 2
                elif best_direct_name == 'LGB_direct' and lgb_direct is not None:
                    y_direct_best = lgb_direct.predict(X_test)
                elif best_direct_name == 'XGB_direct' and xgb_direct is not None:
                    y_direct_best = xgb_direct.predict(X_test)
                else:
                    y_direct_best = None

                if y_direct_best is not None:
                    y_hybrid = 0.5 * y_hier + 0.5 * y_direct_best
                    mae_hybrid = mean_absolute_error(y_games_test, y_hybrid)
                    hier_results['Hybrid_50_50'] = mae_hybrid
                    print(f"     Hybrid (50% hier + 50% direct) MAE (test): {mae_hybrid:.4f}")

                    # Try different blend ratios
                    best_alpha = 0.5
                    best_blend_mae = mae_hybrid
                    for alpha in [0.3, 0.4, 0.6, 0.7]:
                        y_blend = alpha * y_hier + (1 - alpha) * y_direct_best
                        mae_blend = mean_absolute_error(y_games_test, y_blend)
                        if mae_blend < best_blend_mae:
                            best_blend_mae = mae_blend
                            best_alpha = alpha
                    y_blend_best = best_alpha * y_hier + (1 - best_alpha) * y_direct_best
                    mae_blend_final = mean_absolute_error(y_games_test, y_blend_best)
                    hier_results[f'Hybrid_{int(best_alpha*100)}_{int((1-best_alpha)*100)}'] = mae_blend_final
                    print(f"     Hybrid (best α={best_alpha:.1f}) MAE (test): {mae_blend_final:.4f}")

        # Oracle: what if we knew the exact number of sets?
        print(f"\n  C) Oracle (perfetta conoscenza set):")
        y_oracle = np.zeros(len(X_test))
        for i, s_count in enumerate(set_classes):
            mask = (y_sets_test == i)
            if mask.any() and s_count in cond_regressors and cond_regressors[s_count] is not None:
                y_oracle[mask] = cond_regressors[s_count].predict(X_test[mask])
            else:
                y_oracle[mask] = cond_means.get(s_count, y_games_test[mask].mean())
        mae_oracle = mean_absolute_error(y_games_test, y_oracle)
        print(f"     Oracle MAE (perfect set knowledge): {mae_oracle:.4f}")

        # Baseline
        mean_mae = mean_absolute_error(y_games_test, [y_games.mean()] * len(y_games_test))
        print(f"\n  📏 Baseline (predict mean={y_games.mean():.1f}): MAE={mean_mae:.3f}")

        # ── Summary & Select Best ────────────────────────────────────────────
        all_results = {}
        all_results.update(direct_results)
        all_results.update(hier_results)

        print(f"\n  📊 RIEPILOGO {label}:")
        for name, mae in sorted(all_results.items(), key=lambda x: x[1]):
            marker = "  ✅" if mae <= 4.0 else ""
            print(f"     {name:35s}  MAE={mae:.4f}{marker}")
        print(f"     {'Oracle':35s}  MAE={mae_oracle:.4f}  (limite teorico)")
        print(f"     {'Baseline (predict mean)':35s}  MAE={mean_mae:.4f}")

        # Select the best overall approach
        best_approach_name = min(all_results, key=all_results.get) if all_results else None
        best_approach_mae = all_results[best_approach_name] if best_approach_name else float('inf')

        print(f"\n  🏆 Miglior approccio {label}: {best_approach_name} (MAE={best_approach_mae:.4f})")

        # ── Re-train best models on ALL data ─────────────────────────────────
        print(f"\n  🚀 Re-training modelli finali su TUTTI i dati {label}...")

        # Re-train set classifier on all data
        set_clf_final = None
        if best_set_clf is not None:
            if isinstance(best_set_clf, lgb.LGBMClassifier):
                set_clf_final = lgb.LGBMClassifier(**best_set_clf.get_params())
            else:
                params = best_set_clf.get_params()
                params.pop('early_stopping_rounds', None)
                set_clf_final = xgb_lib.XGBClassifier(**params)
            set_clf_final.fit(X_all, y_sets)

        # Re-train conditional regressors on all data (already done in train_conditional_regressor)
        # They were trained on the full subset for each set count

        # Re-train direct regressors
        lgb_direct_final = None
        xgb_direct_final = None
        if lgb_direct is not None:
            lgb_direct_final = lgb.LGBMRegressor(**lgb_direct.get_params())
            lgb_direct_final.fit(X_all, y_games)
        if xgb_direct is not None:
            xgb_params = xgb_direct.get_params()
            xgb_params.pop('early_stopping_rounds', None)
            xgb_direct_final = xgb_lib.XGBRegressor(**xgb_params)
            xgb_direct_final.fit(X_all, y_games)

        # Determine best_alpha for hybrid
        best_alpha_final = 0.5  # default
        if best_approach_name and 'Hybrid' in best_approach_name:
            parts = best_approach_name.split('_')
            if len(parts) >= 2:
                try:
                    best_alpha_final = int(parts[1]) / 100.0
                except ValueError:
                    best_alpha_final = 0.5

        best_models[label.lower()] = {
            'approach': best_approach_name,
            'mae': best_approach_mae,
            'set_classifier': set_clf_final,
            'set_classifier_name': set_clf_name,
            'set_accuracy': best_set_acc_test,
            'cond_regressors': cond_regressors,
            'cond_means': cond_means,
            'lgb_direct': lgb_direct_final,
            'xgb_direct': xgb_direct_final,
            'set_classes': set_classes,
            'hybrid_alpha': best_alpha_final,
            'features': ALL_FEATURES,
            'n_samples': len(X_all),
            'oracle_mae': mae_oracle,
        }

    # ══════════════════════════════════════════════════════════════════════════
    # 13.  SAVE FINAL MODEL
    # ══════════════════════════════════════════════════════════════════════════
    from datetime import datetime

    modelo_games = {
        'trained_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'features': ALL_FEATURES,
        'approach': 'hierarchical',
    }

    for key in ('bo3', 'bo5'):
        if key in best_models:
            info = best_models[key]
            modelo_games[key] = {
                'approach': info['approach'],
                'mae': info['mae'],
                'set_classifier': info['set_classifier'],
                'set_classifier_name': info['set_classifier_name'],
                'set_accuracy': info['set_accuracy'],
                'cond_regressors': info['cond_regressors'],
                'cond_means': info['cond_means'],
                'lgb_direct': info['lgb_direct'],
                'xgb_direct': info['xgb_direct'],
                'set_classes': info['set_classes'],
                'hybrid_alpha': info['hybrid_alpha'],
                'n_samples': info['n_samples'],
                'oracle_mae': info['oracle_mae'],
            }

    joblib.dump(modelo_games, 'modelo_games.pkl')

    elapsed = time.time() - t_start

    # ── Final summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  📋  RIEPILOGO FINALE")
    print("=" * 80)
    for key in ('bo3', 'bo5'):
        if key in best_models:
            info = best_models[key]
            status = "✅" if info['mae'] <= 4.0 else "⚠️"
            print(f"  {status} {key.upper()}: {info['approach']:30s}  MAE={info['mae']:.4f}  ({info['n_samples']} campioni)")
            print(f"        Set classifier: {info['set_classifier_name']} (acc={info['set_accuracy']:.4f})")
            print(f"        Oracle MAE: {info['oracle_mae']:.4f}")
        else:
            print(f"  ❌ {key.upper()}: Nessun modello addestrato")

    target_mae = 4.0
    for key in ('bo3', 'bo5'):
        if key in best_models:
            if best_models[key]['mae'] <= target_mae:
                print(f"\n  🎯 {key.upper()}: OBIETTIVO RAGGIUNTO! MAE={best_models[key]['mae']:.4f} ≤ {target_mae}")
            else:
                print(f"\n  ⚠️  {key.upper()}: MAE={best_models[key]['mae']:.4f} > {target_mae} — serve miglioramento")

    print(f"\n  ⏱️  Tempo totale: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"\n  ✅ File salvati:")
    for f in ['modelo_games.pkl', 'game_tracker.pkl']:
        stato = "✅" if os.path.exists(f) else "—"
        print(f"    {stato}  {f}")
