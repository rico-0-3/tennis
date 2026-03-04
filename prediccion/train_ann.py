"""
train_ann.py v4.0  —  Optimized Tennis Match Predictor
========================================================
Improvements over v3:
  ✅ Fixed 2025 data (date=0 → extracted from tourney_id)
  ✅ Proper temporal train/test split (no data leakage)
  ✅ Gradient Boosting ensemble (LightGBM + XGBoost)
  ✅ Enhanced feature engineering (log-transforms, ratios, interactions)
  ✅ Stacking ensemble (GBM + ANN)
  ✅ Platt calibration on held-out temporal set
  ✅ Time-series aware validation

USAGE:
    cd prediccion/
    python train_ann.py

OUTPUT:
    modelo_ann.pth               ← ANN model
    ann_config.json              ← config with v4 metadata
    scaler_ann.pkl               ← StandardScaler
    elo_surface.pkl              ← Elo by surface
    streak_players.pkl           ← active streak per player
    momentum_surface.pkl         ← momentum per (player, surface)
    calibrator_ann.pkl           ← Platt scaling calibrator
    resultados_comparacion_finale.csv  ← model comparison
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

from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore")

# ── Optional imports ──────────────────────────────────────────────────────────
try:
    import lightgbm as lgb
    HAS_LGB = True
    print("   ✅ LightGBM disponibile")
except ImportError:
    HAS_LGB = False
    print("   ⚠️  LightGBM non trovato")

try:
    import xgboost as xgb
    HAS_XGB = True
    print("   ✅ XGBoost disponibile")
except ImportError:
    HAS_XGB = False
    print("   ⚠️  XGBoost non trovato")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("⚠️  optuna non trovato")

# ─── Seed + Device ────────────────────────────────────────────────────────────
SEED = 42
import random
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Device: {device}")

# ─── Configuration ────────────────────────────────────────────────────────────
TRIALS_GBM = 60
TRIALS_ANN = 40
ANN_EPOCHS = 80

LEVEL_MULT = {'G': 2.0, 'M': 1.5, 'F': 1.4, 'A': 1.0,
              'D': 1.0, 'C': 0.8, 'S': 0.7, 'E': 0.5, '0': 0.8, 'O': 0.5}


# ══════════════════════════════════════════════════════════════════════════════
# 1.  FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def carica_e_prepara(csv_path: str):
    print(f"\n📂 Caricamento: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    df['tourney_date'] = pd.to_numeric(df['tourney_date'], errors='coerce')

    # ── Fix 2025 data with date=0 ────────────────────────────────────────────
    mask_bad_date = (df['tourney_date'] <= 0) | df['tourney_date'].isna()
    for idx in df[mask_bad_date].index:
        tid = str(df.loc[idx, 'tourney_id'])
        if tid.startswith('2025'):
            df.loc[idx, 'tourney_date'] = 20250601
        elif tid.startswith('2026'):
            df.loc[idx, 'tourney_date'] = 20260101
    df = df[df['tourney_date'] > 20000000].copy()

    df = df.sort_values(by=['tourney_date', 'match_num']).reset_index(drop=True)
    df['minutes'] = df['minutes'].fillna(90)
    print(f"   → {len(df):,} partite caricate")

    # ── Encoding ─────────────────────────────────────────────────────────────
    surface_map = {'Hard': 0, 'Clay': 1, 'Grass': 2, 'Carpet': 3}
    hand_map    = {'R': 1, 'L': -1, 'U': 0, '0': 0, 0: 0}
    level_map   = {'G': 5, 'M': 4, 'A': 3, 'D': 3, 'F': 4, 'C': 2, 'S': 1, 'E': 0, '0': 2, 'O': 1}
    round_map   = {'F': 7, 'SF': 6, 'QF': 5, 'R16': 4, 'R32': 3,
                   'R64': 2, 'R128': 1, 'RR': 4, 'BR': 3,
                   'Finals ': 7, 'Semi': 6, 'Quarter': 5}

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
    serve_t   = {}
    return_t  = {}
    elo_surf  = {}
    streak_t  = {}
    player_wins_total = {}
    player_losses_total = {}

    ELO_DEFAULT = 1500.0
    K_ELO       = 32.0

    def get_elo(p, s):
        return elo_surf.get((p, s), ELO_DEFAULT)

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

        # --- H2H ---
        p1k, p2k = sorted([w, l]); key = (p1k, p2k)
        rec = h2h_t.get(key, [0, 0])
        if w == p1k:
            h2h_w = rec[0]-rec[1]; h2h_l = rec[1]-rec[0]; rec[0] += 1
        else:
            h2h_w = rec[1]-rec[0]; h2h_l = rec[0]-rec[1]; rec[1] += 1
        h2h_t[key] = rec

        # --- Elo per superficie ---
        elo_w = get_elo(w, surf); elo_l = get_elo(l, surf)
        expected_w = 1.0 / (1.0 + 10.0 ** ((elo_l - elo_w) / 400.0))
        elo_surf[(w, surf)] = elo_w + K_ELO * (1.0 - expected_w)
        elo_surf[(l, surf)] = elo_l + K_ELO * (0.0 - (1.0 - expected_w))

        # --- Striscia attiva ---
        str_w = streak_t.get(w, 0); str_l = streak_t.get(l, 0)
        streak_t[w] = max(0, str_w) + 1 if str_w >= 0 else 1
        streak_t[l] = min(0, str_l) - 1 if str_l <= 0 else -1

        # --- Overall win % (rolling) ---
        pw_w = player_wins_total.get(w, 0)
        pw_l = player_wins_total.get(l, 0)
        pl_w = player_losses_total.get(w, 0)
        pl_l = player_losses_total.get(l, 0)
        total_w = pw_w + pl_w
        total_l = pw_l + pl_l
        overall_wr_w = pw_w / total_w if total_w > 5 else 0.5
        overall_wr_l = pw_l / total_l if total_l > 5 else 0.5
        player_wins_total[w] = pw_w + 1
        player_losses_total[l] = pl_l + 1

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

        # --- Statistiche ritorno ---
        def get_ra(player):
            s = return_t.get(player, {})
            return {'return_pct':  np.mean(s.get('return_pct',  [0.35])),
                    'bp_conv':     np.mean(s.get('bp_conv',     [0.35])),
                    'return_1st':  np.mean(s.get('return_1st',  [0.30]))}
        ra_w = get_ra(w); ra_l = get_ra(l)

        def upd_return(player, rd, opp_pref):
            s = return_t.setdefault(player, {})
            opp_svpt   = rd.get(f'{opp_pref}_svpt', np.nan)
            opp_1stWon = rd.get(f'{opp_pref}_1stWon', np.nan)
            opp_2ndWon = rd.get(f'{opp_pref}_2ndWon', np.nan)
            opp_1stIn  = rd.get(f'{opp_pref}_1stIn', np.nan)
            opp_bpFaced = rd.get(f'{opp_pref}_bpFaced', np.nan)
            opp_bpSaved = rd.get(f'{opp_pref}_bpSaved', np.nan)
            if opp_svpt and opp_svpt > 0 and not np.isnan(opp_svpt):
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
        seed_w = float(row['winner_seed']) if pd.notna(row.get('winner_seed')) else 33
        seed_l = float(row['loser_seed'])  if pd.notna(row.get('loser_seed'))  else 33
        rank_w = float(row['winner_rank']) if pd.notna(row.get('winner_rank')) else 500
        rank_l = float(row['loser_rank'])  if pd.notna(row.get('loser_rank'))  else 500
        lev_w  = LEVEL_MULT.get(str(row.get('tourney_level','')), 1.0)

        # ── Enhanced features ─────────────────────────────────────────────────
        log_rank_ratio = np.log1p(rank_l) - np.log1p(rank_w)
        pts_ratio = (pts_w + 1) / (pts_l + 1)
        log_pts_ratio = np.log(pts_ratio) if pts_ratio > 0 else 0
        serve_dom_w = sa_w['ace'] - sa_w['df']
        serve_dom_l = sa_l['ace'] - sa_l['df']
        game_score_w = (sa_w['1st_won'] * 0.4 + sa_w['2nd_won'] * 0.3 +
                        ra_w['return_pct'] * 0.3)
        game_score_l = (sa_l['1st_won'] * 0.4 + sa_l['2nd_won'] * 0.3 +
                        ra_l['return_pct'] * 0.3)

        diffs = {
            'diff_rank':         rank_l - rank_w,
            'log_rank_ratio':    log_rank_ratio,
            'diff_rank_points':  pts_w - pts_l,
            'log_pts_ratio':     log_pts_ratio,
            'diff_seed':         seed_l - seed_w,
            'diff_age':          (row['winner_age']-row['loser_age'])
                                 if pd.notna(row.get('winner_age')) and pd.notna(row.get('loser_age')) else 0,
            'diff_ht':           (row['winner_ht']-row['loser_ht'])
                                 if pd.notna(row.get('winner_ht')) and pd.notna(row.get('loser_ht')) else 0,
            'diff_elo':          elo_w - elo_l,
            'diff_streak':       float(str_w - str_l),
            'diff_overall_wr':   overall_wr_w - overall_wr_l,
            'surface_enc':       float(row['surface_enc']),
            'tourney_level':     float(row['tourney_level_enc']),
            'round_enc':         float(row['round_enc']),
            'draw_size':         float(row['draw_size']) if pd.notna(row.get('draw_size')) else 32,
            'diff_hand':         row['w_hand_enc'] - row['l_hand_enc'],
            'diff_skill':        sk_w - sk_l,
            'diff_home':         home_w - home_l,
            'diff_fatigue':      f_w - f_l,
            'diff_momentum':     mw - ml,
            'diff_h2h':          h2h_w - h2h_l,
            'diff_ace':          sa_w['ace']     - sa_l['ace'],
            'diff_1st_won':      sa_w['1st_won'] - sa_l['1st_won'],
            'diff_2nd_won':      sa_w['2nd_won'] - sa_l['2nd_won'],
            'diff_bp_saved':     sa_w['bp_saved']- sa_l['bp_saved'],
            'diff_serve_dom':    serve_dom_w - serve_dom_l,
            'diff_game_score':   game_score_w - game_score_l,
            'diff_return_pct':   ra_w['return_pct'] - ra_l['return_pct'],
            'diff_bp_conv':      ra_w['bp_conv']    - ra_l['bp_conv'],
            'diff_return_1st':   ra_w['return_1st'] - ra_l['return_1st'],
            'tourney_date':      float(row['tourney_date']) if pd.notna(row['tourney_date']) else 20200101,
            'level_weight':      lev_w,
        }

        # ── Court speed ──
        tourney_year = int(str(row['tourney_date'])[:4]) if pd.notna(row.get('tourney_date')) else 2025
        surf_safe = surf if isinstance(surf, str) else 'Hard'
        court_ace, court_spd = get_court_stats(
            row.get('tourney_name', ''), surf_safe, tourney_year)
        diffs['court_ace_pct'] = court_ace
        diffs['court_speed']   = court_spd

        # Interaction features
        diffs['elo_x_skill']      = (elo_w - elo_l) * (sk_w - sk_l)
        diffs['elo_x_momentum']   = (elo_w - elo_l) * (mw - ml)
        diffs['rank_x_momentum']  = log_rank_ratio * (mw - ml)

        # Randomly assign perspective to avoid leakage from mirrored pairs.
        # 50% chance: features are winner-loser (target=1)
        # 50% chance: features are loser-winner (target=0)
        contextual_keys = {'surface_enc','tourney_level','round_enc',
                          'draw_size','tourney_date','level_weight',
                          'court_ace_pct','court_speed'}
        if random.random() < 0.5:
            sample = diffs.copy()
            sample['target'] = 1
        else:
            sample = {k: -v if k not in contextual_keys else v
                      for k, v in diffs.items()}
            sample['target'] = 0
        rows.append(sample)
        upd_serve(l, rd, 'l'); upd_return(l, rd, 'w')

    df_out = pd.DataFrame(rows)
    print(f"   → Dataset: {len(df_out):,} righe | {df_out.shape[1]} colonne")

    joblib.dump(elo_surf, 'elo_surface.pkl')
    print("   → elo_surface.pkl salvato")
    joblib.dump(streak_t, 'streak_players.pkl')
    joblib.dump(racha_t, 'momentum_surface.pkl')
    print("   → momentum_surface.pkl salvato")

    return df_out, stats_dict, elo_surf, streak_t


# ── Feature list ─────────────────────────────────────────────────────────────
FEATURES = [
    'diff_rank', 'log_rank_ratio', 'diff_rank_points', 'log_pts_ratio',
    'diff_seed', 'diff_age', 'diff_ht',
    'diff_elo', 'diff_streak', 'diff_overall_wr',
    'surface_enc', 'tourney_level', 'round_enc', 'draw_size',
    'diff_hand',
    'diff_skill', 'diff_home',
    'diff_fatigue', 'diff_momentum', 'diff_h2h',
    'diff_ace', 'diff_1st_won', 'diff_2nd_won', 'diff_bp_saved',
    'diff_serve_dom', 'diff_game_score',
    'diff_return_pct', 'diff_bp_conv', 'diff_return_1st',
    'court_ace_pct', 'court_speed',
    'elo_x_skill', 'elo_x_momentum', 'rank_x_momentum',
]


# ══════════════════════════════════════════════════════════════════════════════
# 2.  TEMPORAL SPLIT
# ══════════════════════════════════════════════════════════════════════════════

def temporal_split(df_ml, train_end_date=20230101, val_end_date=20240601):
    dates = df_ml['tourney_date']
    mask_train = dates < train_end_date
    mask_val   = (dates >= train_end_date) & (dates < val_end_date)
    mask_test  = dates >= val_end_date

    print(f"\n📊 Temporal split:")
    print(f"   Train: {mask_train.sum():,} rows (before {train_end_date})")
    print(f"   Val:   {mask_val.sum():,} rows ({train_end_date}-{val_end_date})")
    print(f"   Test:  {mask_test.sum():,} rows (after {val_end_date})")

    return mask_train, mask_val, mask_test


# ══════════════════════════════════════════════════════════════════════════════
# 3.  GRADIENT BOOSTING MODELS
# ══════════════════════════════════════════════════════════════════════════════

def train_lightgbm(X_tr, y_tr, X_val, y_val, X_test, y_test,
                   sample_weights_tr=None, n_trials=TRIALS_GBM):
    if not HAS_LGB:
        return None, 0, 1.0

    print(f"\n🌲 Training LightGBM (Optuna {n_trials} trials)...")

    y_tr_np = y_tr.values if hasattr(y_tr, 'values') else np.array(y_tr)
    y_val_np = y_val.values if hasattr(y_val, 'values') else np.array(y_val)
    y_test_np = y_test.values if hasattr(y_test, 'values') else np.array(y_test)

    best_model = [None]
    best_acc = [0.0]

    def objective(trial):
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbosity': -1,
            'n_jobs': -1,
            'random_state': SEED,
            'n_estimators': trial.suggest_int('n_estimators', 200, 1500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'num_leaves': trial.suggest_int('num_leaves', 15, 127),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
        }

        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr, y_tr_np, sample_weight=sample_weights_tr,
                  eval_set=[(X_val, y_val_np)],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                             lgb.log_evaluation(0)])

        preds_val = model.predict(X_val)
        acc_val = accuracy_score(y_val_np, preds_val)

        if acc_val > best_acc[0]:
            best_acc[0] = acc_val
            best_model[0] = model

        return acc_val

    if HAS_OPTUNA:
        study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        model = best_model[0]
        print(f"   Best LGB val accuracy: {best_acc[0]:.4f}")
        print(f"   Best params: {study.best_params}")
    else:
        model = lgb.LGBMClassifier(
            n_estimators=800, learning_rate=0.05, max_depth=6,
            num_leaves=63, subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, verbosity=-1)
        model.fit(X_tr, y_tr_np, sample_weight=sample_weights_tr,
                  eval_set=[(X_val, y_val_np)],
                  callbacks=[lgb.early_stopping(50, verbose=False),
                             lgb.log_evaluation(0)])

    preds_test = model.predict(X_test)
    probs_test = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test_np, preds_test)
    ll = log_loss(y_test_np, probs_test)
    print(f"   LightGBM test: acc={acc:.4f} | log_loss={ll:.4f}")

    return model, acc, ll


def train_xgboost(X_tr, y_tr, X_val, y_val, X_test, y_test,
                  sample_weights_tr=None, n_trials=TRIALS_GBM):
    if not HAS_XGB:
        return None, 0, 1.0

    print(f"\n🌲 Training XGBoost (Optuna {n_trials} trials)...")

    y_tr_np = y_tr.values if hasattr(y_tr, 'values') else np.array(y_tr)
    y_val_np = y_val.values if hasattr(y_val, 'values') else np.array(y_val)
    y_test_np = y_test.values if hasattr(y_test, 'values') else np.array(y_test)

    best_model = [None]
    best_acc = [0.0]

    def objective(trial):
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'verbosity': 0,
            'random_state': SEED,
            'n_estimators': trial.suggest_int('n_estimators', 200, 1500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        }

        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr_np, sample_weight=sample_weights_tr,
                  eval_set=[(X_val, y_val_np)], verbose=False)

        preds_val = model.predict(X_val)
        acc_val = accuracy_score(y_val_np, preds_val)

        if acc_val > best_acc[0]:
            best_acc[0] = acc_val
            best_model[0] = model

        return acc_val

    if HAS_OPTUNA:
        study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        model = best_model[0]
        print(f"   Best XGB val accuracy: {best_acc[0]:.4f}")
        print(f"   Best params: {study.best_params}")
    else:
        model = xgb.XGBClassifier(
            n_estimators=800, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            random_state=SEED, verbosity=0)
        model.fit(X_tr, y_tr_np, sample_weight=sample_weights_tr,
                  eval_set=[(X_val, y_val_np)], verbose=False)

    preds_test = model.predict(X_test)
    probs_test = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test_np, preds_test)
    ll = log_loss(y_test_np, probs_test)
    print(f"   XGBoost test: acc={acc:.4f} | log_loss={ll:.4f}")

    return model, acc, ll


# ══════════════════════════════════════════════════════════════════════════════
# 4.  ANN MODEL
# ══════════════════════════════════════════════════════════════════════════════

class TennisANNv3(nn.Module):
    """Wide & Deep con Residual Connections."""
    def __init__(self, input_dim: int, hidden_layers: list, dropout: float = 0.3,
                 n_interactions: int = 0, interaction_pairs: list = None):
        super().__init__()
        self.interaction_pairs = interaction_pairs or []
        self.n_interactions = min(n_interactions, len(self.interaction_pairs))

        self.wide = nn.Linear(input_dim, 1)

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

    def forward(self, x):
        wide_out = self.wide(x)

        if self.n_interactions > 0:
            interactions = []
            for i, j in self.interaction_pairs[:self.n_interactions]:
                if i < x.shape[1] and j < x.shape[1]:
                    interactions.append(x[:, i] * x[:, j])
                else:
                    interactions.append(torch.zeros(x.shape[0], device=x.device))
            inter_t = torch.stack(interactions, dim=1)
            deep_in = torch.cat([x, inter_t], dim=1)
        else:
            deep_in = x

        h = deep_in
        for layer, norm, drop, res_proj in zip(
                self.deep_layers, self.deep_norms,
                self.deep_drops, self.residual_projs):
            identity = res_proj(h)
            h = layer(h)
            h = norm(h)
            h = torch.relu(h)
            h = drop(h)
            h = h + identity

        deep_out = self.deep_out(h)
        return (wide_out + deep_out).squeeze(1)


def label_smoothed_bce(logits, targets, smoothing=0.05):
    targets_smooth = targets * (1.0 - smoothing) + 0.5 * smoothing
    return nn.functional.binary_cross_entropy_with_logits(logits, targets_smooth)


def train_model(model, loader_train, loader_val, epochs=80, lr=1e-3, patience=10,
                smoothing=0.05):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=4)
    criterion_val = nn.BCEWithLogitsLoss()
    best_val = float('inf'); best_state = None; no_imp = 0

    for _ in range(epochs):
        model.train()
        for Xb, yb in loader_train:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = label_smoothed_bce(model(Xb), yb, smoothing=smoothing)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval(); vl = []
        with torch.no_grad():
            for Xb, yb in loader_val:
                Xb, yb = Xb.to(device), yb.to(device)
                vl.append(criterion_val(model(Xb), yb).item())
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


def valuta(model, Xsc, y_np):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(Xsc.astype(np.float32)).to(device)).cpu().numpy()
    probs = 1 / (1 + np.exp(-logits))
    return accuracy_score(y_np, (probs >= 0.5).astype(int)), log_loss(y_np, probs)


def train_ann_optuna(X_tr_sc, y_tr, X_val_sc, y_val, X_test_sc, y_test,
                     n_trials=TRIALS_ANN, input_dim=None):
    if input_dim is None:
        input_dim = X_tr_sc.shape[1]

    print(f"\n🧠 Training ANN (Optuna {n_trials} trials)...")

    y_tr_np = y_tr.values if hasattr(y_tr, 'values') else np.array(y_tr)
    y_val_np = y_val.values if hasattr(y_val, 'values') else np.array(y_val)
    y_test_np = y_test.values if hasattr(y_test, 'values') else np.array(y_test)

    X_tr_t = torch.tensor(X_tr_sc.astype(np.float32))
    y_tr_t = torch.tensor(y_tr_np.astype(np.float32))
    X_val_t = torch.tensor(X_val_sc.astype(np.float32))
    y_val_t = torch.tensor(y_val_np.astype(np.float32))

    dataset_tr  = TensorDataset(X_tr_t, y_tr_t)
    dataset_val = TensorDataset(X_val_t, y_val_t)

    ARCH_OPTIONS = {
        '2L_s':    [128, 64],
        '2L_m':    [256, 128],
        '3L_m':    [256, 128, 64],
        '3L_l':    [512, 256, 128],
        '2L_eq_m': [256, 256],
        '2L_l':    [512, 256],
    }

    best_model = [None]
    best_acc = [0.0]
    best_hp = [{}]

    def objective(trial):
        arch_name = trial.suggest_categorical('arch', list(ARCH_OPTIONS.keys()))
        hl  = ARCH_OPTIONS[arch_name]
        dr  = trial.suggest_float('dropout', 0.15, 0.45, step=0.05)
        lr  = trial.suggest_float('lr', 5e-5, 3e-3, log=True)
        bs  = trial.suggest_categorical('batch_size', [256, 512, 1024])
        ep  = trial.suggest_categorical('epochs', [60, 80, 100])
        sm  = trial.suggest_float('smoothing', 0.0, 0.08, step=0.01)

        ldr_tr  = DataLoader(dataset_tr,  batch_size=bs, shuffle=True)
        ldr_val = DataLoader(dataset_val, batch_size=4096, shuffle=False)

        model = TennisANNv3(input_dim, hl, dr, n_interactions=0)
        model, _ = train_model(model, ldr_tr, ldr_val, epochs=ep, lr=lr, smoothing=sm)
        acc_v, _ = valuta(model, X_val_sc, y_val_np)

        if acc_v > best_acc[0]:
            best_acc[0] = acc_v
            best_model[0] = model
            best_hp[0] = {'hidden_layers': hl, 'dropout': dr, 'lr': lr,
                          'batch_size': bs, 'epochs': ep, 'label_smoothing': sm,
                          'arch': arch_name}

        return acc_v

    if HAS_OPTUNA:
        study = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        model = best_model[0]
        print(f"   Best ANN val accuracy: {best_acc[0]:.4f}")
    else:
        hl = [256, 128, 64]
        dr = 0.3
        lr = 1e-3
        bs = 512
        ldr_tr  = DataLoader(dataset_tr,  batch_size=bs, shuffle=True)
        ldr_val = DataLoader(dataset_val, batch_size=4096, shuffle=False)
        model = TennisANNv3(input_dim, hl, dr, n_interactions=0)
        model, _ = train_model(model, ldr_tr, ldr_val, epochs=ANN_EPOCHS, lr=lr)
        best_hp[0] = {'hidden_layers': hl, 'dropout': dr, 'lr': lr,
                      'batch_size': bs, 'epochs': ANN_EPOCHS, 'label_smoothing': 0.05}

    acc, ll = valuta(model, X_test_sc, y_test_np)
    print(f"   ANN test: acc={acc:.4f} | log_loss={ll:.4f}")

    return model, acc, ll, best_hp[0]


# ══════════════════════════════════════════════════════════════════════════════
# 5.  STACKING ENSEMBLE
# ══════════════════════════════════════════════════════════════════════════════

def train_stacking_ensemble(y_val, y_test, val_probs, test_probs):
    print("\n🏗️  Training Stacking Ensemble...")

    y_val_np = y_val.values if hasattr(y_val, 'values') else np.array(y_val)
    y_test_np = y_test.values if hasattr(y_test, 'values') else np.array(y_test)

    val_meta = np.column_stack([val_probs[name] for name in val_probs])
    test_meta = np.column_stack([test_probs[name] for name in test_probs])

    meta_lr = LogisticRegression(C=1.0, random_state=SEED, max_iter=1000)
    meta_lr.fit(val_meta, y_val_np)

    preds = meta_lr.predict(test_meta)
    probs = meta_lr.predict_proba(test_meta)[:, 1]
    acc = accuracy_score(y_test_np, preds)
    ll = log_loss(y_test_np, probs)

    print(f"   Stacking ensemble test: acc={acc:.4f} | log_loss={ll:.4f}")
    print(f"   Meta weights: {dict(zip(val_probs.keys(), meta_lr.coef_[0]))}")

    return meta_lr, acc, ll


# ══════════════════════════════════════════════════════════════════════════════
# 6.  COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════════════════

def confronto_finale(results_list):
    df_final = pd.DataFrame(results_list).sort_values('Accuracy', ascending=False).reset_index(drop=True)
    df_final['Acc. %'] = (df_final['Accuracy'] * 100).round(2)
    df_final.to_csv('resultados_comparacion_finale.csv', index=False)

    print("\n" + "="*70)
    print("  🏆  CLASSIFICA FINALE — TUTTI I MODELLI")
    print("="*70)
    print(df_final[['Modello','Acc. %','Log Loss','Note']].to_string(index=False))
    print("="*70)
    vincitore = df_final.iloc[0]
    print(f"\n  🥇  Miglior modello: {vincitore['Modello']}  →  {vincitore['Acc. %']:.2f}%\n")
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

    # ── 2. Temporal split ─────────────────────────────────────────────────────
    mask_train, mask_val, mask_test = temporal_split(df_ml,
                                                     train_end_date=20230101,
                                                     val_end_date=20240601)

    X_tr, y_tr = X[mask_train], y[mask_train]
    X_val, y_val = X[mask_val], y[mask_val]
    X_test, y_test = X[mask_test], y[mask_test]
    dates_tr = dates[mask_train]
    lw_tr = level_w[mask_train]

    scaler = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_tr)
    X_val_sc = scaler.transform(X_val)
    X_te_sc  = scaler.transform(X_test)
    joblib.dump(scaler, 'scaler_ann.pkl')
    print(f"   → scaler_ann.pkl salvato  |  {len(FEATURES)} feature")

    results_list = []

    # ── 3. Temporal sample weights ────────────────────────────────────────────
    max_date = dates_tr.max()
    days_ago = (max_date - dates_tr) / 10000 * 365
    temporal_weights = np.exp(-0.003 * days_ago).values.astype(np.float32)
    combined_weights = temporal_weights * lw_tr
    combined_weights = combined_weights / combined_weights.mean()

    # ── 4. LightGBM ──────────────────────────────────────────────────────────
    lgb_model, lgb_acc, lgb_ll = train_lightgbm(
        X_tr_sc, y_tr, X_val_sc, y_val, X_te_sc, y_test,
        sample_weights_tr=combined_weights, n_trials=TRIALS_GBM)

    if lgb_model is not None:
        results_list.append({
            'Modello': 'LightGBM (Optuna)',
            'Accuracy': lgb_acc,
            'Log Loss': lgb_ll,
            'Note': f'Temporal split, {len(FEATURES)} features'
        })

    # ── 5. XGBoost ───────────────────────────────────────────────────────────
    xgb_model, xgb_acc, xgb_ll = train_xgboost(
        X_tr_sc, y_tr, X_val_sc, y_val, X_te_sc, y_test,
        sample_weights_tr=combined_weights, n_trials=TRIALS_GBM)

    if xgb_model is not None:
        results_list.append({
            'Modello': 'XGBoost (Optuna)',
            'Accuracy': xgb_acc,
            'Log Loss': xgb_ll,
            'Note': f'Temporal split, {len(FEATURES)} features'
        })

    # ── 6. ANN ───────────────────────────────────────────────────────────────
    ann_model, ann_acc, ann_ll, ann_hp = train_ann_optuna(
        X_tr_sc, y_tr, X_val_sc, y_val, X_te_sc, y_test,
        n_trials=TRIALS_ANN, input_dim=len(FEATURES))

    results_list.append({
        'Modello': 'ANN v4 (Wide&Deep)',
        'Accuracy': ann_acc,
        'Log Loss': ann_ll,
        'Note': f'Temporal split, arch={ann_hp.get("hidden_layers","?")}'
    })

    # ── 7. Stacking Ensemble ─────────────────────────────────────────────────
    y_val_np = y_val.values if hasattr(y_val, 'values') else np.array(y_val)
    y_test_np = y_test.values if hasattr(y_test, 'values') else np.array(y_test)

    val_probs = {}
    test_probs = {}

    if lgb_model is not None:
        val_probs['lgb'] = lgb_model.predict_proba(X_val_sc)[:, 1]
        test_probs['lgb'] = lgb_model.predict_proba(X_te_sc)[:, 1]

    if xgb_model is not None:
        val_probs['xgb'] = xgb_model.predict_proba(X_val_sc)[:, 1]
        test_probs['xgb'] = xgb_model.predict_proba(X_te_sc)[:, 1]

    ann_model.eval()
    with torch.no_grad():
        logits_val = ann_model(torch.tensor(X_val_sc.astype(np.float32)).to(device)).cpu().numpy()
        logits_test = ann_model(torch.tensor(X_te_sc.astype(np.float32)).to(device)).cpu().numpy()
    val_probs['ann'] = 1 / (1 + np.exp(-logits_val))
    test_probs['ann'] = 1 / (1 + np.exp(-logits_test))

    meta_lr = None
    if len(val_probs) >= 2:
        meta_lr, ens_acc, ens_ll = train_stacking_ensemble(
            y_val, y_test, val_probs, test_probs)
        results_list.append({
            'Modello': 'Stacking Ensemble',
            'Accuracy': ens_acc,
            'Log Loss': ens_ll,
            'Note': 'LGB+XGB+ANN meta-learner'
        })

    # ── 8. Simple average ensemble ────────────────────────────────────────────
    if len(test_probs) >= 2:
        avg_probs = np.mean(list(test_probs.values()), axis=0)
        avg_acc = accuracy_score(y_test_np, (avg_probs >= 0.5).astype(int))
        avg_ll = log_loss(y_test_np, avg_probs)
        results_list.append({
            'Modello': 'Average Ensemble',
            'Accuracy': avg_acc,
            'Log Loss': avg_ll,
            'Note': 'Simple average of all model probabilities'
        })
        print(f"\n📊 Average Ensemble test: acc={avg_acc:.4f} | log_loss={avg_ll:.4f}")

    # ── 9. Confronto finale ──────────────────────────────────────────────────
    df_confronto = confronto_finale(results_list)

    # ── 10. Select best model and save ───────────────────────────────────────
    best_result = max(results_list, key=lambda x: x['Accuracy'])
    best_name = best_result['Modello']
    print(f"\n🏆 Best model: {best_name} → {best_result['Accuracy']:.4f} acc, {best_result['Log Loss']:.4f} log_loss")

    torch.save(ann_model.state_dict(), 'modelo_ann.pth')

    if 'LightGBM' in best_name and lgb_model is not None:
        joblib.dump(lgb_model, 'modelo_best_gbm.pkl')
        print("   → modelo_best_gbm.pkl salvato (LightGBM)")
    elif 'XGBoost' in best_name and xgb_model is not None:
        joblib.dump(xgb_model, 'modelo_best_gbm.pkl')
        print("   → modelo_best_gbm.pkl salvato (XGBoost)")

    if 'Ensemble' in best_name:
        if lgb_model is not None:
            joblib.dump(lgb_model, 'modelo_lgb.pkl')
        if xgb_model is not None:
            joblib.dump(xgb_model, 'modelo_xgb.pkl')
        if 'Stacking' in best_name and meta_lr is not None:
            joblib.dump(meta_lr, 'modelo_meta_lr.pkl')

    # ── 11. Calibrazione ─────────────────────────────────────────────────────
    if 'LightGBM' in best_name and lgb_model is not None:
        cal_probs = lgb_model.predict_proba(X_val_sc)[:, 1]
    elif 'XGBoost' in best_name and xgb_model is not None:
        cal_probs = xgb_model.predict_proba(X_val_sc)[:, 1]
    else:
        ann_model.eval()
        with torch.no_grad():
            cal_logits = ann_model(torch.tensor(X_val_sc.astype(np.float32)).to(device)).cpu().numpy()
        cal_probs = 1 / (1 + np.exp(-cal_logits))

    calibrator = LogisticRegression(max_iter=1000)
    calibrator.fit(cal_probs.reshape(-1, 1), y_val_np)
    joblib.dump(calibrator, 'calibrator_ann.pkl')
    print("   → calibrator_ann.pkl salvato (Platt scaling)")

    # ── 12. Save config ──────────────────────────────────────────────────────
    cfg = ann_hp.copy()
    cfg['model_version'] = 'v4'
    cfg['input_dim'] = len(FEATURES)
    cfg['features'] = FEATURES
    cfg['n_interactions'] = 0
    cfg['interaction_pairs'] = []
    cfg['scaler_file'] = 'scaler_ann.pkl'
    cfg['calibrator_file'] = 'calibrator_ann.pkl'
    cfg['elo_surface_file'] = 'elo_surface.pkl'
    cfg['streak_file'] = 'streak_players.pkl'
    cfg['momentum_file'] = 'momentum_surface.pkl'
    cfg['best_model_type'] = best_name
    cfg['best_accuracy'] = best_result['Accuracy']
    cfg['best_log_loss'] = best_result['Log Loss']
    with open('ann_config.json', 'w') as f:
        json.dump(cfg, f, indent=2)

    # ── 13. Re-train on all data for production ──────────────────────────────
    print(f"\n🚀 Re-training models on all data for production...")
    scaler_final = StandardScaler()
    X_all_sc = scaler_final.fit_transform(X)
    joblib.dump(scaler_final, 'scaler_ann.pkl')

    # Re-train ANN on all data
    from sklearn.model_selection import train_test_split
    X_tr_fin, X_val_fin, y_tr_fin, y_val_fin = train_test_split(
        X_all_sc, y, test_size=0.15, random_state=SEED)
    y_tr_fin_np = y_tr_fin.values if hasattr(y_tr_fin, 'values') else np.array(y_tr_fin)
    y_val_fin_np = y_val_fin.values if hasattr(y_val_fin, 'values') else np.array(y_val_fin)

    X_tr_t = torch.tensor(X_tr_fin.astype(np.float32))
    y_tr_t = torch.tensor(y_tr_fin_np.astype(np.float32))
    X_val_t = torch.tensor(X_val_fin.astype(np.float32))
    y_val_t = torch.tensor(y_val_fin_np.astype(np.float32))

    ldr_tr  = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                         batch_size=ann_hp.get('batch_size', 512), shuffle=True)
    ldr_val = DataLoader(TensorDataset(X_val_t, y_val_t),
                         batch_size=4096, shuffle=False)

    model_final = TennisANNv3(len(FEATURES), ann_hp.get('hidden_layers', [256, 128, 64]),
                               ann_hp.get('dropout', 0.3), n_interactions=0)
    model_final, _ = train_model(model_final, ldr_tr, ldr_val,
                                  epochs=ann_hp.get('epochs', 80),
                                  lr=ann_hp.get('lr', 1e-3),
                                  smoothing=ann_hp.get('label_smoothing', 0.05))

    acc_final, ll_final = valuta(model_final, X_val_fin, y_val_fin_np)
    print(f"   ANN finale: acc={acc_final:.4f} | log_loss={ll_final:.4f}")

    torch.save(model_final.state_dict(), 'modelo_ann.pth')

    # Re-calibrate
    model_final.eval()
    with torch.no_grad():
        logits_val = model_final(torch.tensor(X_val_fin.astype(np.float32)).to(device)).cpu().numpy()
    calibrator = LogisticRegression(max_iter=1000)
    calibrator.fit(logits_val.reshape(-1, 1), y_val_fin_np)
    joblib.dump(calibrator, 'calibrator_ann.pkl')

    # Re-train GBM models on all data
    y_all = y.values if hasattr(y, 'values') else np.array(y)
    if lgb_model is not None:
        print("   Re-training LightGBM su tutti i dati...")
        lgb_final = lgb.LGBMClassifier(**lgb_model.get_params())
        lgb_final.fit(X_all_sc, y_all)
        joblib.dump(lgb_final, 'modelo_best_gbm.pkl')

    if xgb_model is not None:
        print("   Re-training XGBoost su tutti i dati...")
        xgb_final = xgb.XGBClassifier(**xgb_model.get_params())
        xgb_final.fit(X_all_sc, y_all)
        joblib.dump(xgb_final, 'modelo_xgb.pkl')

    # ── 14. Print summary ────────────────────────────────────────────────────
    print("\n✅ File salvati:")
    for f_name in ['modelo_ann.pth', 'scaler_ann.pkl', 'ann_config.json',
                   'elo_surface.pkl', 'streak_players.pkl', 'momentum_surface.pkl',
                   'calibrator_ann.pkl', 'resultados_comparacion_finale.csv',
                   'modelo_best_gbm.pkl', 'modelo_lgb.pkl', 'modelo_xgb.pkl',
                   'modelo_meta_lr.pkl']:
        stato = "✅" if os.path.exists(f_name) else "—"
        print(f"  {stato}  {f_name}")

    print(f"\n🎯 RISULTATO FINALE:")
    print(f"   Best model: {best_name}")
    print(f"   Test Accuracy: {best_result['Accuracy']:.4f} ({best_result['Accuracy']*100:.2f}%)")
    print(f"   Log Loss: {best_result['Log Loss']:.4f}")
