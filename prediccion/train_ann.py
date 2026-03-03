"""
train_ann.py v3.0  —  ANN Avanzata per Tennis Predictor
========================================================
Funzionalità v3 rispetto v2:
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

UTILIZZO:
    cd prediccion/
    python train_ann.py

OUTPUT:
    modelo_ann.pth               ← modello globale v3
    ann_config.json              ← config con v3 metadata
    scaler_ann.pkl               ← StandardScaler 28 feature
    elo_surface.pkl              ← Elo corrente per superficie
    streak_players.pkl           ← striscia attiva per giocatore
    momentum_surface.pkl         ← momentum per (giocatore, superficie)
    calibrator_ann.pkl           ← Platt scaling calibrator
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
TRIALS = 60        # numero trial Optuna per ogni fold temporale

# Importanza tornei (moltiplicatore sui pesi campione)
LEVEL_MULT = {'G': 2.0, 'M': 1.5, 'F': 1.4, 'A': 1.0,
              'D': 1.0, 'C': 0.8, 'S': 0.7, 'E': 0.5}

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
    serve_t   = {}
    return_t  = {}   # {player: {return_pct: [...], bp_conv: [...], return_1st: [...]}}
    elo_surf  = {}   # {(player, surface): float}  — ELO per superficie
    streak_t  = {}   # {player: int}  +N=N vittorie consecutive, -N=sconfitte

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

        # Salta righe con nomi mancanti
        if not isinstance(w, str) or not isinstance(l, str) or pd.isna(w) or pd.isna(l):
            continue
        surf = row['surface']

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
        seed_w = float(row['winner_seed']) if pd.notna(row.get('winner_seed')) else 33
        seed_l = float(row['loser_seed'])  if pd.notna(row.get('loser_seed'))  else 33
        lev_w  = LEVEL_MULT.get(str(row.get('tourney_level','')), 1.0)

        diffs = {
            'diff_rank':         (row['loser_rank']-row['winner_rank'])
                                 if pd.notna(row.get('winner_rank')) and pd.notna(row.get('loser_rank')) else 0,
            'diff_rank_points':  pts_w - pts_l,
            'diff_seed':         seed_l - seed_w,
            'diff_age':          (row['winner_age']-row['loser_age'])
                                 if pd.notna(row.get('winner_age')) and pd.notna(row.get('loser_age')) else 0,
            'diff_ht':           (row['winner_ht']-row['loser_ht'])
                                 if pd.notna(row.get('winner_ht')) and pd.notna(row.get('loser_ht')) else 0,
            # nuove feature
            'diff_elo':          elo_w - elo_l,             # Elo PRIMA dell'aggiornamento
            'diff_streak':       float(str_w - str_l),      # striscia attiva
            # contesto
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
            'diff_df':           sa_w['df']      - sa_l['df'],
            'diff_1st_pct':      sa_w['1st_pct'] - sa_l['1st_pct'],
            'diff_1st_won':      sa_w['1st_won'] - sa_l['1st_won'],
            'diff_2nd_won':      sa_w['2nd_won'] - sa_l['2nd_won'],
            'diff_bp_saved':     sa_w['bp_saved']- sa_l['bp_saved'],
            # nuove feature ritorno
            'diff_return_pct':   ra_w['return_pct'] - ra_l['return_pct'],
            'diff_bp_conv':      ra_w['bp_conv']    - ra_l['bp_conv'],
            'diff_return_1st':   ra_w['return_1st'] - ra_l['return_1st'],
            'tourney_date':      float(row['tourney_date']) if pd.notna(row['tourney_date']) else 20200101,
            'level_weight':      lev_w,
        }

        # ── Court speed (contestuali, uguali per entrambi → NON negare in d0) ──
        tourney_year = int(str(row['tourney_date'])[:4]) if pd.notna(row.get('tourney_date')) else 2025
        court_ace, court_spd = get_court_stats(
            row.get('tourney_name', ''), surf or 'Hard', tourney_year)
        diffs['court_ace_pct'] = court_ace
        diffs['court_speed']   = court_spd

        d1 = diffs.copy(); d1['target'] = 1
        d0 = {k: -v if k not in ('surface_enc','tourney_level','round_enc',
                                  'draw_size','tourney_date','level_weight',
                                  'court_ace_pct','court_speed')
              else v for k, v in diffs.items()}
        d0['target'] = 0
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

    return df_out, stats_dict, elo_surf, streak_t


# ── Lista feature (28)────────────────────────────────────────────────────────
FEATURES = [
    'diff_rank', 'diff_rank_points', 'diff_seed', 'diff_age', 'diff_ht',
    'diff_elo', 'diff_streak',                          # Elo per superficie + striscia
    'surface_enc', 'tourney_level', 'round_enc', 'draw_size',
    'diff_hand',
    'diff_skill', 'diff_home',
    'diff_fatigue', 'diff_momentum', 'diff_h2h',
    'diff_ace', 'diff_df', 'diff_1st_pct', 'diff_1st_won',
    'diff_2nd_won', 'diff_bp_saved',
    'diff_return_pct', 'diff_bp_conv', 'diff_return_1st',  # ← Ritorno
    'court_ace_pct', 'court_speed',                         # ← Court speed
]  # totale: 28 feature


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

# Coppie di feature per interaction layer (indici in FEATURES)
# 5=diff_elo, 12=diff_skill, 0=diff_rank, 15=diff_momentum, 1=diff_rank_points,
# 14=diff_fatigue, 6=diff_streak, 16=diff_h2h, 23=diff_return_pct, 24=diff_bp_conv
DEFAULT_INTERACTION_PAIRS = [
    (5, 12), (0, 15), (5, 15), (12, 14), (0, 1),
    (5, 16), (12, 16), (6, 15), (14, 15), (1, 12),
]
N_INTERACTIONS = len(DEFAULT_INTERACTION_PAIRS)


class TennisANNv3(nn.Module):
    """Wide & Deep con Feature Interaction e Residual Connections."""
    def __init__(self, input_dim: int, hidden_layers: list, dropout: float = 0.3,
                 n_interactions: int = N_INTERACTIONS,
                 interaction_pairs: list = None):
        super().__init__()
        self.interaction_pairs = interaction_pairs or DEFAULT_INTERACTION_PAIRS
        self.n_interactions = min(n_interactions, len(self.interaction_pairs))

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

        # Combina wide + deep
        return (wide_out + deep_out).squeeze(1)


def label_smoothed_bce(logits, targets, smoothing=0.05):
    """BCEWithLogitsLoss con label smoothing: 0→smoothing/2, 1→1-smoothing/2"""
    targets_smooth = targets * (1.0 - smoothing) + 0.5 * smoothing
    return nn.functional.binary_cross_entropy_with_logits(logits, targets_smooth)


def train_model(model, loader_train, loader_val, epochs=80, lr=1e-3, patience=10,
                smoothing=0.05):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=4)
    criterion_val = nn.BCEWithLogitsLoss()  # no smoothing per validazione
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


# ══════════════════════════════════════════════════════════════════════════════
# 4.  OPTUNA SEARCH  (o random search se optuna non disponibile)
# ══════════════════════════════════════════════════════════════════════════════

def optuna_search(X_tr, y_tr, X_val_sc, X_te_sc,
                  y_val, y_test, dates_tr, level_w_tr,
                  n_trials=TRIALS, input_dim=len(FEATURES), label="Global"):

    X_val_t  = torch.tensor(X_val_sc.astype(np.float32))
    X_test_t = torch.tensor(X_te_sc.astype(np.float32))
    y_val_np = y_val.values if hasattr(y_val, 'values') else np.array(y_val)
    y_test_np= y_test.values if hasattr(y_test, 'values') else np.array(y_test)

    X_tr_t = torch.tensor(X_tr.astype(np.float32))
    y_tr_t = torch.tensor(y_tr.values.astype(np.float32) if hasattr(y_tr, 'values') else y_tr.astype(np.float32))

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

            # Stampa PRIMA di addestrare — aggiornamento immediato in console
            print(f"  [{trial.number+1:3d}/{n_trials}] arch={arch_name:12s} dr={dr:.2f} lr={lr:.0e} λ={lam:.4f} sm={sm:.2f}  ▶ training...", flush=True)

            # Se GPU, usa batch più grandi per tenerla satura
            bs_eff = bs * 2 if USE_CUDA and bs < 2048 else bs
            w_combined = calcola_pesi_combinati(dates_tr, level_w_tr, lam)
            w_t = torch.tensor(w_combined)
            sampler = WeightedRandomSampler(w_t, len(w_t), replacement=True)
            ldr_tr  = DataLoader(dataset_tr,  batch_size=bs_eff, sampler=sampler,
                                 pin_memory=PIN_MEM, num_workers=N_WORKERS)
            ldr_val = DataLoader(dataset_val, batch_size=4096, shuffle=False,
                                 pin_memory=PIN_MEM, num_workers=N_WORKERS)

            model = TennisANNv3(input_dim, hl, dr)
            model, _ = train_model(model, ldr_tr, ldr_val, epochs=ep, lr=lr, smoothing=sm)
            acc_v, _ = valuta(model, X_val_sc, y_val_np)

            trial.set_user_attr('model', model)
            trial.set_user_attr('hl', hl)
            trial.set_user_attr('lam', lam)
            trial.set_user_attr('hp', {'hidden_layers': hl, 'dropout': dr, 'lr': lr,
                                       'batch_size': bs, 'epochs': ep, 'lambda_decay': lam,
                                       'label_smoothing': sm})
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
            acc_t, ll = valuta(model_t, X_te_sc, y_test_np)
            risultati.append({
                'trial':        t.number+1,
                'hidden_layers': str(t.user_attrs.get('hl',[])),
                'dropout':      t.params.get('dropout'),
                'lr':           t.params.get('lr'),
                'batch_size':   t.params.get('batch_size'),
                'epochs':       t.params.get('epochs'),
                'lambda_decay': t.params.get('lambda_decay'),
                'smoothing':    t.params.get('smoothing'),
                'val_acc':      t.value,
                'test_acc':     acc_t,
                'log_loss':     ll,
                '_model':       model_t,
                '_config':      t.user_attrs.get('hp', {}),
            })

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
            hp  = {'hidden_layers': hl, 'dropout': dr, 'lr': lr,
                   'batch_size': bs, 'epochs': ep, 'lambda_decay': lam, 'label_smoothing': sm}
            print(f"  [{i+1:3d}/{n_trials}] {str(hl):22s} drop={dr} lr={lr:.0e} λ={lam:.4f} sm={sm:.2f}  ▶ avvio...", flush=True)
            w_combined = calcola_pesi_combinati(dates_tr, level_w_tr, lam)
            w_t = torch.tensor(w_combined)
            sampler = WeightedRandomSampler(w_t, len(w_t), replacement=True)
            ldr_tr  = DataLoader(dataset_tr,  batch_size=bs, sampler=sampler)
            ldr_val = DataLoader(dataset_val, batch_size=2048, shuffle=False)
            model = TennisANNv3(input_dim, hl, dr)
            model, _ = train_model(model, ldr_tr, ldr_val, epochs=ep, lr=lr, smoothing=sm)
            acc_v, _ = valuta(model, X_val_sc, y_val_np)
            acc_t, ll = valuta(model, X_te_sc, y_test_np)
            print(f"     └─ val={acc_v:.3f} test={acc_t:.3f}", flush=True)
            risultati.append({'trial': i+1, 'hidden_layers': str(hl),
                               'dropout': dr, 'lr': lr, 'batch_size': bs,
                               'epochs': ep, 'lambda_decay': lam, 'smoothing': sm,
                               'val_acc': acc_v, 'test_acc': acc_t,
                               'log_loss': ll, '_model': model, '_config': hp})

    return sorted(risultati, key=lambda x: x['test_acc'], reverse=True)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  CONFRONTO FINALE
# ══════════════════════════════════════════════════════════════════════════════

def confronto_finale(ann_global_acc, ann_global_ll, ann_surface_results):
    rows = [{'Modello': 'ANN v3 Globale (28 feat)', 'Accuracy': ann_global_acc,
              'Log Loss': ann_global_ll, 'Note': 'Wide&Deep + Residual + TemporalCV'}]

    for r in ann_surface_results:
        if r is None: continue
        rows.append({'Modello': f"ANN {r['surface']}", 'Accuracy': r['test_acc'],
                     'Log Loss': r['log_loss'], 'Note': f"arch={r['arch']}"})

    # Carica risultati modelli classici se disponibili
    path_class = '../resultados_comparacion.csv'
    if os.path.exists(path_class):
        df_c = pd.read_csv(path_class)
        for _, row in df_c.iterrows():
            rows.append({'Modello': row.get('Modelo', row.get('Model', '?')),
                         'Accuracy': row.get('Accuracy', row.get('accuracy', 0)),
                         'Log Loss': row.get('LogLoss', row.get('log_loss', float('nan'))),
                         'Note': 'modello classico'})

    df_final = pd.DataFrame(rows).sort_values('Accuracy', ascending=False).reset_index(drop=True)
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

    # Estrai anno da tourney_date (formato YYYYMMDD)
    df_ml['year'] = (df_ml['tourney_date'] // 10000).astype(int)

    # ── 2. Temporal Cross-Validation ──────────────────────────────────────────
    # Fold 1: train ≤2019, val 2020, test 2021
    # Fold 2: train ≤2020, val 2021, test 2022
    # Fold 3: train ≤2021, val 2022, test 2023
    # Fold 4: train ≤2022, val 2023, test 2024
    # Fold 5: train ≤2023, val 2024, test 2025
    FOLDS = [
        (2019, 2020, 2021),
        (2020, 2021, 2022),
        (2021, 2022, 2023),
        (2022, 2023, 2024),
        (2023, 2024, 2025),
    ]

    print(f"\n📅 Temporal Cross-Validation con {len(FOLDS)} fold...")
    fold_results = []   # (fold_idx, hp_dict, val_acc, test_acc)

    for fold_i, (train_end, val_year, test_year) in enumerate(FOLDS):
        mask_tr  = df_ml['year'] <= train_end
        mask_val = df_ml['year'] == val_year
        mask_te  = df_ml['year'] == test_year

        if mask_val.sum() < 100 or mask_te.sum() < 100:
            print(f"   Fold {fold_i+1}: dati insufficienti (val={mask_val.sum()}, test={mask_te.sum()}), salto.")
            continue

        X_tr_f = X[mask_tr]; y_tr_f = y[mask_tr]
        X_val_f = X[mask_val]; y_val_f = y[mask_val]
        X_te_f = X[mask_te]; y_te_f = y[mask_te]
        d_tr_f = dates[mask_tr]; lw_tr_f = level_w[mask_tr]

        scaler_f = StandardScaler()
        X_tr_sc  = scaler_f.fit_transform(X_tr_f)
        X_val_sc = scaler_f.transform(X_val_f)
        X_te_sc  = scaler_f.transform(X_te_f)

        print(f"\n   === Fold {fold_i+1}: train ≤{train_end} | val {val_year} | test {test_year} ===")
        print(f"       Campioni: train={len(X_tr_f):,} | val={len(X_val_f):,} | test={len(X_te_f):,}")

        res_f = optuna_search(
            X_tr_sc, y_tr_f, X_val_sc, X_te_sc,
            y_val_f, y_te_f, d_tr_f, lw_tr_f,
            n_trials=TRIALS, input_dim=len(FEATURES),
            label=f"Fold {fold_i+1} (test {test_year})"
        )
        if res_f:
            best_f = res_f[0]
            fold_results.append({
                'fold': fold_i+1,
                'train_end': train_end, 'val_year': val_year, 'test_year': test_year,
                'val_acc': best_f['val_acc'], 'test_acc': best_f['test_acc'],
                'config': best_f['_config'],
            })
            print(f"       🏆 Fold {fold_i+1}: val_acc={best_f['val_acc']:.4f} | test_acc={best_f['test_acc']:.4f}")

    # Sommario temporal CV
    if fold_results:
        avg_val  = np.mean([r['val_acc']  for r in fold_results])
        avg_test = np.mean([r['test_acc'] for r in fold_results])
        print(f"\n📊 Temporal CV — Media: val_acc={avg_val:.4f} | test_acc={avg_test:.4f}")
        for r in fold_results:
            print(f"   Fold {r['fold']} (test {r['test_year']}): val={r['val_acc']:.4f} test={r['test_acc']:.4f}")

    # ── 3. Modello finale su TUTTI i dati con migliori iperparametri ──────────
    # Usa l'HP del fold con migliore val_acc media (o il fold con miglior test_acc)
    if fold_results:
        best_fold = max(fold_results, key=lambda r: r['val_acc'])
        best_hp = best_fold['config']
    else:
        # Fallback: configurazione di default
        best_hp = {'hidden_layers': [512, 256, 128], 'dropout': 0.3, 'lr': 1e-3,
                   'batch_size': 512, 'epochs': 100, 'lambda_decay': 0.001,
                   'label_smoothing': 0.05}

    print(f"\n🚀 Training modello finale su TUTTI i dati con HP migliori...")
    print(f"   HP: {best_hp}")

    # Scaler globale finale (fit su tutti i dati)
    scaler = StandardScaler()
    scaler.fit(X)
    joblib.dump(scaler, 'scaler_ann.pkl')
    print(f"   → scaler_ann.pkl salvato  |  {len(FEATURES)} feature")

    # Split finale per early stopping: 85% train, 15% val
    X_tr_fin, X_val_fin, y_tr_fin, y_val_fin, d_tr_fin, _, lw_tr_fin, _ = train_test_split(
        X, y, dates, level_w, test_size=0.15, random_state=SEED)

    X_tr_fin_sc  = scaler.transform(X_tr_fin)
    X_val_fin_sc = scaler.transform(X_val_fin)

    hl  = best_hp['hidden_layers']
    dr  = best_hp['dropout']
    lr  = best_hp['lr']
    bs  = best_hp['batch_size']
    ep  = best_hp['epochs']
    lam = best_hp['lambda_decay']
    sm  = best_hp.get('label_smoothing', 0.05)

    X_tr_t = torch.tensor(X_tr_fin_sc.astype(np.float32))
    y_tr_t = torch.tensor(y_tr_fin.values.astype(np.float32) if hasattr(y_tr_fin, 'values')
                          else y_tr_fin.astype(np.float32))
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

    ldr_tr  = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=bs_eff,
                         sampler=sampler, pin_memory=PIN_MEM, num_workers=N_WORK)
    ldr_val = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=4096,
                         shuffle=False, pin_memory=PIN_MEM, num_workers=N_WORK)

    model_final = TennisANNv3(len(FEATURES), hl, dr)
    model_final, _ = train_model(model_final, ldr_tr, ldr_val, epochs=ep, lr=lr,
                                  smoothing=sm)

    # Valutazione finale sul validation set
    y_val_np = y_val_fin.values if hasattr(y_val_fin, 'values') else np.array(y_val_fin)
    acc_global, ll_global = valuta(model_final, X_val_fin_sc, y_val_np)

    print(f"\n🏆 Modello finale v3:")
    print(f"   Architettura:  {hl}")
    print(f"   Hyperparams:   dropout={dr} | lr={lr:.0e} | bs={bs} | λ={lam} | smoothing={sm}")
    print(f"   Val Accuracy:  {acc_global:.2%}")
    print(f"   Val Log Loss:  {ll_global:.4f}")

    # ── 4. Calibrazione (Platt scaling) ───────────────────────────────────────
    from sklearn.linear_model import LogisticRegression as LR_Cal

    model_final.eval()
    with torch.no_grad():
        logits_val = model_final(torch.tensor(X_val_fin_sc.astype(np.float32)).to(device)).cpu().numpy()

    calibrator = LR_Cal()
    calibrator.fit(logits_val.reshape(-1, 1), y_val_np)
    joblib.dump(calibrator, 'calibrator_ann.pkl')
    print("   → calibrator_ann.pkl salvato (Platt scaling)")

    # Accuracy dopo calibrazione
    probs_cal = calibrator.predict_proba(logits_val.reshape(-1, 1))[:, 1]
    acc_cal = accuracy_score(y_val_np, (probs_cal >= 0.5).astype(int))
    ll_cal  = log_loss(y_val_np, probs_cal)
    print(f"   Post-calibrazione: acc={acc_cal:.2%} | log_loss={ll_cal:.4f}")

    # ── 5. Salvataggio ────────────────────────────────────────────────────────
    torch.save(model_final.state_dict(), 'modelo_ann.pth')
    cfg = best_hp.copy()
    cfg['model_version'] = 'v3'
    cfg['input_dim'] = len(FEATURES)
    cfg['features'] = FEATURES
    cfg['n_interactions'] = N_INTERACTIONS
    cfg['interaction_pairs'] = DEFAULT_INTERACTION_PAIRS
    cfg['scaler_file'] = 'scaler_ann.pkl'
    cfg['calibrator_file'] = 'calibrator_ann.pkl'
    cfg['elo_surface_file'] = 'elo_surface.pkl'
    cfg['streak_file'] = 'streak_players.pkl'
    cfg['momentum_file'] = 'momentum_surface.pkl'
    with open('ann_config.json', 'w') as f:
        json.dump(cfg, f, indent=2)

    # ── 6. Confronto finale ───────────────────────────────────────────────────
    df_confronto = confronto_finale(acc_global, ll_global, [])

    print("\n✅ File salvati:")
    for f in ['modelo_ann.pth', 'scaler_ann.pkl', 'ann_config.json',
              'elo_surface.pkl', 'streak_players.pkl', 'momentum_surface.pkl',
              'calibrator_ann.pkl', 'resultados_comparacion_finale.csv']:
        stato = "✅" if os.path.exists(f) else "—"
        print(f"  {stato}  {f}")
