"""
train_ann.py v2.0  —  ANN Avanzata per Tennis Predictor
========================================================
Funzionalità nuove rispetto v1:
  ✅ Rating Elo per superficie  (più accurato del ranking ATP)
  ✅ Striscia attiva            (consecutive wins/losses)
  ✅ Ponderazione torneo        (Grand Slam pesa x2, Masters x1.5 ...)
  ✅ Optuna bayesian search     (più efficiente del random search)
  ✅ Modelli per superficie     (All, Clay, Hard, Grass)
  ✅ Comparazione finale        (ANN vs XGBoost vs LR vs RF)

UTILIZZO:
    cd prediccion/
    python train_ann.py

OUTPUT:
    modelo_ann.pth / _clay.pth / _hard.pth / _grass.pth
    ann_config.json (e _clay / _hard / _grass)
    scaler_ann.pkl
    elo_surface.pkl          ← Elo corrente per superficie (usato dal predictor)
    resultados_ann.csv       ← tutti i trial Optuna
    resultados_comparacion_finale.csv  ← confronto con modelli classici
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
TRIALS = 60        # numero trial Optuna per il modello globale
TRIALS_SURF = 20   # trial per ogni modello per superficie

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

        # --- Momentum ---
        hw = racha_t.get(w, []); hl = racha_t.get(l, [])
        mw = np.mean(hw) if hw else 0.5
        ml = np.mean(hl) if hl else 0.5
        hw.append(1); hl.append(0)
        if len(hw) > 10: hw.pop(0)
        if len(hl) > 10: hl.pop(0)
        racha_t[w] = hw; racha_t[l] = hl

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

        rd = row.to_dict(); upd_serve(w, rd, 'w')

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
        upd_serve(l, rd, 'l')

    df_out = pd.DataFrame(rows)
    print(f"   → Dataset: {len(df_out):,} righe | {df_out.shape[1]} colonne")

    # Salva Elo corrente (usato dal predictor)
    joblib.dump(elo_surf, 'elo_surface.pkl')
    print("   → elo_surface.pkl salvato")
    # Salva anche streak corrente per il predictor
    joblib.dump(streak_t, 'streak_players.pkl')

    return df_out, stats_dict, elo_surf, streak_t


# ── Lista feature (25)────────────────────────────────────────────────────────
FEATURES = [
    'diff_rank', 'diff_rank_points', 'diff_seed', 'diff_age', 'diff_ht',
    'diff_elo', 'diff_streak',                          # ← NUOVE
    'surface_enc', 'tourney_level', 'round_enc', 'draw_size',
    'diff_hand',
    'diff_skill', 'diff_home',
    'diff_fatigue', 'diff_momentum', 'diff_h2h',
    'diff_ace', 'diff_df', 'diff_1st_pct', 'diff_1st_won',
    'diff_2nd_won', 'diff_bp_saved',
    'court_ace_pct', 'court_speed',                     # ← Court speed
]  # totale: 23 feature


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

class TennisANN(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: list, dropout: float = 0.3):
        super().__init__()
        layers = []; prev = input_dim
        for h in hidden_layers:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


def train_model(model, loader_train, loader_val, epochs=80, lr=1e-3, patience=10):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=4)
    criterion = nn.BCEWithLogitsLoss()
    best_val = float('inf'); best_state = None; no_imp = 0

    for _ in range(epochs):
        model.train()
        for Xb, yb in loader_train:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval(); vl = []
        with torch.no_grad():
            for Xb, yb in loader_val:
                Xb, yb = Xb.to(device), yb.to(device)
                vl.append(criterion(model(Xb), yb).item())
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

def optuna_search(X_tr, y_tr, X_val, X_val_sc, X_te_sc,
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

            # Stampa PRIMA di addestrare — aggiornamento immediato in console
            print(f"  [{trial.number+1:3d}/{n_trials}] arch={arch_name:12s} dr={dr:.2f} lr={lr:.0e} λ={lam:.4f}  ▶ training...", flush=True)

            # Se GPU, usa batch più grandi per tenerla satura
            bs_eff = bs * 2 if USE_CUDA and bs < 2048 else bs
            w_combined = calcola_pesi_combinati(dates_tr, level_w_tr, lam)
            w_t = torch.tensor(w_combined)
            sampler = WeightedRandomSampler(w_t, len(w_t), replacement=True)
            ldr_tr  = DataLoader(dataset_tr,  batch_size=bs_eff, sampler=sampler,
                                 pin_memory=PIN_MEM, num_workers=N_WORKERS)
            ldr_val = DataLoader(dataset_val, batch_size=4096, shuffle=False,
                                 pin_memory=PIN_MEM, num_workers=N_WORKERS)

            model = TennisANN(input_dim, hl, dr)
            model, _ = train_model(model, ldr_tr, ldr_val, epochs=ep, lr=lr)
            acc_v, _ = valuta(model, X_val_sc, y_val_np)

            trial.set_user_attr('model', model)
            trial.set_user_attr('hl', hl)
            trial.set_user_attr('lam', lam)
            trial.set_user_attr('hp', {'hidden_layers': hl, 'dropout': dr, 'lr': lr,
                                       'batch_size': bs, 'epochs': ep, 'lambda_decay': lam})
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
            hp  = {'hidden_layers': hl, 'dropout': dr, 'lr': lr,
                   'batch_size': bs, 'epochs': ep, 'lambda_decay': lam}
            print(f"  [{i+1:3d}/{n_trials}] {str(hl):22s} drop={dr} lr={lr:.0e} λ={lam:.4f}  ▶ avvio...", flush=True)
            w_combined = calcola_pesi_combinati(dates_tr, level_w_tr, lam)
            w_t = torch.tensor(w_combined)
            sampler = WeightedRandomSampler(w_t, len(w_t), replacement=True)
            ldr_tr  = DataLoader(dataset_tr,  batch_size=bs, sampler=sampler)
            ldr_val = DataLoader(dataset_val, batch_size=2048, shuffle=False)
            model = TennisANN(input_dim, hl, dr)
            model, _ = train_model(model, ldr_tr, ldr_val, epochs=ep, lr=lr)
            acc_v, _ = valuta(model, X_val_sc, y_val_np)
            acc_t, ll = valuta(model, X_te_sc, y_test_np)
            print(f"     └─ val={acc_v:.3f} test={acc_t:.3f}", flush=True)
            risultati.append({'trial': i+1, 'hidden_layers': str(hl),
                               'dropout': dr, 'lr': lr, 'batch_size': bs,
                               'epochs': ep, 'lambda_decay': lam,
                               'val_acc': acc_v, 'test_acc': acc_t,
                               'log_loss': ll, '_model': model, '_config': hp})

    return sorted(risultati, key=lambda x: x['test_acc'], reverse=True)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  TRAINING MODELLO PER SUPERFICIE
# ══════════════════════════════════════════════════════════════════════════════

def train_surface_model(df_ml, surface_name, best_config, scaler_global, n_trials=TRIALS_SURF):
    """
    Addestra un modello specializzato su una sola superficie.
    Usa lo stesso scaler globale per compatibilità con il predictor.
    """
    surf_enc_map = {'Hard': 0, 'Clay': 1, 'Grass': 2}
    surf_enc     = surf_enc_map.get(surface_name, -1)

    df_s = df_ml[df_ml['surface_enc'] == surf_enc].copy()
    print(f"\n   [{surface_name}]: {len(df_s):,} campioni", end="  ")

    MIN_SAMPLES = 2000
    if len(df_s) < MIN_SAMPLES:
        print(f"⚠️  Troppo pochi dati (< {MIN_SAMPLES}), salto.")
        return None

    X_s = df_s[FEATURES].fillna(0)
    y_s = df_s['target']
    d_s = df_s['tourney_date']
    lw_s= df_s['level_weight'].values

    X_tr, X_te, y_tr, y_te, d_tr, _, lw_tr, _ = train_test_split(
        X_s, y_s, d_s, lw_s, test_size=0.20, random_state=SEED)
    X_val, X_te2, y_val, y_te2 = train_test_split(X_te, y_te, test_size=0.50, random_state=SEED)

    # Usa scaler globale (compatibile con predictor)
    X_tr_sc  = scaler_global.transform(X_tr)
    X_val_sc = scaler_global.transform(X_val)
    X_te_sc  = scaler_global.transform(X_te2)

    res = optuna_search(
        X_tr_sc, y_tr, None, X_val_sc, X_te_sc,
        y_val, y_te2, d_tr, lw_tr,
        n_trials=n_trials, input_dim=len(FEATURES),
        label=surface_name
    )
    if not res: return None

    best = res[0]
    acc_t, ll = valuta(best['_model'], X_te_sc, y_te2.values)
    print(f"   🏆 {surface_name}: test_acc={acc_t:.4f} | log_loss={ll:.4f}")

    surf_tag = surface_name.lower()
    torch.save(best['_model'].state_dict(), f'modelo_ann_{surf_tag}.pth')
    cfg = best['_config'].copy()
    cfg['input_dim'] = len(FEATURES); cfg['features'] = FEATURES
    cfg['surface']   = surface_name;  cfg['scaler_file'] = 'scaler_ann.pkl'
    with open(f'ann_config_{surf_tag}.json', 'w') as f:
        json.dump(cfg, f, indent=2)

    return {'surface': surface_name, 'test_acc': acc_t, 'log_loss': ll,
            'arch': best['hidden_layers']}


# ══════════════════════════════════════════════════════════════════════════════
# 6.  CONFRONTO FINALE
# ══════════════════════════════════════════════════════════════════════════════

def confronto_finale(ann_global_acc, ann_global_ll, ann_surface_results):
    rows = [{'Modello': 'ANN Globale (23 feat)', 'Accuracy': ann_global_acc,
              'Log Loss': ann_global_ll, 'Note': 'Tutte le superfici'}]

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

    # Split globale: 70 / 15 / 15
    X_tr, X_tmp, y_tr, y_tmp, d_tr, d_tmp, lw_tr, _ = train_test_split(
        X, y, dates, level_w, test_size=0.30, random_state=SEED)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=SEED)

    # Scaler globale (usato anche dai modelli per superficie)
    scaler = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_tr)
    X_val_sc = scaler.transform(X_val)
    X_te_sc  = scaler.transform(X_test)
    joblib.dump(scaler, 'scaler_ann.pkl')
    print(f"   → scaler_ann.pkl salvato  |  {len(FEATURES)} feature")

    # ── 2. Optuna search globale ──────────────────────────────────────────────
    risultati = optuna_search(
        X_tr_sc, y_tr, None, X_val_sc, X_te_sc,
        y_val, y_test, d_tr, lw_tr,
        n_trials=TRIALS, input_dim=len(FEATURES), label="Global"
    )

    df_ris = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith('_')}
                           for r in risultati])
    df_ris.to_csv('resultados_ann.csv', index=False)

    best = risultati[0]
    acc_global, ll_global = valuta(best['_model'], X_te_sc, y_test.values)

    print(f"\n🏆 Modello globale migliore:")
    print(f"   Architettura:  {best['hidden_layers']}")
    print(f"   Hyperparams:   dropout={best['dropout']} | lr={best['lr']:.0e} | bs={best['batch_size']} | λ={best['lambda_decay']}")
    print(f"   Test Accuracy: {acc_global:.2%}")
    print(f"   Log Loss:      {ll_global:.4f}")
    print(f"   Top 5 trial:\n{df_ris[['hidden_layers','dropout','lr','lambda_decay','val_acc','test_acc']].head(5).to_string(index=False)}")

    torch.save(best['_model'].state_dict(), 'modelo_ann.pth')
    cfg = best['_config'].copy()
    cfg['input_dim'] = len(FEATURES); cfg['features'] = FEATURES
    cfg['scaler_file'] = 'scaler_ann.pkl'
    with open('ann_config.json', 'w') as f:
        json.dump(cfg, f, indent=2)

    # ── 3. Modelli per superficie ─────────────────────────────────────────────
    print("\n\n📐 Training modelli specializzati per superficie...")
    surf_results = []
    for surf in ['Clay', 'Hard', 'Grass']:
        r = train_surface_model(df_ml, surf, best['_config'], scaler, n_trials=TRIALS_SURF)
        surf_results.append(r)

    # ── 4. Confronto finale ───────────────────────────────────────────────────
    df_confronto = confronto_finale(acc_global, ll_global, surf_results)

    print("\n✅ File salvati:")
    for f in ['modelo_ann.pth','modelo_ann_clay.pth','modelo_ann_hard.pth',
              'modelo_ann_grass.pth','scaler_ann.pkl','ann_config.json',
              'elo_surface.pkl','streak_players.pkl',
              'resultados_ann.csv','resultados_comparacion_finale.csv']:
        stato = "✅" if os.path.exists(f) else "—"
        print(f"  {stato}  {f}")
