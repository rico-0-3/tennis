"""
train_special_bets.py
=====================
Predizione esatta (Regressione) di statistiche specifiche:
 1. Totale Ace
 2. Totale Doppi Falli
 3. Totale Break

Approccio Avanzato (stile train_ann.py):
 - Feature Engineering: Rolling stats, Sum/Diff features.
 - Modelli: ANN (Wide & Deep), XGBoost, LightGBM.
 - Ottimizzazione: Optuna (o Random Search).
 - Ensemble: Stacking, Average, Top-5 ANN.
 - Metrica: MAE (Mean Absolute Error).
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_poisson_deviance
from scipy.stats import poisson

warnings.filterwarnings("ignore")

# ─── COURT SPEED HELPER ──────────────────────────────────────────────────────
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scraping'))
try:
    from court_speed_helper import get_court_stats
    HAS_COURT_SPEED = True
    print("✅ court_speed_helper importato con successo.")
except ImportError:
    HAS_COURT_SPEED = False
    print("⚠️ court_speed_helper non trovato.")
    def get_court_stats(name, surface, year): return 0.0, 0.5 # Default dummy

# ─── CONFIGURAZIONE ──────────────────────────────────────────────────────────
CSV_PATH = 'historialTenis.csv'
if not os.path.exists(CSV_PATH):
    CSV_PATH = os.path.join('..', 'scraping', 'historialTenis.csv')

TARGETS = {
    'total_aces':   {'name': 'Totale Ace'},
    'total_dfs':    {'name': 'Totale Doppi Falli'},
    'total_breaks': {'name': 'Totale Break'},
}

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Device: {device}")

TRIALS = 5        # Trial Optuna per ANN (come richiesto)
TRIALS_GBM = 20   # Trial Optuna per XGB/LGB (come richiesto)

BO3 = True        # Abilita/Disabilita Best of 3
BO5 = True        # Abilita/Disabilita Best of 5

# Gestione librerie opzionali
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("⚠️ Optuna non trovato. Usare 'pip install optuna'. Fallback su parametri fissi.")

# ─── HELPER FUNCTIONS ────────────────────────────────────────────────────────

def calculate_breaks(row):
    """Calcola totale break nel match."""
    try:
        w_faced = float(row.get('w_bpFaced', 0))
        w_saved = float(row.get('w_bpSaved', 0))
        l_faced = float(row.get('l_bpFaced', 0))
        l_saved = float(row.get('l_bpSaved', 0))
        if np.isnan(w_faced) or np.isnan(l_faced): return np.nan
        return (w_faced - w_saved) + (l_faced - l_saved)
    except:
        return np.nan

# ─── CARICAMENTO E FEATURE ENGINEERING ───────────────────────────────────────

def load_and_process(csv_path):
    print(f"📂 Caricamento dati da: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    df['tourney_date'] = pd.to_numeric(df['tourney_date'], errors='coerce')
    df = df.sort_values(['tourney_date', 'match_num']).reset_index(drop=True)
    
    # 1. Calcolo Target
    print("   Calcolo target (Ace, DF, Breaks)...")
    df['total_aces'] = df['w_ace'] + df['l_ace']
    df['total_dfs']  = df['w_df'] + df['l_df']
    df['total_breaks'] = df.apply(calculate_breaks, axis=1)
    
    df = df.dropna(subset=['total_aces', 'total_dfs', 'total_breaks'])
    print(f"   → {len(df)} match validi con statistiche complete.")

    # 2. Rolling Stats
    history = {}
    features_list = []
    
    print("   Generazione feature storiche (con Tie-Break, Pressione DF e Altezze)...")
    for idx, row in df.iterrows():
        w, l = row['winner_name'], row['loser_name']
        surf = row['surface']
        t_date = str(row['tourney_date'])
        
        def get_stats(p):
            if p not in history or len(history[p]['games_played']) < 5: 
                return None
            h = history[p]
            tot_sv_gms = sum(h['games_played']) or 1
            tot_ret_gms = sum(h['games_return']) or 1
            tot_2nd_ret = sum(h['ret_2nd_played']) or 1
            
            # Tassi offensivi
            ace_rate = sum(h['ace_for']) / tot_sv_gms
            df_rate = sum(h['df_for']) / tot_sv_gms
            bp_faced_rate = sum(h['bp_faced']) / tot_sv_gms
            
            # 🎾 NUOVO: Service Hold % (per calcolare probabilità Tie-Break)
            hold_pct = 1.0 - (sum(h['bp_lost']) / tot_sv_gms)
            
            # Tassi difensivi
            ace_allowed_rate = sum(h['ace_against']) / tot_ret_gms
            bp_created_rate = sum(h['bp_created']) / tot_ret_gms
            
            # 🎾 NUOVO: Pressione in risposta (Punti vinti sulla 2a dell'avversario)
            ret_2nd_win_pct = sum(h['ret_2nd_won']) / tot_2nd_ret

            return {
                'ace_rate': ace_rate, 'df_rate': df_rate, 'bp_faced_rate': bp_faced_rate, 'hold_pct': hold_pct,
                'ace_allowed_rate': ace_allowed_rate, 'bp_created_rate': bp_created_rate, 'ret_2nd_win_pct': ret_2nd_win_pct
            }

        s_w = get_stats(w)
        s_l = get_stats(l)
        
        if s_w and s_l:
            best_of = 5 if str(row.get('best_of')) == '5' else 3
            
            w_ht = float(row['winner_ht']) if pd.notna(row.get('winner_ht')) else 185
            l_ht = float(row['loser_ht']) if pd.notna(row.get('loser_ht')) else 185
            
            surface_code = {'Hard': 0, 'Clay': 1, 'Grass': 2, 'Carpet': 3}.get(surf, 0)
            year = int(t_date[:4]) if len(t_date) >= 4 else 2024
            
            # RECUPERIAMO ANCHE LA PERCENTUALE ACE DEL CAMPO
            court_ace, court_spd = get_court_stats(row.get('tourney_name', ''), surf, year)
            
            expected_games = 22 if best_of == 3 else 38
            
            w_exp_aces = ((s_w['ace_rate'] + s_l['ace_allowed_rate']) / 2.0) * (expected_games / 2)
            l_exp_aces = ((s_l['ace_rate'] + s_w['ace_allowed_rate']) / 2.0) * (expected_games / 2)
            w_exp_bps = ((s_w['bp_faced_rate'] + s_l['bp_created_rate']) / 2.0) * (expected_games / 2)
            l_exp_bps = ((s_l['bp_faced_rate'] + s_w['bp_created_rate']) / 2.0) * (expected_games / 2)

            feat = {
                'surface': surface_code,
                'court_speed': court_spd,
                'court_ace_pct': court_ace, # 🌟 AGGIUNTA QUI
                
                'best_of': best_of,
                'sum_ht': w_ht + l_ht,
                
                # 🌟 LE 3 NUOVE FEATURE 🌟
                'diff_ht': abs(w_ht - l_ht), # 1. Vantaggio Fisico al servizio
                'prob_tb': s_w['hold_pct'] * s_l['hold_pct'], # 2. Probabilità Tie-Break (moltiplicatore Ace)
                'avg_pressure': (s_w['ret_2nd_win_pct'] + s_l['ret_2nd_win_pct']) / 2.0, # 3. Pressione sulla 2a (moltiplicatore DF)
                
                'proj_total_aces': w_exp_aces + l_exp_aces,
                'proj_total_dfs': (s_w['df_rate'] + s_l['df_rate']) * (expected_games / 2),
                'proj_total_bps': w_exp_bps + l_exp_bps,
                
                'diff_ace_rate': abs(s_w['ace_rate'] - s_l['ace_rate']),
                'diff_def_rate': abs(s_w['ace_allowed_rate'] - s_l['ace_allowed_rate']),
                
                'total_aces': row['total_aces'],
                'total_dfs': row['total_dfs'],
                'total_breaks': row['total_breaks']
            }
            features_list.append(feat)
        
        # Aggiorna storico con i nuovi dati (Punti sulla 2a e Break subiti)
        try:
            w_sv_gms = float(row['w_SvGms']) if pd.notna(row.get('w_SvGms')) else 1.0
            l_sv_gms = float(row['l_SvGms']) if pd.notna(row.get('l_SvGms')) else 1.0
            
            # Calcolo Palle Break Perse
            w_bp_lost = max(0.0, float(row['w_bpFaced']) - float(row['w_bpSaved']))
            l_bp_lost = max(0.0, float(row['l_bpFaced']) - float(row['l_bpSaved']))
            
            # Calcolo Punti giocati e persi sulla 2a
            w_2nd_played = max(1.0, float(row['w_svpt']) - float(row['w_1stIn']))
            w_2nd_lost = w_2nd_played - float(row['w_2ndWon']) # Vinti dal ricevitore (L)
            
            l_2nd_played = max(1.0, float(row['l_svpt']) - float(row['l_1stIn']))
            l_2nd_lost = l_2nd_played - float(row['l_2ndWon']) # Vinti dal ricevitore (W)

            def update(p, ace_f, df_f, bp_f, bp_l, gms_sv, ace_a, bp_c, gms_ret, ret_2nd_w, ret_2nd_p):
                if p not in history:
                    history[p] = {'ace_for':[], 'df_for':[], 'bp_faced':[], 'bp_lost':[], 'games_played':[],
                                  'ace_against':[], 'bp_created':[], 'games_return':[], 'ret_2nd_won':[], 'ret_2nd_played':[]}
                h = history[p]
                h['ace_for'].append(ace_f); h['df_for'].append(df_f); h['bp_faced'].append(bp_f); h['bp_lost'].append(bp_l)
                h['games_played'].append(gms_sv)
                
                h['ace_against'].append(ace_a); h['bp_created'].append(bp_c)
                h['games_return'].append(gms_ret); h['ret_2nd_won'].append(ret_2nd_w); h['ret_2nd_played'].append(ret_2nd_p)
                
                if len(h['ace_for']) > 30: 
                    for k in h: h[k].pop(0)

            # W riceve il servizio di L (quindi W vince i l_2nd_lost)
            update(w, float(row['w_ace']), float(row['w_df']), float(row['w_bpFaced']), w_bp_lost, w_sv_gms,
                      float(row['l_ace']), float(row['l_bpFaced']), l_sv_gms, l_2nd_lost, l_2nd_played)
                      
            # L riceve il servizio di W (quindi L vince i w_2nd_lost)
            update(l, float(row['l_ace']), float(row['l_df']), float(row['l_bpFaced']), l_bp_lost, l_sv_gms,
                      float(row['w_ace']), float(row['w_bpFaced']), w_sv_gms, w_2nd_lost, w_2nd_played)
        except:
            continue  
                          
    return pd.DataFrame(features_list), history

# ─── MODELLO ANN (Regressione) ───────────────────────────────────────────────

class TennisANN_Reg(nn.Module):
    """Wide & Deep ANN per Regressione."""
    def __init__(self, input_dim, hidden_layers=[64, 32], dropout=0.3):
        super().__init__()
        self.wide = nn.Linear(input_dim, 1)
        
        layers = []
        in_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.Dropout(dropout))
            in_dim = h
        self.deep = nn.Sequential(*layers)
        self.deep_out = nn.Linear(in_dim, 1)
        
    def forward(self, x):
        wide_out = self.wide(x)
        deep_out = self.deep_out(self.deep(x))
        return wide_out + deep_out

def train_ann_model(model, loader_tr, loader_val, epochs=80, lr=0.001): # Epochs aumentate a 80
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss() # Ottimizziamo MSE, ma valutiamo MAE
    best_val_loss = float('inf')
    best_state = None
    patience = 7
    no_imp = 0
    
    for ep in range(epochs):
        model.train()
        for Xb, yb in loader_tr:
            Xb, yb = Xb.to(device), yb.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            
        model.eval()
        val_losses = []
        with torch.no_grad():
            for Xb, yb in loader_val:
                Xb, yb = Xb.to(device), yb.to(device).float().unsqueeze(1)
                pred = model(Xb)
                val_losses.append(mean_absolute_error(yb.cpu(), pred.cpu())) # Monitoriamo MAE
        
        avg_val = np.mean(val_losses)
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = model.state_dict()
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience: break
            
    if best_state: model.load_state_dict(best_state)
    return model, best_val_loss

# ─── OPTIMIZATION & TRAINING ─────────────────────────────────────────────────

def optimize_ann(X_tr, y_tr, X_val, y_val, n_trials=TRIALS):
    print(f"   🧠 Ottimizzazione ANN ({n_trials} trials, max 80 epochs)...")
    
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr.values, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val.values, dtype=torch.float32)
    
    ds_tr = TensorDataset(X_tr_t, y_tr_t)
    ds_val = TensorDataset(X_val_t, y_val_t)
    
    results = []
    
    def objective(trial):
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        n_layers = trial.suggest_int('n_layers', 1, 3)
        layers = []
        for i in range(n_layers):
            layers.append(trial.suggest_int(f'l{i}', 32, 256))
        bs = trial.suggest_categorical('bs', [64, 128, 256])
        
        loader_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True)
        loader_val = DataLoader(ds_val, batch_size=1024)
        
        model = TennisANN_Reg(X_tr.shape[1], layers, dropout)
        model, val_mae = train_ann_model(model, loader_tr, loader_val, epochs=80, lr=lr) # Epochs 80
        
        trial.set_user_attr('model', model)
        return val_mae

    if HAS_OPTUNA:
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # Raccogli i migliori modelli per l'ensemble
        for t in study.trials:
            if t.value is not None and 'model' in t.user_attrs:
                results.append({'model': t.user_attrs['model'], 'mae': t.value})
    else:
        # Fallback manuale
        model = TennisANN_Reg(X_tr.shape[1], [128, 64], 0.3)
        loader_tr = DataLoader(ds_tr, batch_size=128, shuffle=True)
        loader_val = DataLoader(ds_val, batch_size=1024)
        model, val_mae = train_ann_model(model, loader_tr, loader_val)
        results.append({'model': model, 'mae': val_mae})
        
    results.sort(key=lambda x: x['mae'])
    return results

def optimize_xgb(X_tr, y_tr, X_val, y_val, n_trials=TRIALS_GBM):
    if not HAS_XGB: return None
    print(f"   🌲 Ottimizzazione XGBoost ({n_trials} trials)...")
    
    def objective(trial):
        params = {
                    'objective': 'reg:tweedie',           # Abbandoniamo Poisson per Tweedie
                    'tweedie_variance_power': 1.5,        # 1.5 è perfetto per conteggi sovrdispersi
                    'n_estimators': trial.suggest_int('n_estimators', 100, 800),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                    'max_depth': trial.suggest_int('max_depth', 3, 8),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'n_jobs': -1,
                    'random_state': SEED
                }
        model = xgb.XGBRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict(X_val)
        return mean_absolute_error(y_val, preds)

    if HAS_OPTUNA:
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        best_params = study.best_params
    else:
        best_params = {'n_estimators': 300, 'learning_rate': 0.05, 'max_depth': 5}
        
    model = xgb.XGBRegressor(**best_params, n_jobs=-1, random_state=SEED)
    model.fit(X_tr, y_tr)
    return model

def optimize_lgb(X_tr, y_tr, X_val, y_val, n_trials=TRIALS_GBM):
    if not HAS_LGB: return None
    print(f"   🍃 Ottimizzazione LightGBM ({n_trials} trials)...")
    
    def objective(trial):
        params = {
                    'objective': 'tweedie',
                    'tweedie_variance_power': 1.5,
                    'n_estimators': trial.suggest_int('n_estimators', 100, 800),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                    'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'n_jobs': -1,
                    'random_state': SEED,
                    'verbose': -1
                }
        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.log_evaluation(0)])
        preds = model.predict(X_val)
        return mean_absolute_error(y_val, preds)

    if HAS_OPTUNA:
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        best_params = study.best_params
    else:
        best_params = {'n_estimators': 300, 'learning_rate': 0.05, 'num_leaves': 31}
        
    model = lgb.LGBMRegressor(**best_params, n_jobs=-1, random_state=SEED, verbose=-1)
    model.fit(X_tr, y_tr)
    return model

# ─── ENSEMBLE & EVALUATION ───────────────────────────────────────────────────

def get_preds(model, X, model_type='sklearn'):
    if model_type == 'torch':
        model.eval()
        with torch.no_grad():
            t_x = torch.tensor(X, dtype=torch.float32).to(device)
            return model(t_x).cpu().numpy().flatten()
    else:
        return model.predict(X)

def train_and_evaluate_target(df, target_col, target_name):
    print(f"\n{'='*60}")
    print(f"🎯 {target_name.upper()}")
    print(f"{'='*60}")
    
    # Prepare Data
    drop_cols = list(TARGETS.keys()) # Rimuovi tutti i target dalle feature
    X = df.drop(columns=drop_cols, errors='ignore')
    y = df[target_col]
    
    # Split
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.15, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=SEED)
    
    # Scaling
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)
    X_test_sc = scaler.transform(X_test)
    
    # 1. Train ANN
    ann_results = optimize_ann(X_train_sc, y_train, X_val_sc, y_val)
    best_ann = ann_results[0]['model']
    
    # 2. Train GBMs
    xgb_model = optimize_xgb(X_train_sc, y_train, X_val_sc, y_val)
    lgb_model = optimize_lgb(X_train_sc, y_train, X_val_sc, y_val)
    
    # 3. Generate Predictions for Stacking (on Val set)
    preds_val = {}
    preds_test = {}
    
    # ANN Best
    preds_val['ANN'] = get_preds(best_ann, X_val_sc, 'torch')
    preds_test['ANN'] = get_preds(best_ann, X_test_sc, 'torch')
    
    # ANN Top 5 Avg
    top5_anns = [r['model'] for r in ann_results[:5]]
    p_val_list = [get_preds(m, X_val_sc, 'torch') for m in top5_anns]
    p_test_list = [get_preds(m, X_test_sc, 'torch') for m in top5_anns]
    preds_val['ANN5'] = np.mean(p_val_list, axis=0)
    preds_test['ANN5'] = np.mean(p_test_list, axis=0)
    
    # XGB
    if xgb_model:
        preds_val['XGB'] = get_preds(xgb_model, X_val_sc)
        preds_test['XGB'] = get_preds(xgb_model, X_test_sc)
        
    # LGB
    if lgb_model:
        preds_val['LGB'] = get_preds(lgb_model, X_val_sc)
        preds_test['LGB'] = get_preds(lgb_model, X_test_sc)
        
    # 4. Strategies Evaluation
    strategies = {}
    
    # Single Models
    strategies['ANN Best'] = preds_test['ANN']
    if xgb_model: strategies['XGBoost'] = preds_test['XGB']
    if lgb_model: strategies['LightGBM'] = preds_test['LGB']
    
    # Ensembles
    # Avg (ANN5 + XGB + LGB)
    components = ['ANN5']
    if xgb_model: components.append('XGB')
    if lgb_model: components.append('LGB')
    
    avg_pred = np.mean([preds_test[k] for k in components], axis=0)
    strategies['Ensemble Avg'] = avg_pred
    
    # Stacking (Ridge Regression on Val preds)
    X_stack_val = np.column_stack([preds_val[k] for k in components])
    X_stack_test = np.column_stack([preds_test[k] for k in components])
    
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(X_stack_val, y_val)
    stack_pred = meta_model.predict(X_stack_test)
    strategies['Stacking'] = stack_pred
    
    # 5. Compare Results (Poisson Deviance & MAE)
    print("\n📊 RISULTATI (Valutazione Poisson):")
    best_dev = float('inf')
    best_strat = ""
    
    results_summary = []
    
    for name, preds in strategies.items():
        # La devianza di Poisson esplode se c'è un numero <= 0. 
        # I GBM in Poisson escono positivi, ma il Ridge dello stacking o l'ANN potrebbero sbagliare.
        preds_safe = np.clip(preds, 1e-4, None) 
        
        mae = mean_absolute_error(y_test, preds_safe)
        
        try:
            dev = mean_poisson_deviance(y_test, preds_safe)
        except ValueError:
            dev = float('inf') # Se qualcosa va storto coi dati
            
        print(f"   🔹 {name:<15} | MAE: {mae:.3f} | Devianza Poisson: {dev:.3f}")
        
        # Scegliamo il modello migliore in base alla Devianza, non al MAE!
        if dev < best_dev:
            best_dev = dev
            best_strat = name
            
        results_summary.append({'strategy': name, 'mae': mae, 'dev': dev})
            
    print(f"\n🏆 MIGLIORE STRATEGIA: {best_strat} (Devianza: {best_dev:.3f})")
    
    idx_ex = 0
    real = y_test.iloc[idx_ex]
    pred = strategies[best_strat][idx_ex]
    print(f"   Esempio: Reale={real:.1f} vs Lambda Predetto={pred:.1f}")
    
    # Mappiamo i nomi delle strategie ai modelli reali
    modelli_singoli = {
        'ANN Best': best_ann,
        'XGBoost': xgb_model,
        'LightGBM': lgb_model
    }
    
    # Prepariamo il pacchetto da esportare in base a chi ha vinto
    if best_strat in modelli_singoli:
        obj_to_save = {'type': 'single', 'model': modelli_singoli[best_strat]}
    elif best_strat == 'Ensemble Avg':
        obj_to_save = {'type': 'avg', 'models': [m for m in [best_ann, xgb_model, lgb_model] if m is not None]}
    elif best_strat == 'Stacking':
        obj_to_save = {'type': 'stacking', 'models': [m for m in [best_ann, xgb_model, lgb_model] if m is not None], 'meta': meta_model}
    else:
        obj_to_save = {'type': 'single', 'model': best_ann} # Fallback sicurezza
        
    obj_to_save['scaler'] = scaler # Salva lo scaler specifico per questo target!
    
    obj_to_save['mae'] = [r['mae'] for r in results_summary if r['strategy'] == best_strat][0]
    obj_to_save['devianza'] = best_dev
    obj_to_save['avg_value'] = y_test.mean()
    # ------------------------------------------------------------------

    return {
        'target': target_name,
        'best_strategy': best_strat,
        'mae': [r['mae'] for r in results_summary if r['strategy'] == best_strat][0],
        'devianza': best_dev,
        'avg_value': y_test.mean(),
        'model_export': obj_to_save  # <--- Aggiungi questa riga al return!
    }


# ─── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not os.path.exists(CSV_PATH):
        print(f"❌ Errore: File {CSV_PATH} non trovato.")
        sys.exit(1)
        
    # 1. Load Data
    df_ml, history = load_and_process(CSV_PATH)
        
    if df_ml.empty:
        print("❌ Nessun dato valido estratto.")
        sys.exit(1)
        
    print(f"\nDataset pronto: {len(df_ml)} righe, {len(df_ml.columns)} colonne.")
    
    final_report = []
    
    # 2. Split Best of 3 vs Best of 5 (Cruciale per i totali)
    df_bo3 = df_ml[df_ml['best_of'] == 3].copy()
    df_bo5 = df_ml[df_ml['best_of'] == 5].copy()
    
    datasets = [('Best of 3', df_bo3), ('Best of 5', df_bo5)]
    
    for ds_name, df_curr in datasets:
        if ds_name == 'Best of 3' and not BO3:
                print(f"\n⚠️ Salto {ds_name}: disabilitato per configurazione.")
                continue    
                
        if ds_name == 'Best of 5' and not BO5:
                print(f"\n⚠️ Salto {ds_name}: disabilitato per configurazione.")
                continue
                
        if len(df_curr) < 100:
            print(f"\n⚠️ Salto {ds_name}: troppi pochi dati ({len(df_curr)}).")
            continue
            
        print(f"\n\n{'#'*60}")
        print(f"📂 DATASET: {ds_name.upper()} ({len(df_curr)} match)")
        print(f"{'#'*60}")
        
        for col, info in TARGETS.items():
            # Passiamo un nome composto per il report finale
            full_name = f"{info['name']} ({ds_name})"
            res = train_and_evaluate_target(df_curr, col, full_name)
            final_report.append(res)
        
    # 3. Riepilogo Finale
    print(f"\n{'='*80}")
    print("🏆 CLASSIFICA FINALE DEI MODELLI (POISSON EVALUATION)")
    print(f"{'='*80}")
    print(f"{'Statistica':<30} | {'Miglior Modello':<15} | {'MAE':<8} | {'Devianza':<8}")
    print("-" * 80)
    
    for r in final_report:
        print(f"{r['target']:<30} | {r['best_strategy']:<15} | {r['mae']:.2f}     | {r['devianza']:.2f}")
        
    print("-" * 80)
    print("💡 NOTA SULLE SCOMMESSE:")
    print("   Non esiste un mercato 'più facile' in assoluto. Il MAE ti dice quanti eventi sbagli in media,")
    print("   mentre la Devianza misura la bontà della distribuzione. Ora i tuoi modelli prevedono il valore")
    print("   atteso (Lambda). Inserisci questo Lambda nella formula di Poisson per trovare le VERE probabilità!")
    
    # --- DA INSERIRE ALLA FINE ASSOLUTA DI train_special_bets.py ---
    print("\n" + "="*80)
    print("💾 SALVATAGGIO MODELLI AUTOMATICO IN CORSO...")
    
    # Creiamo un dizionario dinamico con tutti i vincitori
    modelli_vincitori = {}
    for r in final_report:
        modelli_vincitori[r['target']] = r['model_export']
        
    export_data = {
        'history': history,
        'models': modelli_vincitori # Ora contiene tutto: Ace(Bo3), Break(Bo5), ecc.
    }
    
    save_path = os.path.join('..', 'prediccion', 'modelos_special_bets.pkl')
    # Se la cartella non esiste, salvalo nella dir corrente come fallback
    if not os.path.exists(os.path.dirname(save_path)): save_path = 'modelos_special_bets.pkl'
        
    joblib.dump(export_data, save_path)
    print(f"✅ Modelli per Scommesse Speciali esportati con successo in: {save_path}")
    print("="*80)

