# Tennis Predictor AI — Documentazione Tecnica

## Struttura del Progetto

```
tennis/
├── aggiorna_tutto.py            # Pipeline sequenziale principale (orchestrator)
├── run_pipeline.py              # CLI per GitHub Actions (--scraping, --ann, etc.)
├── main.py                      # Home page Streamlit (0_Inicio)
├── requirements.txt
├── .github/workflows/update_pipeline.yml  # CI/CD via GitHub Actions
├── scraping/                    # Moduli estrazione dati
├── prediccion/                  # ML, training, predizione
├── pages/                       # Dashboard Streamlit multi-pagina
└── assets/style.css
```

---

## Pipeline Completa (aggiorna_tutto.py)

Eseguire da root: `python aggiorna_tutto.py` oppure `AGGIORNA.bat`

Ordine delle fasi:

| Fase | Script | Output |
|------|--------|--------|
| 1 | `scraper_ranking.py` | `scraping/ranking_2026.csv` |
| 2 | `download_data.py` | `scraping/master_dataset.csv` |
| 3 | `scraper_bio_jugadores.py` | `scraping/bio_jugadores.json` |
| 4 | `generar_perfiles.py` | `scraping/perfiles_jugadores.pkl` |
| 5 | `scraper_court_speed.py` + `enriquecer_court_speed.py` | `scraping/court_speed_dict.pkl` |
| 6 | `scraper_proximas_partidas.py` | `scraping/proximas_partidas.json` |
| 7 | `train_ann.py` | `prediccion/modelo_finale.pkl` + ausiliari |
| 8 | `train_special_bet.py` | `prediccion/modelos_special_bets.pkl` |
| 9 | `predecir_proximas.py` | `prediccion/predicciones_proximas.json` |

Il `master_dataset.csv` viene filtrato prima del training: rimossi W/O, RET, DEF, BYE.

`run_pipeline.py` è identico ma accetta flag CLI (`--scraping`, `--profili`, `--ann`, ecc.) per GitHub Actions.

---

## Scraping (scraping/)

### scraper_ranking.py
- **Fonte**: atptour.com (undetected-chrome anti-bot)
- **Output**: `ranking_2026.csv` (rank, player, points, slug)
- Retry automatico 3 tentativi; rimozione cookie banner via JS

### download_data.py
- **Fonte**: TennisMyLife API (`stats.tennismylife.org/api/data-files`)
- **Output**: `master_dataset.csv` (~150k righe, 1991–2026)
- Incrementale: ri-scarica solo anno corrente e precedente, skippa anni storici
- `is_atp_main()`: filtra solo ATP main tour (scarta challenger, ongoing)

### generar_perfiles.py
- **Input**: master_dataset.csv + ranking_2026.csv + bio_jugadores.json
- **Output**: `perfiles_jugadores.pkl` — `dict[player_name → dict]`
- Struttura profilo:
  ```python
  {
    "rank": 2, "points": 8234, "age": 22.7, "ht": 183, "ioc": "ESP",
    "momentum": 0.65, "last_5": [1,1,1,0,1],
    "aces": 8.2, "df": 2.1, "serve_win": 68,
    "bp_saved": 62, "service_hold": 85
  }
  ```

### scraper_proximas_partidas.py
- **Fonte**: tennisexplorer.com
- **Output**: `scraping/proximas_partidas.json` — lista match upcoming con player1, player2, superficie, torneo, turno

### scraper_court_speed.py
- **Fonte**: tennisabstract.com surface-speed CGI
- **Output**: `court_speed_dict.pkl` — `dict[torneo → {"ace_pct": %, "speed_idx": float}]`
- Selenium headless, incrementale per anno

---

## Training ANN (prediccion/train_ann.py)

Lancio: `cd prediccion/ && python train_ann.py`

### Dataset split (TEMPORALE — no random)
- Dataset completo: 1991–2026, ma ML usa solo `ML_MIN_YEAR = 2015`
- Glicko-2 processa tutto lo storico (1991+) per avere rating stabili
- Split cronologico: Train → Val → Test per data torneo
- Ogni partita genera **due righe** (mirror pairs):
  - `d1`: feature vincitore, `target=1`
  - `d0`: feature negate (tranne `_SYMM_KEYS`: surface, level, round, is_best_of_5, indoor, level_weight), `target=0`
- Totale ~60k righe da ~30k partite reali

### Feature Engineering (running, leakage-safe)
Per ogni match, le statistiche vengono lette **prima** di aggiornare il dict post-match:

- **Glicko-2 per superficie**: `g2_hard`, `g2_clay`, `g2_grass`, `g2_gen` — aggiornati cronologicamente
- **H2H storico**: `h2h_t[(p1,p2)]` e `h2h_surface_t[(p1,p2,surf)]`
- **Recent form**: `recent_form_t[player]` — ultimi 10 risultati globali
- **Streak**: `streak_t[player]` — striscia attiva (+1 vittoria, -1 sconfitta)
- **Momentum superficie**: `racha_t[(player,surf)]` — ultimi 5 su questa superficie
- **Serve stats**: `serve_t[player]` = `{'ace': [...], '1st_pct': [...], '1st_won': [...], '2nd_won': [...], 'bp_saved': [...]}`
- **Return stats**: `return_t[player]` = `{'return_pct': [...], 'bp_conv': [...], 'return_1st': [...]}`
- **Fatica**: `match_load_t[player]` — match negli ultimi 56 giorni
- **Opponent quality**: ultimi match pesati per rank avversario
- **Upset tendency**: `upset_hist_t[player]` — frequenza sconfitte vs rank peggiori
- **Late round WR**: `late_round_t[player]` — win% in QF/SF/F

### ANN_FEATURES (32 feature v6.0)
```python
[
  'log_rank_ratio', 'log_pts_ratio',
  'diff_glicko', 'diff_glicko_overall', 'diff_glicko_rd',
  'diff_streak', 'diff_recent_form',
  'surface_enc', 'tourney_level', 'round_enc',
  'is_best_of_5', 'indoor',
  'diff_h2h', 'diff_h2h_surface',
  'diff_skill', 'diff_momentum',
  'diff_fatigue', 'diff_days_since_last', 'diff_weeks_load',
  'diff_ace', 'diff_1st_pct', 'diff_1st_won', 'diff_2nd_won', 'diff_bp_saved',
  'diff_return_pct', 'diff_bp_conv', 'diff_return_1st',
  'diff_home', 'diff_opponent_quality', 'diff_upset_tendency', 'diff_late_round_wr',
  'level_weight',
]
```

**v6 rispetto a v5.1**: rimossi `diff_elo`/`diff_elo_overall`, aggiunti `diff_glicko_rd` e `indoor`. ELO sostituito con Glicko-2.

### Architettura (TennisANNv3 — Wide & Deep + Residual)
- **Wide**: `Linear(input_dim → 1)` — cattura effetti lineari
- **Deep**: stack di layer `Linear → BatchNorm → ReLU → Dropout` con residual connections
- **Feature interactions**: 5 coppie moltiplicative — `(0,1)`, `(0,15)`, `(2,7)`, `(5,12)`, `(14,15)`
- **Output**: `logit → sigmoid`
- Label smoothing BCE loss (smoothing=0.05)

### Hyperparameter search (Optuna)
- TRIALS=100, MAX_EPOCHS=120
- Adam + exponential LR decay
- WeightedRandomSampler per bilanciamento classi
- Early stopping: no miglioramento su 10 epoche consecutive

### Calibrazione (Temperature Scaling)
Sostituisce la vecchia isotonica (v5.x). Usa un singolo parametro `T`:
```python
p_cal = sigmoid(logit(p_raw) / T)
```
- Fittatato su **validation set** (no leakage su test)
- `T > 1`: rete sovraconfidente → abbassa le probabilità estreme
- `T < 1`: rete sottoconfidente → alza
- **Strettamente monotona**: input diversi → output sempre diversi (risolve il problema delle probabilità identiche dell'isotonica)
- Implementata in `TemperatureScaler` (`prediction_engine.py`) con API `.predict()` compatibile con `IsotonicRegression`

### Output del training (modelo_finale.pkl)
```python
{
  'strategy': 'ann_best',           # Strategia predizione
  'model_name': 'ANN v6 ...',
  'ann': {'state_dict': ..., 'config': {...}},
  'ann_top5_uncertainty': [...],     # Top-5 arch per uncertainty
  'scaler': StandardScaler(),
  'calibrator': TemperatureScaler(T=...),
  'lgb_surface_models': {'Hard': ..., 'Clay': ..., 'Grass': ...},
  'ann_surface_models': {'Hard': ..., 'Clay': ...},  # Grass → fallback a global
  'accuracy': 0.7353,
  'surface_accuracy': {'Hard': 0.745, 'Clay': 0.721},  # Solo surf con ≥10 campioni
  'trained_date': '...',
  'calibration_curve': {'fraction_pos': [...], 'mean_pred': [...]},
}
```

I pkl ausiliari (h2h, glicko2, serve/return, etc.) vengono salvati **separatamente** nella cartella `prediccion/`.

---

## Predictor Engine (prediccion/prediction_engine.py)

Modulo **condiviso** importato sia dalla UI Streamlit che da `predecir_proximas.py`. Qualsiasi modifica alla logica di predizione va fatta qui.

### Classi e funzioni principali

**`TemperatureScaler`** — calibrazione, API `.predict(array) → array`

**`predici(input_sc, input_t, modelo_finale)`** — seleziona strategia da `modelo_finale['strategy']`:
- `ann_best`: singola ANN migliore
- `ann_top5` / `ann_top5_uncertainty`: media ensemble top-5
- `ensemble_avg`: media ANN + LGB + XGB
- `ensemble_stacking`: meta-learner LogisticRegression
- `lgb_surface`: LGB specifico per superficie
- `ann_surface`: ANN specifica per Hard/Clay, fallback globale per Grass

**`predici_con_cal(input_sc, input_t, modelo_finale)`** — entry point unificato:
1. Chiama `predici()` → prob raw
2. Applica top-5 ensemble (se disponibile) → calcola `ann_std`
3. Applica `_apply_cal()` → prob calibrata
4. Ritorna `(prob_calibrata, nome_modello, ann_std)`

**`_apply_cal(cal, p)`** — gestisce `TemperatureScaler`, `IsotonicRegression`, `LogisticRegression`

**Funzioni utility**:
- `days_since_last(date_int)` — YYYYMMDD → giorni, cap 180
- `weeks_load(match_dates, ref_date)` — match ultimi 56 giorni
- `calc_oq(match_list)` — opponent quality score (pesato per rank)
- `upset_tendency(hist)` — frequenza sconfitte vs rank peggiori (ultimi 20)

### Surface routing (ann_surface)
```python
surf_map = {0: 'Hard', 1: 'Clay', 2: 'Grass'}
surf_name = surf_map.get(surface_enc)
if surf_name in ann_surface_models:
    # usa modello specifico
else:
    # fallback a modelo_finale['ann'] (globale)
```
Grass usa sempre il modello globale (pochi dati per training dedicato).

---

## Predizione Prossime Partite (prediccion/predecir_proximas.py)

Lancio: `python predecir_proximas.py` (da `prediccion/`)

### Flusso
1. Carica: `modelo_finale.pkl`, `perfiles_jugadores.pkl`, `glicko2_stores.pkl`, `serve_stats.pkl`, `return_stats.pkl`
2. Legge `proximas_partidas.json`
3. Per ogni match:
   - Risolve nomi giocatori (cognome+iniziale → nome completo via fuzzy match su perfiles)
   - `build_features()` → DataFrame con 32 feature
   - `scaler.transform(df[ANN_FEATURES].values)` → numpy array (`.values` obbligatorio per evitare `_check_feature_names` di sklearn)
   - `predici()` + calibrazione
   - Ensemble top-5 (media + std)
   - Quote bookmaker: `1 / (prob * BK_OVERROUND)` con `BK_OVERROUND = 2/1.85`
4. Salva `predicciones_proximas.json`

### build_features() — Glicko-2 lookup
```python
g2_s   = glicko2_stores.get(superficie, glicko2_stores.get('Gen', Glicko2Store()))
g2_gen = glicko2_stores.get('Gen', Glicko2Store())
g2_r1,  g2_rd1,  _ = g2_s.get(nombre1)   # rating, rd, volatility
g2_r2,  g2_rd2,  _ = g2_s.get(nombre2)
g2_ov1, _, _       = g2_gen.get(nombre1)  # overall rating
g2_ov2, _, _       = g2_gen.get(nombre2)
```

Serve/return stats da pkl:
```python
def _gs(player, key, default):
    vals = serve_stats.get(player, {}).get(key, [])
    return float(np.mean(vals)) if vals else default
```

---

## Dashboard Streamlit (pages/)

### 1_🔮_Predictor_en_Vivo.py
Match predictor interattivo. Configurazione in sidebar:
- Superficie, Torneo, Paese, Turno, Livello, Format (3/5 set)

Feature caricate da pkl all'avvio (`cargar_todo()`):
- `modelo_finale.pkl`, `perfiles_jugadores.pkl`, `glicko2_stores.pkl`
- `serve_stats.pkl`, `return_stats.pkl`, `court_speed_dict.pkl`
- `h2h_surface.pkl`, `momentum_surface.pkl`, `upset_hist.pkl`, `late_round_hist.pkl`
- `opp_quality.pkl`

Output predizione: probabilità P1/P2, H2H, grafico radar, MetaConfidenceScore, uncertainty (std top-5), quote.

**Sidebar info**: mostra `surface_accuracy` del modello (solo superfici con dati nel test set — Grass può essere assente in certi periodi).

**Bottone**: "PREDICI con ANN v6"

### 2_📊_Analisis_y_Métricas.py
Confronto modelli (bar chart accuratezze) + feature importance da XGBoost.

### 3_🏆_Ranking_y_Perfiles.py
Tabella ranking ATP + profilo singolo giocatore (stats da `perfiles_jugadores.pkl`).

### 4_🔄_Aggiorna_Dati.py
Control panel GitHub Actions. Password: `Tennis2026`.
Poll ogni 10s, timeout 360 poll (~60 min). API: `POST /repos/.../actions/workflows/.../dispatches`.

---

## File PKL — Mappa Completa

Tutti in `prediccion/` salvo diversamente indicato.

| File | Contenuto |
|------|-----------|
| `modelo_finale.pkl` | ANN v6 state_dict + config + scaler + calibrator + top-5 + lgb/xgb/ann surface models |
| `scaler_ann.pkl` | `StandardScaler` fittato su train set |
| `calibrator_ann.pkl` | `TemperatureScaler(T=...)` |
| `glicko2_stores.pkl` | `dict[surface → Glicko2Store]` chiavi: `Hard`, `Clay`, `Grass`, `Gen` |
| `h2h_surface.pkl` | `dict[(p1,p2,surf) → [wins_p1, wins_p2]]` |
| `momentum_surface.pkl` | `dict[(player,surf) → [last 5 results]]` |
| `recent_form.pkl` | `dict[player → [last 10 results]]` |
| `serve_stats.pkl` | `dict[player → {ace:[...], 1st_pct:[...], 1st_won:[...], 2nd_won:[...], bp_saved:[...]}]` |
| `return_stats.pkl` | `dict[player → {return_pct:[...], bp_conv:[...], return_1st:[...]}]` |
| `streak_players.pkl` | `dict[player → streak_value]` |
| `last_match_date.pkl` | `dict[player → YYYYMMDD]` |
| `match_load.pkl` | `dict[player → [match_dates]]` |
| `opp_quality.pkl` | `dict[player → [(result, rank_opp), ...]]` |
| `upset_hist.pkl` | `dict[player → [(result, rank_diff), ...]]` |
| `late_round_hist.pkl` | `dict[player → [results_in_QF_SF_F]]` |
| `close_match_hist.pkl` | `dict[player → [3set_match_results]]` |
| `modelos_special_bets.pkl` | Modelli Poisson/Tweedie per Ace, DF, Break |
| `modelo_lgb.pkl` | `LGBMClassifier` standalone |
| `modelo_xgb.pkl` | `XGBClassifier` standalone |
| `modelo_meta_lr.pkl` | `LogisticRegression` per ensemble stacking |
| `scraping/perfiles_jugadores.pkl` | `dict[player → {rank, points, age, ht, ioc, momentum, last_5, aces, df, ...}]` |
| `scraping/court_speed_dict.pkl` | `dict[torneo → {ace_pct, speed_idx}]` |

---

## GitHub Actions (.github/workflows/update_pipeline.yml)

- **Trigger**: `workflow_dispatch` da Streamlit (POST API con input booleani per ogni fase)
- **Ambiente**: Ubuntu latest, Python 3.11
- **Timeout**: 180 min
- **Steps**: checkout → pip install → `run_pipeline.py` con flag → commit + push (modifica timestamp in requirements.txt per cache busting Streamlit Cloud)

---

## Costanti Globali Importanti

| Costante | Valore | File |
|----------|--------|------|
| `ML_MIN_YEAR` | 2015 | train_ann.py |
| `BK_OVERROUND` | `2/1.85` (~8.1%) | predecir_proximas.py |
| `SEED` | 42 | train_ann.py |
| `MAX_EPOCHS` | 120 | train_ann.py |
| `TRIALS` | 100 | train_ann.py (Optuna) |
| `_SYMM_KEYS` | surface_enc, tourney_level, round_enc, is_best_of_5, indoor, level_weight | train_ann.py |

---

## Bug Noti e Fix Applicati

**`AttributeError: ndarray has no attribute 'values'`** (train_ann.py `calcola_pesi_temporali`):
- Causa: in re-training ann_surface, `d_all_s = d_all.values[mask_all_s]` produce ndarray; `.values` solo su Series
- Fix: `return np.asarray(weights).astype(np.float32)` (riga ~612)

**`ValueError: _check_feature_names`** (sklearn ≥1.2):
- Causa: scaler fittato su ndarray, transform chiamato con DataFrame
- Fix: `scaler.transform(df[ANN_FEATURES].values)` — sempre passare `.values`

**`ValueError: _check_n_features` (32 vs 30)**:
- Causa: pagina predictor aveva `ANN_FEATURES` locale con 30 feature v5.1 che sovra-scriveva l'import
- Fix: rimossa definizione locale, usa import da `prediction_engine.py`

**`G: 0.0%` in sidebar**:
- Causa: `surf_acc.get('Grass', 0)` restituisce 0 se Grass non ha campioni nel test set
- Fix: `"  ".join(f"{k[0]}: {v:.1%}" for k, v in surf_acc.items())` — mostra solo superfici con dati

---

## Metriche Modello v6

- **Accuracy totale**: ~73.5%
- **Log Loss**: ~0.482
- **Hard**: ~74.5% | **Clay**: ~72.1% | **Grass**: ~71.2% (variabile per stagione)
- Dataset: ~60k righe (30k partite × mirror pairs), 2015–2026
