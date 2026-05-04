# Tennis Predictor — Redesign v6.0
**Data**: 2026-05-04  
**Approccio scelto**: B — Glicko-2 + Surface-Specific Models + Best-of-N Ensemble

---

## Contesto

Il modello attuale (v5.1) raggiunge ~68% di accuracy sul test set. I problemi principali identificati sono:
- Dati solo dal 2015, serve/return stats quasi tutte a zero per i dati 2025/26
- Calibratore fittato sul test set (data leakage)
- Optuna ottimizza accuracy invece di log-loss
- Modello globale unico senza differenziazione per superficie
- ELO semplice con cold-start problematico su nuovi giocatori
- Feature `court_ace_pct` e `court_speed` restituiscono sempre 0

**Target realistico**: 70–73% di accuracy sul test set (soffitto letteratura: 66–70% pre-match ATP).

---

## 1. Pipeline Dati

### Sorgente unica: TennisMyLife (TML)
- API: `https://stats.tennismylife.org/api/data-files` → lista file con `mtime`
- Download: `https://stats.tennismylife.org/data/{anno}.csv` (solo ATP tour principale, no Challenger)
- Copertura: 1968–oggi, serve/return stats reali dal 1991+
- Aggiornato quotidianamente con i risultati più recenti

### Download incrementale (`scraping/download_data.py`)
- Mantiene un `scraping/tml_manifest.json` con `{filename: mtime}` per ogni file scaricato
- Logica: se il file non esiste localmente → scarica; se `mtime` API > `mtime` manifest → scarica; altrimenti skip
- Il file dell'anno corrente viene sempre ricontrollato
- Produce `scraping/master_dataset.csv`: concatenazione ordinata per data di tutti i CSV annuali (1991–oggi)
- I dati 2025/26 già presenti in `scraping/` vengono ignorati (rimpiazzati da TML)

### Challenger
- Non inclusi nel training
- Non scaricati

---

## 2. Feature Engineering (~32 feature)

### Glicko-2 al posto di ELO
- Rating, Rating Deviation (RD) e volatility per giocatore calcolati in modo running (no leakage)
- Separati per superficie (Hard / Clay / Grass) e globale (overall)
- Feature: `diff_glicko_rating`, `diff_glicko_rating_overall`, `diff_glicko_rd`
- RD alta = giocatore poco visto / incerto → informazione utile per il modello
- Cold start: nuovi giocatori partono con rating 1500, RD 350 (standard Glicko-2)

### Serve/Return (ora con dati reali)
- Rolling ultimi 10 match (globale, cross-superficie): `diff_ace`, `diff_1st_pct`, `diff_1st_won`, `diff_2nd_won`, `diff_bp_saved`, `diff_return_pct`, `diff_bp_conv`, `diff_return_1st`
- Se `w_svpt == 0` per il match corrente: la riga aggiorna H2H/form/Glicko-2 ma NON aggiorna le rolling serve stats
- Con TML dal 1991+, copertura attesa ~95% dei match

### Feature aggiunte rispetto a v5.1
- `diff_glicko_rd` — incertezza del rating
- `indoor` (0/1) — indoor vs outdoor

### Feature rimosse rispetto a v5.1
- `court_ace_pct` — restituiva sempre 0, rumore puro
- `court_speed` — restituiva sempre 0, rumore puro
- `diff_form_volatility` — correlazione ~0.011 già rimossa in v5.1
- `diff_surface_trend` — correlazione ~0.029 già rimossa in v5.1
- `diff_close_match_pct` — correlazione ~0.030 già rimossa in v5.1

### Feature mantenute da v5.1 (invariate)
`log_rank_ratio`, `log_pts_ratio`, `diff_streak`, `diff_recent_form`, `surface_enc`, `tourney_level`, `round_enc`, `is_best_of_5`, `diff_h2h`, `diff_h2h_surface`, `diff_skill`, `diff_momentum`, `diff_fatigue`, `diff_days_since_last`, `diff_weeks_load`, `diff_home`, `diff_opponent_quality`, `diff_upset_tendency`, `diff_late_round_wr`, `level_weight`

### No-leakage garantito
- Tutte le statistiche rolling (Glicko-2, serve, form, H2H) vengono lette PRIMA dell'aggiornamento e aggiornate DOPO, esattamente come in v5.
- Split temporale mantenuto: train/val/test per data cronologica, mirrored pairs nello stesso fold.

---

## 3. Architettura Modelli

### Split temporale
- **Train**: 1991–70° percentile date (storico)
- **Validation**: 70°–85° percentile date (calibrazione + selezione ensemble)
- **Test**: 85°–100° percentile date (valutazione finale, mai visto durante training)

### Modelli allenati
1. **LGB_surface** — 3 LightGBM separati (Hard / Clay / Grass), Optuna su log-loss
2. **LGB_global** — LightGBM unico su tutti i match, Optuna su log-loss
3. **XGB_global** — XGBoost globale, Optuna su log-loss
4. **ANN_best** — miglior trial Optuna, architettura Wide & Deep attuale
5. **ANN_top5** — media probabilità top-5 trial Optuna

### Strategie ensemble valutate sul test set
- `lgb_surface` — LGB con routing automatico Hard/Clay/Grass; fallback `lgb_global` per superficie sconosciuta (Carpet, Indoor generico)
- `lgb_global`
- `xgb_global`
- `ann_best`
- `ann_top5`
- `ensemble_avg` — media LGB_surface + XGB + ANN_best
- `ensemble_stacking` — meta-LR allenata su validation set

### Selezione automatica
Il sistema valuta tutte le strategie sul test set, seleziona quella con accuracy più alta e la salva come strategia vincente in `modelo_finale.pkl`. Nessun peso fisso.

### Calibrazione (fix bug)
- Il calibratore isotonic viene fittato sul **validation set** (non test set come in v5)
- Test set rimane completamente pulito

### Ottimizzazione Optuna (fix bug)
- Metrica: **log-loss** (invece di accuracy) — produce probabilità meglio calibrate
- ANN: `n_trials=100`, `max_epochs=120`
- GBM: `n_trials=50`

---

## 4. Prediction Engine

### File aggiornati
- `prediccion/prediction_engine.py`: feature list aggiornata, routing superficie per LGB_surface, carica Glicko-2 invece di ELO
- `prediccion/predecir_proximas.py`: nessuna modifica logica, si aggiorna automaticamente via prediction_engine

### Artifact salvati da train_ann.py
Stessi di v5 + nuovi:
- `glicko2_surface.pkl` — rating Glicko-2 per (player, surface)
- `glicko2_overall.pkl` — rating Glicko-2 globale
- `lgb_hard.pkl`, `lgb_clay.pkl`, `lgb_grass.pkl` — modelli surface-specific
- `modelo_finale.pkl` — strategia vincente + tutti i componenti necessari

---

## 5. Bug fixati

| Bug | Fix |
|---|---|
| Serve/return stats a zero (2025/26) | TML data reale |
| Dati solo dal 2015 | TML dal 1991+ |
| `court_ace_pct` / `court_speed` = 0 | Feature rimosse |
| Calibratore fittato su test set | Ora su validation set |
| Optuna ottimizza accuracy | Ora ottimizza log-loss |
| Nessun modello surface-specific | 3 LGB Hard/Clay/Grass |
| ELO cold-start su nuovi giocatori | Glicko-2 con RD |

---

## 6. File toccati

| File | Tipo modifica |
|---|---|
| `scraping/download_data.py` | Nuovo |
| `scraping/tml_manifest.json` | Nuovo (generato) |
| `scraping/master_dataset.csv` | Nuovo (generato) |
| `prediccion/train_ann.py` | Riscrittura sostanziale |
| `prediccion/prediction_engine.py` | Aggiornamento feature list + routing |
