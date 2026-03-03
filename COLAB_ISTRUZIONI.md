# 🎾 Training ANN su Google Colab

## File da caricare (4 file)

| File | Cartella locale | Descrizione |
|------|----------------|-------------|
| `train_ann.py` | `prediccion/` | Script di training ANN |
| `historialTenis.csv` | `prediccion/` | Dataset completo partite |
| `court_speed_dict.pkl` | `scraping/` | Dizionario velocità campo |
| `court_speed_helper.py` | `scraping/` | Helper per court speed |

## Struttura su Colab

```
/content/
├── prediccion/
│   ├── train_ann.py
│   └── historialTenis.csv (da scraping)
└── scraping/
    ├── court_speed_dict.pkl
    └── court_speed_helper.py
```

## Comandi su Colab

```python
# Cella 1 — Installa dipendenze
!pip install optuna torch pandas scikit-learn joblib

# Cella 2 — Crea cartelle e carica file
import os
os.makedirs('prediccion', exist_ok=True)
os.makedirs('scraping', exist_ok=True)
# Poi carica i 4 file nelle rispettive cartelle con il file manager di Colab

# Cella 3 — Training
%cd prediccion
!python train_ann.py
```

## File generati da scaricare

Dopo il training, scarica questi file dalla cartella `prediccion/` di Colab e mettili nella cartella `prediccion/` locale:

- `modelo_ann.pth` — Modello globale
- `modelo_ann_clay.pth` — Modello Clay
- `modelo_ann_hard.pth` — Modello Hard
- `modelo_ann_grass.pth` — Modello Grass
- `ann_config.json` — Config globale
- `ann_config_clay.json` — Config Clay
- `ann_config_hard.json` — Config Hard
- `ann_config_grass.json` — Config Grass
- `scaler_ann.pkl` — Scaler feature
- `elo_surface.pkl` — Rating Elo per superficie
- `streak_players.pkl` — Striscia attiva giocatori
- `stats_superficie_v2.pkl` — Win-rate per superficie
- `resultados_comparacion_finale.csv` — Confronto modelli

## Note

- **Runtime**: Usa GPU T4 su Colab (Runtime → Cambia tipo di runtime → T4 GPU)
- **Tempo**: ~15-30 min con GPU, ~1-2h su CPU
- `aggiorna_tutto.py` **NON serve su Colab** — si usa solo in locale per scraping + modelli classici
