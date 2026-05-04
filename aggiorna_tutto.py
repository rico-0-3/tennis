"""
aggiorna_tutto.py  —  Pipeline completo del Tennis Predictor (SEQUENZIALE)
=========================================================================
Esegui semplicemente:
    python aggiorna_tutto.py

Oppure con doppio click su  AGGIORNA.bat

Pipeline SEQUENZIALE:
  FASE 1:   Scraping ranking ATP
  FASE 2:   Download incrementale dati TML (TennisMyLife) -> master_dataset.csv
  FASE 5b:  Bio giocatori (DOB + altezza) — solo se ESEGUI_BIO=True
  FASE 5:   Profili giocatori
  FASE 6:   Court Speed (scraping + arricchimento)
  FASE 10:  Predizioni prossime partite
  FASE 7:   Training modelli (XGBoost, Ensemble, LR) — solo se ESEGUI_MODELLI=True
  FASE 8:   Training ANN — solo se ESEGUI_ANN=True
  FASE 9:   Training Scommesse Speciali — solo se ESEGUI_SPECIAL_BETS=True
"""

import subprocess
import sys
import os
import time
import shutil

# ─── Configurazione ──────────────────────────────────────────────────────────

ROOT       = os.path.dirname(os.path.abspath(__file__))
VENV_PY    = os.path.join(ROOT, "venv", "Scripts", "python.exe")   # Windows
SCRAPING   = os.path.join(ROOT, "scraping")
PREDICCION = os.path.join(ROOT, "prediccion")

# Se il venv non esiste tentiamo di usare il Python corrente
PYTHON = VENV_PY if os.path.exists(VENV_PY) else sys.executable

# ─── Flag opzionali ───────────────────────────────────────────────────────────
ESEGUI_SCRAPING    = True    # Scarica nuovi dati ATP (ranking + download TML)
ESEGUI_BIO         = True    # Scraping bio (DOB + altezza) da tennisstats.com
ESEGUI_PROFILI     = True    # Rigenera profili giocatori
ESEGUI_COURT_SPEED = True    # Scraping velocità campo + arricchimento CSV
ESEGUI_MODELLI     = False   # Riaddestra XGBoost, Ensemble, LR
ESEGUI_ANN         = False   # True = addestra la rete neurale (lento su CPU ~1-2h)
ESEGUI_SPECIAL_BETS = False  # True = addestra i modelli Poisson/Tweedie per Ace, DF e Break

# ─── Helper ───────────────────────────────────────────────────────────────────

W = 60

def sezione(titolo: str):
    print(f"\n{'='*W}")
    print(f"  {titolo}")
    print(f"{'='*W}", flush=True)

def esegui(script: str, cwd: str, desc: str = ""):
    """Esegue uno script Python con output in tempo reale."""
    print(f"\n>  {desc or script}")
    print(f"   ({cwd})")
    print("-"*W)

    try:
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        proc = subprocess.Popen(
            [PYTHON, "-u", "-X", "utf8", script],
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )

        for line in proc.stdout:
            print("   " + line, end="", flush=True)

        proc.wait()

        if proc.returncode != 0:
            print(f"\n  '{script}' terminato con errore (codice {proc.returncode})")
            return False

        print(f"\n  '{script}' completato con successo", flush=True)
        return True

    except FileNotFoundError:
        print(f"\n  Script non trovato: {os.path.join(cwd, script)}")
        return False
    except Exception as e:
        print(f"\n  Errore imprevisto: {e}")
        return False

def copia_se_esiste(src: str, dst: str):
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"   Copiato: {os.path.basename(src)}  ->  {os.path.relpath(dst, ROOT)}")
    else:
        print(f"   Non trovato: {os.path.relpath(src, ROOT)}")

def filtra_ritiri_e_copia(src: str, dst: str):
    """Filtra match con RET e W/O prima di copiare in prediccion."""
    if not os.path.exists(src):
        print(f"   Non trovato: {os.path.relpath(src, ROOT)}")
        return

    try:
        import pandas as pd

        print(f"   Caricamento dataset: {os.path.basename(src)}")
        df = pd.read_csv(src, low_memory=False)

        n_originale = len(df)
        print(f"   Match totali: {n_originale:,}")

        # Filtra ritiri (RET) e walkovers (W/O)
        mask_ritiri = df['score'].astype(str).str.contains('RET', na=False, case=False)
        mask_wo = df['score'].astype(str).str.contains('W/O', na=False, case=False)

        n_ritiri = mask_ritiri.sum()
        n_wo = mask_wo.sum()

        df_pulito = df[~(mask_ritiri | mask_wo)].copy()
        n_finale = len(df_pulito)

        df_pulito.to_csv(dst, index=False)

        print(f"   Rimossi {n_ritiri:,} ritiri (RET)")
        print(f"   Rimossi {n_wo:,} walkovers (W/O)")
        print(f"   Match puliti: {n_finale:,} ({100*n_finale/n_originale:.1f}%)")
        print(f"   Salvato: {os.path.relpath(dst, ROOT)}")

    except Exception as e:
        print(f"   Errore durante il filtraggio: {e}")
        print(f"   Fallback: copia normale")
        shutil.copy2(src, dst)

# ─── 0. Controllo dipendenze ──────────────────────────────────────────────────

def controlla_dipendenze():
    print("UNA VOLTA FINITO TUTTO SCARICARE PREDICCION (CARTELLA) E SOSTITUIRLA (SE ERA STATA COPIATA PER INTERO)")
    sezione("0  Verifica dipendenze")
    print(f"   Python: {PYTHON}")

# ─── PIPELINE ────────────────────────────────────────────────────────────────

def main():
    inizio = time.time()

    print("=" * W)
    print("  TENNIS PREDICTOR — PIPELINE SEQUENZIALE")
    print(f"  {time.strftime('%d/%m/%Y %H:%M:%S')}")
    print("=" * W)

    controlla_dipendenze()

    scraping_ok = {}

    # =========================================================================
    # FASE 1 — Scraping Ranking
    # =========================================================================
    if ESEGUI_SCRAPING:
        sezione("FASE 1 — Scraping Ranking ATP")
        scraping_ok["RANK"] = esegui("scraper_ranking.py", SCRAPING, "Ranking ATP")

    # =========================================================================
    # FASE 2 — Download incrementale TML (TennisMyLife)
    # Scarica solo i file aggiornati, produce master_dataset.csv
    # =========================================================================
    if ESEGUI_SCRAPING:
        sezione("FASE 2 — Download dati TML (TennisMyLife)")
        scraping_ok["MATCH"] = esegui("download_data.py", SCRAPING, "Download incrementale TML -> master_dataset.csv")
        if not scraping_ok["MATCH"]:
            print("   Download TML fallito. Impossibile continuare.")
            return

        # Filtra ritiri e walkovers, copia master_dataset in prediccion
        filtra_ritiri_e_copia(
            os.path.join(SCRAPING, "master_dataset.csv"),
            os.path.join(PREDICCION, "master_dataset.csv")
        )

    # =========================================================================
    # FASE 5b — Bio giocatori (DOB + altezza da tennisstats.com)
    # =========================================================================
    if ESEGUI_BIO:
        sezione("FASE 5b — Bio giocatori (tennisstats.com)")
        ok_bio = esegui("scraper_bio_jugadores.py", SCRAPING, "Bio giocatori (DOB + altezza)")
        if ok_bio:
            copia_se_esiste(
                os.path.join(SCRAPING, "bio_jugadores.json"),
                os.path.join(PREDICCION, "bio_jugadores.json")
            )

    # =========================================================================
    # FASE 5 — Profili giocatori
    # =========================================================================
    if ESEGUI_PROFILI:
        sezione("FASE 5 — Profili giocatori")
        ok = esegui("generar_perfiles.py", SCRAPING, "Profili giocatori")
        if ok:
            copia_se_esiste(
                os.path.join(SCRAPING, "perfiles_jugadores.pkl"),
                os.path.join(PREDICCION, "perfiles_jugadores.pkl")
            )
            copia_se_esiste(
                os.path.join(SCRAPING, "bio_jugadores.json"),
                os.path.join(PREDICCION, "bio_jugadores.json")
            )

    # =========================================================================
    # FASE 6 — Court Speed (scraping + arricchimento) — PRIMA del training
    # =========================================================================
    if ESEGUI_COURT_SPEED:
        sezione("FASE 6 — Court Speed (scraping + arricchimento)")

        ok_speed = esegui("scraper_court_speed.py", SCRAPING, "Court Speed 1991-2026")
        if ok_speed:
            esegui("enriquecer_court_speed.py", SCRAPING, "Arricchimento court speed")
            copia_se_esiste(
                os.path.join(SCRAPING, "court_speed_dict.pkl"),
                os.path.join(PREDICCION, "court_speed_dict.pkl")
            )
        else:
            print("   Court Speed fallito — court_speed_dict.pkl non aggiornato")

    # =========================================================================
    # FASE 10 — Predizioni Prossime Partite (sempre)
    # =========================================================================
    sezione("FASE 10 — Predizioni Prossime Partite")
    esegui("scraper_proximas_partidas.py", SCRAPING,   "Scraping partite upcoming")
    esegui("predecir_proximas.py",         PREDICCION, "Predizioni batch prossime partite")

    # =========================================================================
    # FASE 7 — Training modelli (sequenziale)
    # =========================================================================
    if ESEGUI_MODELLI:
        sezione("FASE 7 — Training modelli")

        modelli = [
            ("predict_xgboost.py",  "Training XGBoost"),
            ("predict_ensemble.py", "Training Ensemble (LR+RF+XGB)"),
            ("predict_LR.py",      "Training Logistic Regression"),
        ]
        for script, desc in modelli:
            esegui(script, PREDICCION, desc)

    # =========================================================================
    # FASE 8 — Training ANN (opzionale)
    # =========================================================================
    if ESEGUI_ANN:
        sezione("FASE 8 — Training ANN")
        print("   Suggerimento: usa colab_training.ipynb su Colab con GPU T4")
        ok = esegui("train_ann.py", PREDICCION, "Optuna Bayesian search + ANN")
        if ok:
            print("   Modello ANN addestrato e salvato in prediccion/")
    else:
        sezione("FASE 8 — Training ANN")
        print("   Saltato (ESEGUI_ANN = False).")
        print("   ->  Imposta ESEGUI_ANN = True oppure usa Google Colab")

    # =========================================================================
    # FASE 9 — Training Scommesse Speciali (Ace, Doppi Falli, Break)
    # =========================================================================
    if ESEGUI_SPECIAL_BETS:
        sezione("FASE 9 — Training Scommesse Speciali")
        ok = esegui("train_special_bet.py", PREDICCION, "Training modelli Ace, DF, Break (Poisson/Tweedie)")
        if ok:
            print("   Modelli speciali addestrati e salvati in prediccion/")
    else:
        sezione("FASE 9 — Training Scommesse Speciali")
        print("   Saltato (ESEGUI_SPECIAL_BETS = False).")

    # ─── Riepilogo ────────────────────────────────────────────────────────────
    fine   = time.time()
    minuti = (fine - inizio) / 60

    sezione("RIEPILOGO")
    print(f"   Tempo totale: {minuti:.1f} minuti")
    print()

    files_check = [
        (os.path.join(SCRAPING,   "master_dataset.csv"),          "master_dataset.csv (dataset TML)"),
        (os.path.join(SCRAPING,   "ranking_2026.csv"),             "ranking_2026.csv"),
        (os.path.join(SCRAPING,   "perfiles_jugadores.pkl"),       "perfiles_jugadores.pkl"),
        (os.path.join(SCRAPING,   "court_speed_dict.pkl"),         "court_speed_dict.pkl"),
        (os.path.join(PREDICCION, "master_dataset.csv"),           "prediccion/master_dataset.csv"),
        (os.path.join(PREDICCION, "modelo_finale.pkl"),            "modelo_finale.pkl (ANN v6)"),
        (os.path.join(PREDICCION, "scaler_ann.pkl"),               "scaler_ann.pkl (ANN)"),
        (os.path.join(PREDICCION, "glicko2_stores.pkl"),           "glicko2_stores.pkl"),
        (os.path.join(PREDICCION, "modelos_special_bets.pkl"),     "modelos_special_bets.pkl"),
    ]
    for path, desc in files_check:
        stato = "OK" if os.path.exists(path) else "MANCANTE"
        print(f"   {stato}  {desc}")

    print()
    print("   Per avviare la dashboard:")
    print(f"       cd \"{ROOT}\"")
    print(f"       \".\\venv\\Scripts\\streamlit.exe\" run \"0_Inicio.py\"")
    print("=" * W)


if __name__ == "__main__":
    main()
