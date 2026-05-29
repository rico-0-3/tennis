"""
run_pipeline.py  —  Pipeline CLI per GitHub Actions
====================================================
Versione di aggiorna_tutto.py che accetta flag da riga di comando.
Pensata per essere eseguita da GitHub Actions.

Uso:
    python run_pipeline.py --scraping --profili --court-speed
    python run_pipeline.py --scraping --profili --court-speed --modelli --ann --special-bets
"""

import subprocess
import sys
import os
import time
import shutil
import argparse

# ─── Configurazione ──────────────────────────────────────────────────────────

ROOT       = os.path.dirname(os.path.abspath(__file__))
SCRAPING   = os.path.join(ROOT, "scraping")
PREDICCION = os.path.join(ROOT, "prediccion")
PYTHON     = sys.executable

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

# ─── PIPELINE ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pipeline Tennis Predictor")
    parser.add_argument("--scraping",     action="store_true", help="Download TML (ranking + dati partite)")
    parser.add_argument("--profili",      action="store_true", help="Profili giocatori")
    parser.add_argument("--bio",          action="store_true", help="Bio giocatori (tennisstats.com)")
    parser.add_argument("--court-speed",  action="store_true", help="Court Speed")
    parser.add_argument("--modelli",      action="store_true", help="Training modelli")
    parser.add_argument("--ann",          action="store_true", help="Training ANN")
    parser.add_argument("--special-bets", action="store_true", help="Training Special Bets (Ace, DF, Break)")
    args = parser.parse_args()

    if not any([args.scraping, args.profili, args.bio,
                args.court_speed, args.modelli, args.ann, args.special_bets]):
        print("  Nessuna fase selezionata. Usa --scraping, --profili, ecc.")
        sys.exit(1)

    inizio = time.time()

    print("=" * W)
    print("  TENNIS PREDICTOR — PIPELINE (GitHub Actions)")
    print(f"  {time.strftime('%d/%m/%Y %H:%M:%S')}")
    print("=" * W)
    print(f"   Python: {PYTHON}")

    # ═══ FASE 1 — Scraping Ranking ═══════════════════════════════════════════
    if args.scraping:
        sezione("FASE 1 — Scraping Ranking ATP")
        esegui("scraper_ranking.py", SCRAPING, "Ranking ATP")

    # ═══ FASE 2 — Download incrementale TML ══════════════════════════════════
    if args.scraping:
        sezione("FASE 2 — Download dati TML (TennisMyLife)")
        ok_tml = esegui("download_data.py", SCRAPING, "Download incrementale TML -> master_dataset.csv")
        if not ok_tml:
            print("   Download TML fallito. Impossibile continuare.")
            return

        # ── FASE 2b: risultati recenti da TennisExplorer (colma il lag TML) ──
        sezione("FASE 2b — Risultati recenti (TennisExplorer, gap TML)")
        esegui("scraper_resultados_recientes.py", SCRAPING,
               "Risultati mancanti fino a ieri (TennisExplorer)")

        filtra_ritiri_e_copia(
            os.path.join(SCRAPING,   "master_dataset.csv"),
            os.path.join(PREDICCION, "master_dataset.csv"),
        )

    # ═══ FASE 5b — Bio giocatori ════════════════════════════════════════════
    if args.bio:
        sezione("FASE 5b — Bio giocatori (tennisstats.com)")
        ok_bio = esegui("scraper_bio_jugadores.py", SCRAPING, "Bio giocatori (DOB + altezza)")
        if ok_bio:
            copia_se_esiste(
                os.path.join(SCRAPING, "bio_jugadores.json"),
                os.path.join(PREDICCION, "bio_jugadores.json"),
            )

    # ═══ FASE 5 — Profili giocatori ══════════════════════════════════════════
    if args.profili:
        sezione("FASE 5 — Profili giocatori")
        ok = esegui("generar_perfiles.py", SCRAPING, "Profili giocatori")
        if ok:
            copia_se_esiste(
                os.path.join(SCRAPING, "perfiles_jugadores.pkl"),
                os.path.join(PREDICCION, "perfiles_jugadores.pkl"),
            )
            copia_se_esiste(
                os.path.join(SCRAPING, "bio_jugadores.json"),
                os.path.join(PREDICCION, "bio_jugadores.json"),
            )

    # ═══ FASE 6 — Court Speed ════════════════════════════════════════════════
    if args.court_speed:
        sezione("FASE 6 — Court Speed (scraping + arricchimento)")
        ok_speed = esegui("scraper_court_speed.py", SCRAPING, "Court Speed 1991-2026")
        if ok_speed:
            esegui("enriquecer_court_speed.py", SCRAPING, "Arricchimento court speed")
            copia_se_esiste(
                os.path.join(SCRAPING, "court_speed_dict.pkl"),
                os.path.join(PREDICCION, "court_speed_dict.pkl"),
            )
        else:
            print("   Court Speed fallito — court_speed_dict.pkl non aggiornato")

    # ═══ FASE 10 — Predizioni Prossime Partite (sempre) ══════════════════════
    sezione("FASE 10 — Predizioni Prossime Partite")
    esegui("scraper_proximas_partidas.py", SCRAPING,   "Scraping partite upcoming")
    esegui("predecir_proximas.py",         PREDICCION, "Predizioni batch prossime partite")

    # ═══ FASE 7 — Training modelli ═══════════════════════════════════════════
    if args.modelli:
        sezione("FASE 7 — Training modelli")
        modelli = [
            ("predict_xgboost.py",  "Training XGBoost"),
            ("predict_ensemble.py", "Training Ensemble (LR+RF+XGB)"),
            ("predict_LR.py",       "Training Logistic Regression"),
        ]
        for script, desc in modelli:
            esegui(script, PREDICCION, desc)

    # ═══ FASE 8 — Training ANN ═══════════════════════════════════════════════
    if args.ann:
        sezione("FASE 8 — Training ANN")
        ok = esegui("train_ann.py", PREDICCION, "Optuna Bayesian search + ANN")
        if ok:
            print("   Modello ANN addestrato e salvato in prediccion/")

    # ═══ FASE 9 — Training Scommesse Speciali ════════════════════════════════
    if args.special_bets:
        sezione("FASE 9 — Training Scommesse Speciali")
        ok = esegui("train_special_bet.py", PREDICCION, "Training modelli Ace, DF, Break")
        if ok:
            print("   Modelli speciali addestrati e salvati in prediccion/")

    # ── Riepilogo ────────────────────────────────────────────────────────────
    fine   = time.time()
    minuti = (fine - inizio) / 60

    sezione("RIEPILOGO")
    print(f"   Tempo totale: {minuti:.1f} minuti")
    print()

    files_check = [
        (os.path.join(SCRAPING,   "master_dataset.csv"),        "master_dataset.csv (dataset TML)"),
        (os.path.join(SCRAPING,   "ranking_2026.csv"),          "ranking_2026.csv"),
        (os.path.join(SCRAPING,   "perfiles_jugadores.pkl"),    "perfiles_jugadores.pkl"),
        (os.path.join(SCRAPING,   "court_speed_dict.pkl"),      "court_speed_dict.pkl"),
        (os.path.join(PREDICCION, "master_dataset.csv"),        "prediccion/master_dataset.csv"),
        (os.path.join(PREDICCION, "modelo_finale.pkl"),         "modelo_finale.pkl (ANN v6)"),
        (os.path.join(PREDICCION, "scaler_ann.pkl"),            "scaler_ann.pkl"),
        (os.path.join(PREDICCION, "glicko2_stores.pkl"),        "glicko2_stores.pkl"),
        (os.path.join(PREDICCION, "modelos_special_bets.pkl"),  "modelos_special_bets.pkl"),
    ]
    for path, desc in files_check:
        stato = "OK" if os.path.exists(path) else "MANCANTE"
        print(f"   {stato}  {desc}")

    print()
    print("=" * W)

if __name__ == "__main__":
    main()
