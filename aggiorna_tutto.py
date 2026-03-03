"""
aggiorna_tutto.py  —  Pipeline completo del Tennis Predictor (SEQUENZIALE)
=========================================================================
Esegui semplicemente:
    python aggiorna_tutto.py

Oppure con doppio click su  AGGIORNA.bat

Pipeline SEQUENZIALE:
  FASE 1:  Scraping ranking ATP
  FASE 2:  Scraping partite ATP 2025-2026
  FASE 3:  Arricchimento + correzione dati
  FASE 4:  Fusione storico
  FASE 5:  Profili giocatori
  FASE 6:  Training modelli (XGBoost, Ensemble, LR)
  FASE 7:  Training ANN (opzionale)
  FASE 8:  Court Speed (scraping + arricchimento)
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
ESEGUI_SCRAPING    = False    # Scarica nuovi dati ATP (richiede Chrome installato)
ESEGUI_FUSIONE     = False    # Fonde storico + nuovi dati
ESEGUI_PROFILI     = False    # Rigenera profili giocatori
ESEGUI_COURT_SPEED = True    # Scraping velocità campo + arricchimento CSV
ESEGUI_MODELLI     = True    # Riaddestra XGBoost, Ensemble, LR
ESEGUI_ANN         = False   # True = addestra la rete neurale (lento su CPU ~1-2h)

# ─── Helper ───────────────────────────────────────────────────────────────────

W = 60

def sezione(titolo: str):
    print(f"\n{'='*W}")
    print(f"  {titolo}")
    print(f"{'='*W}", flush=True)

def esegui(script: str, cwd: str, desc: str = ""):
    """Esegue uno script Python con output in tempo reale."""
    print(f"\n▶  {desc or script}")
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
            print(f"\n❌  '{script}' terminato con errore (codice {proc.returncode})")
            return False

        print(f"\n✅  '{script}' completato con successo", flush=True)
        return True

    except FileNotFoundError:
        print(f"\n❌  Script non trovato: {os.path.join(cwd, script)}")
        return False
    except Exception as e:
        print(f"\n❌  Errore imprevisto: {e}")
        return False

def copia_se_esiste(src: str, dst: str):
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"   📋  Copiato: {os.path.basename(src)}  →  {os.path.relpath(dst, ROOT)}")
    else:
        print(f"   ⚠️   Non trovato: {os.path.relpath(src, ROOT)}")

# ─── 0. Controllo dipendenze ──────────────────────────────────────────────────

def controlla_dipendenze():
    print("UNA VOLTA FINITO TUTTO SCARICARE PREDICCION (CARTELLA) E SOSTITUIRLA (SE ERA STATA COPIATA PER INTERO)")
    sezione("0️⃣  Verifica dipendenze")
    print(f"   Python: {PYTHON}")

# ─── PIPELINE ────────────────────────────────────────────────────────────────

def main():
    inizio = time.time()

    print("=" * W)
    print("  🎾  TENNIS PREDICTOR — PIPELINE SEQUENZIALE")
    print(f"  {time.strftime('%d/%m/%Y %H:%M:%S')}")
    print("=" * W)

    controlla_dipendenze()

    scraping_ok = {}

    # ══════════════════════════════════════════════════════════════════════════
    # FASE 1 — Scraping Ranking
    # ══════════════════════════════════════════════════════════════════════════
    if ESEGUI_SCRAPING:
        sezione("1️⃣  FASE 1 — Scraping Ranking ATP")
        scraping_ok["RANK"] = esegui("scraper_ranking.py", SCRAPING, "Ranking ATP")

    # ══════════════════════════════════════════════════════════════════════════
    # FASE 2 — Scraping Partite
    # ══════════════════════════════════════════════════════════════════════════
    if ESEGUI_SCRAPING:
        sezione("2️⃣  FASE 2 — Scraping Partite ATP")
        scraping_ok["MATCH"] = esegui("scraper_2026_final.py", SCRAPING, "Partite ATP 2025-2026")

    # ══════════════════════════════════════════════════════════════════════════
    # FASE 3 — Elaborazione dati
    # ══════════════════════════════════════════════════════════════════════════
    if ESEGUI_SCRAPING:
        sezione("3️⃣  FASE 3 — Elaborazione dati")

        if scraping_ok.get("MATCH", False):
            esegui("enriquecer_2026.py",             SCRAPING, "Arricchimento dati 2026")
            esegui("corregir_superficie_ranking.py", SCRAPING, "Correzione superfici e ranking")
        else:
            print("   ⚠️   Scraping partite fallito — uso dati precedenti")

    # ══════════════════════════════════════════════════════════════════════════
    # FASE 4 — Fusione storico
    # ══════════════════════════════════════════════════════════════════════════
    if ESEGUI_FUSIONE:
        sezione("4️⃣  FASE 4 — Fusione dataset")
        ok = esegui("fusionar_historico_final.py", SCRAPING, "Fusione storico completo")
        if not ok:
            print("   ❌  Fusione fallita. Impossibile continuare.")
            return
        copia_se_esiste(
            os.path.join(SCRAPING, "historialTenis.csv"),
            os.path.join(PREDICCION, "historialTenis.csv")
        )

    # ══════════════════════════════════════════════════════════════════════════
    # FASE 5 — Profili giocatori
    # ══════════════════════════════════════════════════════════════════════════
    if ESEGUI_PROFILI:
        sezione("5️⃣  FASE 5 — Profili giocatori")
        ok = esegui("generar_perfiles.py", SCRAPING, "Profili giocatori")
        if ok:
            copia_se_esiste(
                os.path.join(SCRAPING, "perfiles_jugadores.pkl"),
                os.path.join(PREDICCION, "perfiles_jugadores.pkl")
            )

    # ══════════════════════════════════════════════════════════════════════════
    # FASE 6 — Court Speed (scraping + arricchimento) — PRIMA del training
    # ══════════════════════════════════════════════════════════════════════════
    if ESEGUI_COURT_SPEED:
        sezione("6️⃣  FASE 6 — Court Speed (scraping + arricchimento)")

        ok_speed = esegui("scraper_court_speed.py", SCRAPING, "Court Speed 1991-2026")
        if ok_speed:
            esegui("enriquecer_court_speed.py", SCRAPING, "Arricchimento court speed")
            copia_se_esiste(
                os.path.join(SCRAPING, "court_speed_dict.pkl"),
                os.path.join(PREDICCION, "court_speed_dict.pkl")
            )
        else:
            print("   ⚠️   Court Speed fallito — court_speed_dict.pkl non aggiornato")

    # ══════════════════════════════════════════════════════════════════════════
    # FASE 7 — Training modelli (sequenziale)
    # ══════════════════════════════════════════════════════════════════════════
    if ESEGUI_MODELLI:
        sezione("7️⃣  FASE 7 — Training modelli")

        modelli = [
            ("predict_xgboost.py",  "Training XGBoost"),
            ("predict_ensemble.py", "Training Ensemble (LR+RF+XGB)"),
            ("predict_LR.py",      "Training Logistic Regression"),
        ]
        for script, desc in modelli:
            esegui(script, PREDICCION, desc)

    # ══════════════════════════════════════════════════════════════════════════
    # FASE 8 — Training ANN (opzionale)
    # ══════════════════════════════════════════════════════════════════════════
    if ESEGUI_ANN:
        sezione("8️⃣  FASE 8 — Training ANN")
        print("   💡  Suggerimento: usa colab_training.ipynb su Colab con GPU T4")
        ok = esegui("train_ann.py", PREDICCION, "Optuna Bayesian search + ANN")
        if ok:
            print("   🏆  Modello ANN addestrato e salvato in prediccion/")
    else:
        sezione("8️⃣  FASE 8 — Training ANN")
        print("   ℹ️   Saltato (ESEGUI_ANN = False).")
        print("   →   Imposta ESEGUI_ANN = True oppure usa Google Colab")

    # ── Riepilogo ─────────────────────────────────────────────────────────────
    fine   = time.time()
    minuti = (fine - inizio) / 60

    sezione("✅  RIEPILOGO")
    print(f"   ⏱️  Tempo totale: {minuti:.1f} minuti")
    print()

    files_check = [
        (os.path.join(SCRAPING,   "historialTenis.csv"),        "historialTenis.csv (dataset)"),
        (os.path.join(SCRAPING,   "ranking_2026.csv"),           "ranking_2026.csv"),
        (os.path.join(SCRAPING,   "perfiles_jugadores.pkl"),     "perfiles_jugadores.pkl"),
        (os.path.join(SCRAPING,   "court_speed_dict.pkl"),       "court_speed_dict.pkl"),
        (os.path.join(PREDICCION, "modelo_xgboost_final.pkl"),  "modelo_xgboost_final.pkl"),
        (os.path.join(PREDICCION, "modelo_ensemble.pkl"),        "modelo_ensemble.pkl"),
        (os.path.join(PREDICCION, "modelo_ann.pth"),             "modelo_ann.pth (ANN)"),
        (os.path.join(PREDICCION, "scaler_ann.pkl"),             "scaler_ann.pkl (ANN)"),
    ]
    for path, desc in files_check:
        stato = "✅" if os.path.exists(path) else "❌ MANCANTE"
        print(f"   {stato}  {desc}")

    print()
    print("   🚀  Per avviare la dashboard:")
    print(f"       cd \"{ROOT}\"")
    print(f"       \".\\venv\\Scripts\\streamlit.exe\" run \"0_🏠_Inicio.py\"")
    print("=" * W)


if __name__ == "__main__":
    main()
