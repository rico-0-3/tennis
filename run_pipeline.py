"""
run_pipeline.py  —  Pipeline CLI per GitHub Actions
====================================================
Versione di aggiorna_tutto.py che accetta flag da riga di comando.
Pensata per essere eseguita da GitHub Actions.

Uso:
    python run_pipeline.py --scraping --fusione --profili --court-speed
    python run_pipeline.py --scraping --fusione --profili --court-speed --modelli --ann
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

# ─── PIPELINE ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pipeline Tennis Predictor")
    parser.add_argument("--scraping",    action="store_true", help="Scraping ATP")
    parser.add_argument("--fusione",     action="store_true", help="Fusione storico")
    parser.add_argument("--profili",     action="store_true", help="Profili giocatori")
    parser.add_argument("--court-speed", action="store_true", help="Court Speed")
    parser.add_argument("--modelli",     action="store_true", help="Training modelli")
    parser.add_argument("--ann",         action="store_true", help="Training ANN")
    args = parser.parse_args()

    # Se nessun flag passato, non fare nulla
    if not any([args.scraping, args.fusione, args.profili,
                args.court_speed, args.modelli, args.ann]):
        print("⚠️  Nessuna fase selezionata. Usa --scraping, --fusione, ecc.")
        sys.exit(1)

    inizio = time.time()

    print("=" * W)
    print("  🎾  TENNIS PREDICTOR — PIPELINE (GitHub Actions)")
    print(f"  {time.strftime('%d/%m/%Y %H:%M:%S')}")
    print("=" * W)
    print(f"   Python: {PYTHON}")

    scraping_ok = {}

    # ═══ FASE 1 — Scraping Ranking ═══════════════════════════════════════════
    if args.scraping:
        sezione("1️⃣  FASE 1 — Scraping Ranking ATP")
        scraping_ok["RANK"] = esegui("scraper_ranking.py", SCRAPING, "Ranking ATP")

    # ═══ FASE 2 — Scraping Partite ═══════════════════════════════════════════
    if args.scraping:
        sezione("2️⃣  FASE 2 — Scraping Partite ATP")
        scraping_ok["MATCH"] = esegui("scraper_2026_final.py", SCRAPING, "Partite ATP 2025-2026")

    # ═══ FASE 3 — Elaborazione dati ══════════════════════════════════════════
    if args.scraping:
        sezione("3️⃣  FASE 3 — Elaborazione dati")
        if scraping_ok.get("MATCH", False):
            esegui("enriquecer_2026.py",             SCRAPING, "Arricchimento dati 2026")
            esegui("corregir_superficie_ranking.py", SCRAPING, "Correzione superfici e ranking")
        else:
            print("   ⚠️   Scraping partite fallito — uso dati precedenti")

    # ═══ FASE 4 — Fusione storico ════════════════════════════════════════════
    if args.fusione:
        sezione("4️⃣  FASE 4 — Fusione dataset")
        ok = esegui("fusionar_historico_final.py", SCRAPING, "Fusione storico completo")
        if not ok:
            print("   ❌  Fusione fallita. Impossibile continuare.")
            return
        copia_se_esiste(
            os.path.join(SCRAPING, "historialTenis.csv"),
            os.path.join(PREDICCION, "historialTenis.csv")
        )

    # ═══ FASE 5 — Profili giocatori ══════════════════════════════════════════
    if args.profili:
        sezione("5️⃣  FASE 5 — Profili giocatori")
        ok = esegui("generar_perfiles.py", SCRAPING, "Profili giocatori")
        if ok:
            copia_se_esiste(
                os.path.join(SCRAPING, "perfiles_jugadores.pkl"),
                os.path.join(PREDICCION, "perfiles_jugadores.pkl")
            )

    # ═══ FASE 6 — Court Speed ════════════════════════════════════════════════
    if args.court_speed:
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

    # ═══ FASE 7 — Training modelli ═══════════════════════════════════════════
    if args.modelli:
        sezione("7️⃣  FASE 7 — Training modelli")
        modelli = [
            ("predict_xgboost.py",  "Training XGBoost"),
            ("predict_ensemble.py", "Training Ensemble (LR+RF+XGB)"),
            ("predict_LR.py",      "Training Logistic Regression"),
        ]
        for script, desc in modelli:
            esegui(script, PREDICCION, desc)

    # ═══ FASE 8 — Training ANN ═══════════════════════════════════════════════
    if args.ann:
        sezione("8️⃣  FASE 8 — Training ANN")
        ok = esegui("train_ann.py", PREDICCION, "Optuna Bayesian search + ANN")
        if ok:
            print("   🏆  Modello ANN addestrato e salvato in prediccion/")

    # ── Sincronizzazione finale ──────────────────────────────────────────────
    sezione("🔄  SINCRONIZZAZIONE FINALE")
    src_hist = os.path.join(SCRAPING,   "historialTenis.csv")
    dst_hist = os.path.join(PREDICCION, "historialTenis.csv")
    if os.path.exists(src_hist):
        shutil.copy2(src_hist, dst_hist)
        print("   ✅  historialTenis.csv sincronizzato: scraping/ → prediccion/")
    else:
        print("   ⚠️   historialTenis.csv non trovato in scraping/ — nessuna copia")

    # ── Riepilogo ────────────────────────────────────────────────────────────
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
    print("=" * W)


if __name__ == "__main__":
    main()
