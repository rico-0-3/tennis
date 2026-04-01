"""
filter_analysis.py — Analisi dell'impatto dei filtri di confidenza sull'accuracy

Risponde a: "Se uso solo le predizioni con meta-confidence >= X, come cambia l'accuracy?"
Confronta modello base vs. modello base + filtri a varie soglie.

Richiede: error_analysis_output/test_predictions.csv  (generato da train_ann.py)
          error_analysis_output/accuracy_by_segment.json (generato da error_analysis.py, opzionale)

Utilizzo:
    cd prediccion/
    python filter_analysis.py

Output (console + CSV):
    error_analysis_output/filter_analysis.csv
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

ANALYSIS_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'error_analysis_output')
PREDICTIONS_CSV  = os.path.join(ANALYSIS_DIR, 'test_predictions.csv')
SEGMENT_JSON     = os.path.join(ANALYSIS_DIR, 'accuracy_by_segment.json')
OUTPUT_CSV       = os.path.join(ANALYSIS_DIR, 'filter_analysis.csv')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from meta_confidence import MetaConfidenceScorer
    HAS_META = True
except ImportError:
    HAS_META = False


def compute_meta_scores(df: pd.DataFrame, scorer) -> np.ndarray:
    """Calcola meta-confidence score per ogni riga del DataFrame."""
    scores = []
    for _, row in df.iterrows():
        prob = float(row['p_calibrated'])
        std  = float(row.get('ann_std', 0.0))
        feat = row.to_dict()
        res  = scorer.score(prob_calibrated=prob, ann_std=std, features=feat)
        scores.append(res['score'])
    return np.array(scores)


def analyze_filter(df: pd.DataFrame, mask: np.ndarray, label: str) -> dict:
    """Calcola statistiche per un subset filtrato."""
    sub = df[mask]
    n   = len(sub)
    if n == 0:
        return {'label': label, 'n': 0, 'pct_kept': 0.0, 'accuracy': None, 'lift': None}
    acc   = sub['correct'].mean()
    base  = df['correct'].mean()
    lift  = acc - base
    return {
        'label':    label,
        'n':        n,
        'pct_kept': n / len(df) * 100,
        'accuracy': acc,
        'lift':     lift,
    }


def run():
    if not os.path.exists(PREDICTIONS_CSV):
        print(f"❌ File non trovato: {PREDICTIONS_CSV}")
        print("   Esegui prima: python train_ann.py")
        sys.exit(1)

    df = pd.read_csv(PREDICTIONS_CSV)
    prob_col = 'p_calibrated' if 'p_calibrated' in df.columns else 'p_raw'
    df['p_used'] = df[prob_col].clip(0, 1)
    df['p_fav']  = df['p_used'].apply(lambda p: p if p >= 0.5 else 1 - p)
    if 'ann_std' not in df.columns:
        df['ann_std'] = 0.0

    base_acc = df['correct'].mean()
    n_total  = len(df)
    print(f"\n{'='*65}")
    print(f"  ANALISI FILTRI DI CONFIDENZA")
    print(f"{'='*65}")
    print(f"  Test set: {n_total:,} match")
    print(f"  Accuracy base (nessun filtro): {base_acc:.2%}")
    print(f"{'='*65}\n")

    rows = []

    # ── Baseline ──────────────────────────────────────────────────────────────
    rows.append({
        'Filtro': 'BASE (nessuno)',
        'N match': n_total,
        '% kept': 100.0,
        'Accuracy': base_acc,
        'Lift vs base': 0.0,
    })

    # ── Filtro A: soglia su |p - 0.5| (forza predizione) ─────────────────────
    print("── FILTRO A: Forza predizione |p - 0.5| ─────────────────────────")
    thresholds_p = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    for thr in thresholds_p:
        mask = (df['p_fav'] - 0.5) >= thr
        r    = analyze_filter(df, mask, f"|p-0.5| ≥ {thr:.0%}")
        if r['n'] > 0:
            lift_str = f"+{r['lift']:+.2%}" if r['lift'] >= 0 else f"{r['lift']:+.2%}"
            print(f"  |p-0.5| ≥ {thr:.0%}:  {r['n']:5d} match ({r['pct_kept']:5.1f}%)  "
                  f"acc={r['accuracy']:.2%}  lift={lift_str}")
            rows.append({
                'Filtro': f"A: |p-0.5| >= {thr:.0%}",
                'N match': r['n'],
                '% kept': round(r['pct_kept'], 1),
                'Accuracy': r['accuracy'],
                'Lift vs base': r['lift'],
            })

    # ── Filtro B: soglia su ann_std (concordanza modelli) ─────────────────────
    if df['ann_std'].max() > 0:
        print("\n── FILTRO B: Concordanza modelli (ann_std) ───────────────────────")
        thresholds_std = [0.12, 0.10, 0.08, 0.06, 0.04]
        for thr in thresholds_std:
            mask = df['ann_std'] <= thr
            r    = analyze_filter(df, mask, f"std ≤ {thr:.2f}")
            if r['n'] > 0:
                lift_str = f"+{r['lift']:+.2%}" if r['lift'] >= 0 else f"{r['lift']:+.2%}"
                print(f"  std ≤ {thr:.2f}:  {r['n']:5d} match ({r['pct_kept']:5.1f}%)  "
                      f"acc={r['accuracy']:.2%}  lift={lift_str}")
                rows.append({
                    'Filtro': f"B: ann_std <= {thr:.2f}",
                    'N match': r['n'],
                    '% kept': round(r['pct_kept'], 1),
                    'Accuracy': r['accuracy'],
                    'Lift vs base': r['lift'],
                })

    # ── Filtro C: combinato p + std ───────────────────────────────────────────
    if df['ann_std'].max() > 0:
        print("\n── FILTRO C: Combinato (|p-0.5| + std) ──────────────────────────")
        combos = [
            (0.10, 0.10), (0.10, 0.08), (0.15, 0.10), (0.15, 0.08),
            (0.20, 0.10), (0.20, 0.08), (0.25, 0.08),
        ]
        for p_thr, s_thr in combos:
            mask = ((df['p_fav'] - 0.5) >= p_thr) & (df['ann_std'] <= s_thr)
            r    = analyze_filter(df, mask, f"|p|≥{p_thr:.0%} & std≤{s_thr:.2f}")
            if r['n'] >= 30:
                lift_str = f"+{r['lift']:+.2%}" if r['lift'] >= 0 else f"{r['lift']:+.2%}"
                print(f"  |p|≥{p_thr:.0%} & std≤{s_thr:.2f}:  {r['n']:5d} match ({r['pct_kept']:5.1f}%)  "
                      f"acc={r['accuracy']:.2%}  lift={lift_str}")
                rows.append({
                    'Filtro': f"C: |p|>={p_thr:.0%} & std<={s_thr:.2f}",
                    'N match': r['n'],
                    '% kept': round(r['pct_kept'], 1),
                    'Accuracy': r['accuracy'],
                    'Lift vs base': r['lift'],
                })

    # ── Filtro D: Meta-Confidence score (se disponibile) ─────────────────────
    if HAS_META:
        print("\n── FILTRO D: Meta-Confidence Score ──────────────────────────────")
        seg_acc = {}
        if os.path.exists(SEGMENT_JSON):
            with open(SEGMENT_JSON) as f:
                seg_data = json.load(f)
            seg_acc = seg_data.get('by_surface_round', {})

        scorer = MetaConfidenceScorer(segment_accuracy=seg_acc)
        print("   Calcolo meta-confidence per ogni match del test set...")
        meta_scores = compute_meta_scores(df, scorer)
        df['meta_score'] = meta_scores

        thresholds_meta = [30, 40, 50, 55, 60, 65, 70, 75, 80]
        for thr in thresholds_meta:
            mask = df['meta_score'] >= thr
            r    = analyze_filter(df, mask, f"meta ≥ {thr}")
            if r['n'] >= 30:
                lift_str = f"+{r['lift']:+.2%}" if r['lift'] >= 0 else f"{r['lift']:+.2%}"
                print(f"  meta ≥ {thr:3d}:  {r['n']:5d} match ({r['pct_kept']:5.1f}%)  "
                      f"acc={r['accuracy']:.2%}  lift={lift_str}")
                rows.append({
                    'Filtro': f"D: meta_score >= {thr}",
                    'N match': r['n'],
                    '% kept': round(r['pct_kept'], 1),
                    'Accuracy': r['accuracy'],
                    'Lift vs base': r['lift'],
                })

        # ── Distribuzione score ─────────────────────────────────────────────
        print("\n── Distribuzione Meta-Confidence Score ──────────────────────────")
        bins   = [0, 20, 30, 40, 50, 60, 70, 80, 101]
        labels = ['0-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80+']
        df['score_bin'] = pd.cut(df['meta_score'], bins=bins, labels=labels, right=False)
        for lbl in labels:
            sub = df[df['score_bin'] == lbl]
            if len(sub) == 0:
                continue
            acc = sub['correct'].mean()
            print(f"  Score {lbl:6s}: {len(sub):5d} match ({len(sub)/n_total*100:4.1f}%)  acc={acc:.2%}")

    # ── Riepilogo: migliori filtri per lift ───────────────────────────────────
    df_rows = pd.DataFrame(rows)
    df_rows['Accuracy %'] = (df_rows['Accuracy'] * 100).round(2)
    df_rows['Lift %']     = (df_rows['Lift vs base'] * 100).round(2)
    df_rows.to_csv(OUTPUT_CSV, index=False)

    print(f"\n{'='*65}")
    print("  TOP 5 FILTRI PER LIFT (min 100 match)")
    print(f"{'='*65}")
    top = (df_rows[df_rows['N match'] >= 100]
           .sort_values('Lift vs base', ascending=False)
           .head(5))
    for _, r in top.iterrows():
        print(f"  {r['Filtro']:35s}  acc={r['Accuracy %']:.2f}%  "
              f"lift=+{r['Lift %']:.2f}%  kept={r['% kept']:.1f}%")

    print(f"\n✅ Risultati salvati in: {OUTPUT_CSV}")
    return df_rows


if __name__ == '__main__':
    run()
