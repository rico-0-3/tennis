"""
filter_analysis.py — Analisi dell'impatto dei filtri sull'accuracy

Risponde a: "Se uso solo le predizioni con meta-confidence >= X, come cambia l'accuracy?"

Filtri testati:
  A. Upset tendency del favorito (sign-aware) — solo segnale strutturale
  B. Meta-Confidence Score (bucket accuracy + upset tendency)

Richiede: error_analysis_output/test_predictions.csv  (da train_ann.py)
          error_analysis_output/accuracy_by_segment.json (da error_analysis.py, opzionale)

Utilizzo:
    cd prediccion/
    python filter_analysis.py
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

ANALYSIS_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'error_analysis_output')
PREDICTIONS_CSV = os.path.join(ANALYSIS_DIR, 'test_predictions.csv')
SEGMENT_JSON    = os.path.join(ANALYSIS_DIR, 'accuracy_by_segment.json')
OUTPUT_CSV      = os.path.join(ANALYSIS_DIR, 'filter_analysis.csv')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from meta_confidence import MetaConfidenceScorer
    HAS_META = True
except ImportError:
    HAS_META = False


def analyze_filter(df: pd.DataFrame, mask, label: str) -> dict:
    sub   = df[mask]
    n     = len(sub)
    if n == 0:
        return {'label': label, 'n': 0, 'pct_kept': 0.0, 'accuracy': None, 'lift': None}
    acc  = sub['correct'].mean()
    lift = acc - df['correct'].mean()
    return {'label': label, 'n': n,
            'pct_kept': n / len(df) * 100, 'accuracy': acc, 'lift': lift}


def print_row(label, r):
    if r['n'] == 0:
        return
    sign = '+' if r['lift'] >= 0 else ''
    print(f"  {label:40s}  {r['n']:5d} match ({r['pct_kept']:5.1f}%)  "
          f"acc={r['accuracy']:.2%}  lift={sign}{r['lift']:.2%}")


def run():
    if not os.path.exists(PREDICTIONS_CSV):
        print(f"\n❌ File non trovato: {PREDICTIONS_CSV}")
        print("   Esegui prima: python train_ann.py")
        sys.exit(1)

    df = pd.read_csv(PREDICTIONS_CSV)
    prob_col = 'p_calibrated' if 'p_calibrated' in df.columns else 'p_raw'
    df['p_used'] = df[prob_col].clip(0, 1)
    df['p_fav']  = df['p_used'].apply(lambda p: p if p >= 0.5 else 1 - p)

    base_acc = df['correct'].mean()
    n_total  = len(df)

    print(f"\n{'='*70}")
    print(f"  ANALISI FILTRI DI CONFIDENZA")
    print(f"{'='*70}")
    print(f"  Test set: {n_total:,} match")
    print(f"  Accuracy base (nessun filtro): {base_acc:.2%}")
    print(f"{'='*70}")

    rows = [{'Filtro': 'BASE (nessuno)', 'N match': n_total,
             '% kept': 100.0, 'Accuracy': base_acc, 'Lift vs base': 0.0}]

    # ── Filtro A: Upset tendency del FAVORITO (sign-aware) ────────────────────
    # upset_risk > 0 = favorito inaffidabile | upset_risk < 0 = favorito solido
    if 'diff_upset_tendency' in df.columns:
        df['upset_risk'] = df['diff_upset_tendency'] * np.sign(df['p_used'] - 0.5)

        print(f"\n── FILTRO A: Upset tendency del favorito (sign-aware) ──────────────")
        print(f"  (upset_risk < 0 = favorito solido, > 0 = favorito inaffidabile)")

        for thr in [-0.20, -0.10, 0.00, 0.10, 0.20]:
            mask = df['upset_risk'] <= thr
            r    = analyze_filter(df, mask, f"upset_risk ≤ {thr:+.2f}")
            if r['n'] >= 30:
                print_row(f"upset_risk <= {thr:+.2f}", r)
                rows.append({'Filtro': f"A: upset_risk <= {thr:+.2f}",
                             'N match': r['n'], '% kept': round(r['pct_kept'], 1),
                             'Accuracy': r['accuracy'], 'Lift vs base': r['lift']})

        # Distribuzione upset_risk per capire i bucket
        print(f"\n  Distribuzione upset_risk:")
        for lo, hi in [(-1, -0.20), (-0.20, -0.10), (-0.10, 0.10), (0.10, 0.20), (0.20, 1)]:
            sub = df[(df['upset_risk'] >= lo) & (df['upset_risk'] < hi)]
            if len(sub) >= 10:
                print(f"    [{lo:+.2f}, {hi:+.2f}):  {len(sub):5d} match  "
                      f"acc={sub['correct'].mean():.2%}")

    # ── Filtro B: Meta-Confidence Score ───────────────────────────────────────
    if HAS_META:
        bucket_acc  = {}
        segment_acc = {}
        if os.path.exists(SEGMENT_JSON):
            with open(SEGMENT_JSON) as f:
                seg_data = json.load(f)
            bucket_acc  = seg_data.get('by_confidence_bucket', {})
            segment_acc = seg_data.get('by_surface_round', {})

        has_buckets = bool(bucket_acc)
        scorer = MetaConfidenceScorer(bucket_accuracy=bucket_acc,
                                      segment_accuracy=segment_acc)

        print(f"\n── FILTRO B: Meta-Confidence Score {'(dati storici presenti)' if has_buckets else '(NO dati storici — solo upset)'}  ──")
        print(f"  Calcolo score per {n_total:,} match...")

        scores = []
        for _, row in df.iterrows():
            res = scorer.score(
                prob_calibrated=float(row['p_used']),
                features=row.to_dict()
            )
            scores.append(res['score'])
        df['meta_score'] = scores

        for thr in [20, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80]:
            mask = df['meta_score'] >= thr
            r    = analyze_filter(df, mask, f"meta >= {thr}")
            if r['n'] >= 30:
                print_row(f"meta_score >= {thr:3d}", r)
                rows.append({'Filtro': f"B: meta_score >= {thr}",
                             'N match': r['n'], '% kept': round(r['pct_kept'], 1),
                             'Accuracy': r['accuracy'], 'Lift vs base': r['lift']})

        # Distribuzione score: accuracy per fascia
        print(f"\n  Distribuzione Meta-Confidence Score:")
        bins   = [0, 10, 20, 30, 40, 50, 60, 70, 80, 101]
        labels = ['0-10','10-20','20-30','30-40','40-50','50-60','60-70','70-80','80+']
        df['score_bin'] = pd.cut(df['meta_score'], bins=bins, labels=labels, right=False)
        for lbl in labels:
            sub = df[df['score_bin'] == lbl]
            if len(sub) == 0:
                continue
            acc = sub['correct'].mean()
            bar = '█' * int(acc * 30)
            print(f"    Score {lbl:6s}: {len(sub):5d} match ({len(sub)/n_total*100:4.1f}%)  "
                  f"acc={acc:.2%}  {bar}")

    # ── Riepilogo finale ──────────────────────────────────────────────────────
    df_rows = pd.DataFrame(rows)
    df_rows['Accuracy %'] = (df_rows['Accuracy'] * 100).round(2)
    df_rows['Lift %']     = (df_rows['Lift vs base'] * 100).round(2)
    df_rows.to_csv(OUTPUT_CSV, index=False)

    print(f"\n{'='*70}")
    print(f"  TOP 5 FILTRI PER LIFT (min 100 match)")
    print(f"{'='*70}")
    top = (df_rows[df_rows['N match'] >= 100]
           .sort_values('Lift vs base', ascending=False)
           .head(5))
    for _, r in top.iterrows():
        sign = '+' if r['Lift %'] >= 0 else ''
        print(f"  {r['Filtro']:40s}  acc={r['Accuracy %']:.2f}%  "
              f"lift={sign}{r['Lift %']:.2f}%  kept={r['% kept']:.1f}%")

    print(f"\n✅ Risultati salvati in: {OUTPUT_CSV}")
    return df_rows


if __name__ == '__main__':
    run()
