"""
error_analysis.py — Analisi degli errori del modello tennis

Script standalone da eseguire DOPO train_ann.py.
Richiede: error_analysis_output/test_predictions.csv

IMPORTANTE: analizza TUTTI i bucket di confidenza, non solo quelli alti.
La domanda chiave è: "per le predizioni intorno al 60%, quali condizioni
le rendono più affidabili del solito?" Trovare questi pattern è
più utile che sapere solo che le predizioni >85% sono affidabili.

Output:
    error_analysis_output/accuracy_by_segment.json   — accuracy per segmento
    error_analysis_output/failure_tree.pkl            — decision tree per errori
    error_analysis_output/bucket_patterns.json        — pattern per bucket confidenza
    error_analysis_output/report.txt                  — report testuale

Utilizzo:
    cd prediccion/
    python error_analysis.py
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import joblib
warnings.filterwarnings("ignore")

ANALYSIS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'error_analysis_output')
PREDICTIONS_CSV = os.path.join(ANALYSIS_DIR, 'test_predictions.csv')


# Nomi leggibili
SURFACE_NAMES = {0: 'Hard', 1: 'Clay', 2: 'Grass', 3: 'Carpet'}
ROUND_NAMES   = {1: '128mi', 2: '64mi', 3: '32mi', 4: 'Ottavi',
                 5: 'Quarti', 6: 'Semifinale', 7: 'Finale',
                 8: 'Round Robin'}
LEVEL_NAMES   = {0: 'E', 1: 'S', 2: 'C', 3: 'A', 4: 'M/F', 5: 'G'}


def run_analysis():
    if not os.path.exists(PREDICTIONS_CSV):
        print(f"❌ File non trovato: {PREDICTIONS_CSV}")
        print("   Esegui prima train_ann.py per generare le predizioni.")
        sys.exit(1)

    df = pd.read_csv(PREDICTIONS_CSV)
    print(f"✅ Caricato: {len(df):,} righe da test set")

    # Usa p_calibrated se disponibile, altrimenti p_raw
    prob_col = 'p_calibrated' if 'p_calibrated' in df.columns else 'p_raw'
    df['p_used'] = df[prob_col]

    # La probabilità usata è sempre da J1's perspective (target=1 = J1 ha vinto)
    # Per l'analisi vogliamo sempre P(vincitore) >= 0.5
    df['p_fav']     = df['p_used'].clip(0.5, 1.0)  # confidenza verso il favorito
    df['predicted'] = (df['p_used'] >= 0.5).astype(int)
    df['correct']   = (df['predicted'] == df['target']).astype(int)

    overall_acc = df['correct'].mean()
    print(f"   Accuracy complessiva: {overall_acc:.2%}")

    results = {
        'overall_accuracy': float(overall_acc),
        'n_samples':        int(len(df)),
        'by_surface':       {},
        'by_round':         {},
        'by_level':         {},
        'by_confidence_bucket': {},
        'by_surface_round': {},
        'bucket_patterns':  {},
    }

    # ── 1. Accuracy per superficie ─────────────────────────────────────────────
    print("\n📊 Accuracy per superficie:")
    for surf_enc, surf_name in SURFACE_NAMES.items():
        sub = df[df['surface_enc'] == surf_enc]
        if len(sub) < 30:
            continue
        acc = sub['correct'].mean()
        n   = len(sub)
        results['by_surface'][surf_name] = {'accuracy': float(acc), 'n': int(n)}
        bar = '█' * int(acc * 30)
        print(f"   {surf_name:8s}: {acc:.2%}  ({n:4d} match)  {bar}")

    # ── 2. Accuracy per round ──────────────────────────────────────────────────
    print("\n📊 Accuracy per round:")
    for r_enc in sorted(df['round_enc'].unique()):
        sub = df[df['round_enc'] == r_enc]
        if len(sub) < 20:
            continue
        acc = sub['correct'].mean()
        r_name = ROUND_NAMES.get(int(r_enc), f"R{r_enc}")
        results['by_round'][r_name] = {'accuracy': float(acc), 'n': int(len(sub))}
        print(f"   {r_name:18s}: {acc:.2%}  ({len(sub):4d} match)")

    # ── 3. Accuracy per livello torneo ────────────────────────────────────────
    print("\n📊 Accuracy per livello torneo:")
    for lev_enc in sorted(df['tourney_level'].unique()):
        sub = df[df['tourney_level'] == lev_enc]
        if len(sub) < 20:
            continue
        acc = sub['correct'].mean()
        l_name = LEVEL_NAMES.get(int(lev_enc), f"L{lev_enc}")
        results['by_level'][l_name] = {'accuracy': float(acc), 'n': int(len(sub))}
        print(f"   {l_name:6s}: {acc:.2%}  ({len(sub):4d} match)")

    # ── 4. Accuracy per bucket di confidenza ──────────────────────────────────
    print("\n📊 Accuracy per bucket di confidenza (favorito):")
    bucket_edges = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 1.01]
    bucket_labels = ['50-55%', '55-60%', '60-65%', '65-70%', '70-75%', '75-80%', '80-90%', '90%+']
    for i, (lo, hi) in enumerate(zip(bucket_edges[:-1], bucket_edges[1:])):
        sub = df[(df['p_fav'] >= lo) & (df['p_fav'] < hi)]
        if len(sub) < 10:
            continue
        acc = sub['correct'].mean()
        lbl = bucket_labels[i]
        results['by_confidence_bucket'][lbl] = {
            'accuracy': float(acc), 'n': int(len(sub)),
            'p_low': float(lo), 'p_high': float(hi)
        }
        # Flag: è l'accuracy maggiore della media? (indica calibrazione buona)
        flag = "✅" if acc > overall_acc else "⚠️ "
        print(f"   {lbl:10s}: {acc:.2%}  ({len(sub):4d} match) {flag}")

    # ── 5. Accuracy per segmento superficie+round ─────────────────────────────
    print("\n📊 Accuracy per segmento (superficie+round):")
    for surf_enc in sorted(df['surface_enc'].unique()):
        for r_enc in sorted(df['round_enc'].unique()):
            sub = df[(df['surface_enc'] == surf_enc) & (df['round_enc'] == r_enc)]
            if len(sub) < 20:
                continue
            acc = sub['correct'].mean()
            seg_key = f"{int(surf_enc)}_{int(r_enc)}"
            surf_name  = SURFACE_NAMES.get(int(surf_enc), f"S{surf_enc}")
            round_name = ROUND_NAMES.get(int(r_enc), f"R{r_enc}")
            results['by_surface_round'][seg_key] = float(acc)
            flag = "✅" if acc > overall_acc + 0.02 else ("🔴" if acc < overall_acc - 0.02 else "  ")
            print(f"   {surf_name:6s} / {round_name:16s}: {acc:.2%}  ({len(sub):3d}) {flag}")

    # ── 6. Pattern per bucket di confidenza (CHIAVE) ──────────────────────────
    # Per ogni bucket trova quali feature dividono meglio "corretto" vs "sbagliato"
    # Questo risponde a: "per le predizioni intorno al 60%, cosa le rende affidabili?"
    print("\n🔍 Pattern per bucket di confidenza:")
    try:
        from sklearn.tree import DecisionTreeClassifier, export_text
        FEATURES_FOR_TREE = [
            'log_rank_ratio', 'diff_elo', 'diff_skill',
            'surface_enc', 'round_enc', 'tourney_level',
            'diff_upset_tendency', 'diff_recent_form', 'diff_streak',
            'diff_weeks_load', 'diff_fatigue', 'diff_h2h',
        ]
        available_feats = [f for f in FEATURES_FOR_TREE if f in df.columns]

        for i, (lo, hi) in enumerate(zip(bucket_edges[:-1], bucket_edges[1:])):
            lbl = bucket_labels[i]
            sub = df[(df['p_fav'] >= lo) & (df['p_fav'] < hi)].copy()
            if len(sub) < 60:
                results['bucket_patterns'][lbl] = {'note': 'dati insufficienti'}
                continue

            X_sub = sub[available_feats].fillna(0)
            y_sub = sub['correct']

            # Albero superficiale: trova le condizioni che dividono corretto/sbagliato
            tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, random_state=42)
            tree.fit(X_sub, y_sub)

            tree_text = export_text(tree, feature_names=available_feats, max_depth=3)
            acc_sub = y_sub.mean()

            # Estrai le foglie più "affidabili" (leaf con alta accuracy e almeno 20 campioni)
            leaf_info = _extract_leaf_info(tree, X_sub, y_sub, available_feats)

            results['bucket_patterns'][lbl] = {
                'n': int(len(sub)),
                'accuracy': float(acc_sub),
                'tree_text': tree_text,
                'reliable_leaves': leaf_info['reliable'],
                'unreliable_leaves': leaf_info['unreliable'],
            }
            print(f"\n   Bucket {lbl} ({len(sub)} match, acc={acc_sub:.2%}):")
            for leaf in leaf_info['reliable'][:2]:
                print(f"     ✅ AFFIDABILE: {leaf['path']} → acc={leaf['accuracy']:.0%} (n={leaf['n']})")
            for leaf in leaf_info['unreliable'][:2]:
                print(f"     🔴 INAFFIDABILE: {leaf['path']} → acc={leaf['accuracy']:.0%} (n={leaf['n']})")

    except ImportError:
        print("   ⚠️  sklearn non disponibile per decision tree")
        results['bucket_patterns'] = {}

    # ── 7. Decision tree globale per errori (per meta_confidence) ────────────
    try:
        from sklearn.tree import DecisionTreeClassifier
        X_all = df[available_feats].fillna(0)
        y_all = df['correct']
        global_tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=50, random_state=42)
        global_tree.fit(X_all, y_all)
        joblib.dump(global_tree, os.path.join(ANALYSIS_DIR, 'failure_tree.pkl'))
        print(f"\n   ✅ failure_tree.pkl salvato")
    except Exception as e:
        print(f"   ⚠️  failure_tree non salvato: {e}")

    # ── 8. Salva JSON ──────────────────────────────────────────────────────────
    out_json = os.path.join(ANALYSIS_DIR, 'accuracy_by_segment.json')
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n✅ accuracy_by_segment.json salvato")

    # ── 9. Report testuale ─────────────────────────────────────────────────────
    _write_report(results, os.path.join(ANALYSIS_DIR, 'report.txt'))
    print("✅ report.txt salvato")

    return results


def _extract_leaf_info(tree, X, y, feature_names):
    """Estrae info sulle foglie del decision tree."""
    from sklearn.tree import _tree

    leaf_data = []
    tree_ = tree.tree_
    feature_name = [feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined"
                    for i in tree_.feature]

    def traverse(node, path):
        if tree_.feature[node] == _tree.TREE_UNDEFINED:
            # Foglia
            n_node = int(tree_.n_node_samples[node])
            values = tree_.value[node][0]
            n_correct = int(values[1]) if len(values) > 1 else 0
            acc = n_correct / n_node if n_node > 0 else 0.5
            if n_node >= 15:
                leaf_data.append({
                    'path': ' & '.join(path) if path else 'root',
                    'n': n_node,
                    'accuracy': float(acc),
                })
            return

        fname = feature_name[node]
        threshold = tree_.threshold[node]
        traverse(tree_.children_left[node],  path + [f"{fname} ≤ {threshold:.2f}"])
        traverse(tree_.children_right[node], path + [f"{fname} > {threshold:.2f}"])

    traverse(0, [])

    overall_acc = y.mean()
    reliable   = sorted([l for l in leaf_data if l['accuracy'] > float(overall_acc) + 0.05],
                         key=lambda x: -x['accuracy'])
    unreliable = sorted([l for l in leaf_data if l['accuracy'] < float(overall_acc) - 0.05],
                         key=lambda x: x['accuracy'])
    return {'reliable': reliable, 'unreliable': unreliable}


def _write_report(results, path):
    lines = [
        "=" * 70,
        "REPORT ANALISI ERRORI — MODELLO TENNIS ANN",
        "=" * 70,
        f"\nAccuracy complessiva (test set): {results['overall_accuracy']:.2%}",
        f"Campioni nel test set: {results['n_samples']:,}",
        "\n── ACCURACY PER SUPERFICIE ──",
    ]
    for name, v in results['by_surface'].items():
        lines.append(f"  {name:8s}: {v['accuracy']:.2%}  (n={v['n']})")

    lines.append("\n── ACCURACY PER ROUND ──")
    for name, v in results['by_round'].items():
        lines.append(f"  {name:18s}: {v['accuracy']:.2%}  (n={v['n']})")

    lines.append("\n── ACCURACY PER BUCKET CONFIDENZA ──")
    for name, v in results['by_confidence_bucket'].items():
        lines.append(f"  {name:10s}: {v['accuracy']:.2%}  (n={v['n']})")

    lines.append("\n── PATTERN PER BUCKET (TOP-2 FOGLIE AFFIDABILI) ──")
    for bucket, pat in results.get('bucket_patterns', {}).items():
        if 'reliable_leaves' not in pat:
            continue
        lines.append(f"\n  Bucket {bucket} (acc={pat['accuracy']:.0%}, n={pat['n']}):")
        for leaf in pat['reliable_leaves'][:2]:
            lines.append(f"    ✅ {leaf['path']} → {leaf['accuracy']:.0%}")
        for leaf in pat['unreliable_leaves'][:2]:
            lines.append(f"    🔴 {leaf['path']} → {leaf['accuracy']:.0%}")

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


if __name__ == '__main__':
    run_analysis()
