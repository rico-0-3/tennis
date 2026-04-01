"""
meta_confidence.py — Meta-Confidence Score per predizioni tennis

Calcola un punteggio 0-100 che risponde alla domanda:
"Quanto posso fidarmi di questa predizione?"

La formula è deterministica (nessun nuovo ML) e combina 4 componenti:
  A. Forza della predizione (quanto è lontana da 0.5)
  B. Concordanza tra modelli (std del top-5 ensemble)
  C. Contesto storico (accuracy del segmento surface+round)
  D. Struttura del match (ranking gap, upset tendency, carico partite)

Utilizzo:
    from meta_confidence import MetaConfidenceScorer

    scorer = MetaConfidenceScorer.from_files('error_analysis_output/')
    result = scorer.score(
        prob_calibrated=0.74,
        ann_std=0.03,
        features={'surface_enc': 0, 'round_enc': 5, 'log_rank_ratio': 1.8,
                  'diff_upset_tendency': 0.05, 'diff_weeks_load': 1.0, ...}
    )
    # → {'score': 82, 'label': 'HIGH', 'reasons': [...], 'avoid': False}
"""

import json
import os


# Nomi leggibili per superficie e round
SURFACE_NAMES = {0: 'Hard', 1: 'Clay', 2: 'Grass', 3: 'Carpet'}
ROUND_NAMES   = {1: '128mi', 2: '64mi', 3: '32mi', 4: 'Ottavi',
                 5: 'Quarti', 6: 'Semifinale', 7: 'Finale'}


class MetaConfidenceScorer:
    """Calcola meta-confidence score per singola predizione."""

    def __init__(self, segment_accuracy: dict = None):
        """
        Args:
            segment_accuracy: dict {surface_round_key: accuracy_float}
                              esempio: {'Hard_5': 0.77, 'Clay_3': 0.63}
                              Se None, la Componente C viene saltata.
        """
        self.segment_accuracy = segment_accuracy or {}

    @classmethod
    def from_files(cls, analysis_dir: str):
        """Carica segment_accuracy da accuracy_by_segment.json."""
        seg_path = os.path.join(analysis_dir, 'accuracy_by_segment.json')
        if os.path.exists(seg_path):
            with open(seg_path, 'r') as f:
                data = json.load(f)
            seg_acc = data.get('by_surface_round', {})
        else:
            seg_acc = {}
        return cls(segment_accuracy=seg_acc)

    def score(self, prob_calibrated: float, ann_std: float = 0.0, features: dict = None) -> dict:
        """
        Calcola il meta-confidence score (0-100, riscalato da 70 punti reali).

        Componenti attive (max 70 pt grezzo → riscalato a 100):
          A. Forza della predizione  (0-30 pt)
          C. Contesto storico        (0-25 pt)
          D. Struttura del match     (0-15 pt)

        Nota: la Componente B (concordanza modelli) è stata rimossa perché i 5
        modelli Optuna sono addestrati sullo stesso dataset e risultano troppo
        correlati per fornire un segnale di incertezza affidabile.
        ann_std è mantenuto come parametro per compatibilità ma non influenza il punteggio.

        Args:
            prob_calibrated: probabilità calibrata del giocatore 1 (0-1)
            ann_std: ignorato (mantenuto per compatibilità)
            features: dict con le 30 feature (valori originali non scalati)

        Returns:
            dict con score (0-100), label, reasons (lista str), avoid (bool)
        """
        if features is None:
            features = {}
        reasons = []
        raw     = 0  # punteggio grezzo su 70

        # ── Componente A: Forza della predizione (0-30 pt) ─────────────────────
        margin = abs(prob_calibrated - 0.5)
        if margin > 0.30:
            pts_a = 30
            reasons.append(f"✅ Predizione netta: {prob_calibrated:.0%} (margine {margin:.0%})")
        elif margin > 0.20:
            pts_a = 20
            reasons.append(f"✅ Predizione abbastanza netta: {prob_calibrated:.0%}")
        elif margin > 0.10:
            pts_a = 10
            reasons.append(f"⚠️  Predizione incerta: {prob_calibrated:.0%} (vicina al 50/50)")
        else:
            pts_a = 0
            reasons.append(f"🔴 Match equilibratissimo: {prob_calibrated:.0%} — quasi coin flip")
        raw += pts_a

        # ── Componente C: Contesto storico (0-25 pt) ──────────────────────────
        surf_enc  = int(features.get('surface_enc', 0))
        round_enc = int(features.get('round_enc', 3))
        seg_key   = f"{surf_enc}_{round_enc}"
        seg_acc   = self.segment_accuracy.get(seg_key, None)

        surf_name  = SURFACE_NAMES.get(surf_enc, f"Sup{surf_enc}")
        round_name = ROUND_NAMES.get(round_enc, f"Round{round_enc}")

        if seg_acc is None:
            pts_c = 12  # valore neutro se mancano dati storici
            reasons.append(f"⚪ Nessun dato storico per {surf_name}/{round_name}")
        elif seg_acc > 0.75:
            pts_c = 25
            reasons.append(f"✅ Segmento affidabile: {surf_name}/{round_name} acc storica {seg_acc:.0%}")
        elif seg_acc > 0.70:
            pts_c = 15
            reasons.append(f"✅ Segmento nella media: {surf_name}/{round_name} acc {seg_acc:.0%}")
        elif seg_acc > 0.65:
            pts_c = 8
            reasons.append(f"⚠️  Segmento difficile: {surf_name}/{round_name} acc {seg_acc:.0%}")
        else:
            pts_c = 0
            reasons.append(f"🔴 Segmento problematico: {surf_name}/{round_name} acc {seg_acc:.0%}")
        raw += pts_c

        # ── Componente D: Struttura del match (0-15 pt) ────────────────────────
        pts_d = 0
        log_rank = abs(float(features.get('log_rank_ratio', 0.0)))
        upset_t  = float(features.get('diff_upset_tendency', 0.0))
        wload    = abs(float(features.get('diff_weeks_load', 0.0)))

        if log_rank > 1.5:
            pts_d += 8
            reasons.append(f"✅ Gap di ranking significativo (log_ratio={log_rank:.1f})")
        elif log_rank > 0.8:
            pts_d += 4

        if abs(upset_t) < 0.10:
            pts_d += 4
            reasons.append("✅ Nessun giocatore ha tendenza agli upset")
        elif abs(upset_t) > 0.25:
            reasons.append(f"⚠️  Tendenza agli upset rilevata (diff={upset_t:.2f})")

        if wload < 3.0:
            pts_d += 3

        raw += pts_d

        # ── Riscala 0-70 → 0-100 e calcola label ──────────────────────────────
        total = round(min(100, max(0, raw / 70 * 100)))
        if total >= 70:
            label = 'HIGH'
            avoid = False
        elif total >= 40:
            label = 'MEDIUM'
            avoid = False
        else:
            label = 'LOW'
            avoid = True

        return {
            'score':   total,
            'label':   label,
            'reasons': reasons,
            'avoid':   avoid,
            'breakdown': {
                'prediction_strength': pts_a,
                'historical_context':  pts_c,
                'match_structure':     pts_d,
            }
        }


def color_for_label(label: str) -> str:
    """Ritorna colore CSS/hex per il label Streamlit."""
    return {'HIGH': '#00CC96', 'MEDIUM': '#FFA15A', 'LOW': '#EF553B'}.get(label, '#888888')
