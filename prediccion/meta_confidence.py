"""
meta_confidence.py — Meta-Confidence Score per predizioni tennis

Punteggio 0-100 basato su due segnali:

  C. Bucket accuracy storica  (0-60 pt)
     Quanto era accurato il modello storicamente quando ha predetto
     in questo range di probabilità? (es. quando dice ~65%, quante volte aveva ragione?)

  U. Upset tendency del favorito  (0-40 pt)  — SIGN-AWARE
     Il favorito tende a perdere contro avversari peggio piazzati?
     diff_upset_tendency = upt_j1 - upt_j2
     Se J1 è il favorito (prob > 0.5): upset_risk = diff (positivo = J1 inaffidabile)
     Se J2 è il favorito (prob < 0.5): upset_risk = -diff (positivo = J2 inaffidabile)

Normalizzazione adattiva:
  - Se i dati storici (bucket accuracy) sono presenti: score = (C + U) / 100 * 100
  - Se mancano: score = U / 40 * 60  (massimo 60 = MEDIUM, segnale incompleto)

Labels: HIGH >= 65 | MEDIUM 35-64 | LOW < 35
"""

import json
import os

SURFACE_NAMES = {0: 'Hard', 1: 'Clay', 2: 'Grass', 3: 'Carpet'}
ROUND_NAMES   = {1: '128mi', 2: '64mi', 3: '32mi', 4: 'Ottavi',
                 5: 'Quarti', 6: 'Semifinale', 7: 'Finale'}

# Bordi dei bucket di confidenza (basati su p_fav = max(p, 1-p))
_BUCKET_EDGES  = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 1.01]
_BUCKET_LABELS = ['50-55%', '55-60%', '60-65%', '65-70%',
                  '70-75%', '75-80%', '80-90%', '90%+']


def _get_bucket_label(p_fav: float) -> str:
    for lo, hi, lbl in zip(_BUCKET_EDGES[:-1], _BUCKET_EDGES[1:], _BUCKET_LABELS):
        if lo <= p_fav < hi:
            return lbl
    return _BUCKET_LABELS[-1]


class MetaConfidenceScorer:

    def __init__(self, bucket_accuracy: dict = None, segment_accuracy: dict = None):
        """
        Args:
            bucket_accuracy: dict {label: {'accuracy': float, 'n': int}}
                             da accuracy_by_segment.json → by_confidence_bucket
            segment_accuracy: dict {surf_round_key: float}  (tenuto per compatibilità, non usato)
        """
        self.bucket_accuracy  = bucket_accuracy or {}
        self.segment_accuracy = segment_accuracy or {}  # non usato nel calcolo

    @classmethod
    def from_files(cls, analysis_dir: str):
        seg_path = os.path.join(analysis_dir, 'accuracy_by_segment.json')
        bucket_acc  = {}
        segment_acc = {}
        if os.path.exists(seg_path):
            with open(seg_path, 'r') as f:
                data = json.load(f)
            bucket_acc  = data.get('by_confidence_bucket', {})
            segment_acc = data.get('by_surface_round', {})
        return cls(bucket_accuracy=bucket_acc, segment_accuracy=segment_acc)

    def score(self, prob_calibrated: float, ann_std: float = 0.0, features: dict = None) -> dict:
        """
        Args:
            prob_calibrated: probabilità calibrata J1 (0-1)
            ann_std:         ignorato (mantenuto per compatibilità)
            features:        dict con le 30 feature del match

        Returns:
            dict: score (0-100), label, reasons, avoid, breakdown
        """
        if features is None:
            features = {}
        reasons = []

        p_fav = max(prob_calibrated, 1.0 - prob_calibrated)

        # ── Componente C: Bucket accuracy storica (0-60 pt) ───────────────────
        bucket_lbl = _get_bucket_label(p_fav)
        bucket_data = self.bucket_accuracy.get(bucket_lbl, None)
        has_historical = bucket_data is not None and bucket_data.get('n', 0) >= 30

        if not has_historical:
            pts_c = None  # segnale assente
            surf_enc  = int(features.get('surface_enc', 0))
            round_enc = int(features.get('round_enc', 3))
            surf_name  = SURFACE_NAMES.get(surf_enc, f"Sup{surf_enc}")
            round_name = ROUND_NAMES.get(round_enc, f"Round{round_enc}")
            reasons.append(f"⚪ Dati storici non disponibili ({surf_name}/{round_name}) — score ridotto")
        else:
            acc = bucket_data['accuracy']
            n   = bucket_data['n']
            if acc >= 0.78:
                pts_c = 60
                reasons.append(f"✅ Bucket {bucket_lbl}: modello molto affidabile storicamente ({acc:.0%}, n={n})")
            elif acc >= 0.75:
                pts_c = 45
                reasons.append(f"✅ Bucket {bucket_lbl}: buona affidabilità storica ({acc:.0%}, n={n})")
            elif acc >= 0.72:
                pts_c = 30
                reasons.append(f"✅ Bucket {bucket_lbl}: affidabilità nella media ({acc:.0%}, n={n})")
            elif acc >= 0.69:
                pts_c = 15
                reasons.append(f"⚠️  Bucket {bucket_lbl}: affidabilità sotto la media ({acc:.0%}, n={n})")
            else:
                pts_c = 0
                reasons.append(f"🔴 Bucket {bucket_lbl}: modello poco affidabile in questo range ({acc:.0%}, n={n})")

        # ── Componente U: Upset tendency del favorito (0-40 pt) ──────────────
        upset_diff = float(features.get('diff_upset_tendency', 0.0))
        # upset_risk > 0 = il FAVORITO ha alta tendenza agli upset (cattivo segnale)
        # upset_risk < 0 = il FAVORITO ha bassa tendenza (buon segnale)
        sign = 1.0 if prob_calibrated >= 0.5 else -1.0
        upset_risk = upset_diff * sign

        if upset_risk < -0.20:
            pts_u = 40
            reasons.append(f"✅ Il favorito è molto solido contro avversari peggio piazzati (risk={upset_risk:.2f})")
        elif upset_risk < -0.10:
            pts_u = 30
            reasons.append(f"✅ Il favorito ha bassa tendenza agli upset (risk={upset_risk:.2f})")
        elif upset_risk <= 0.10:
            pts_u = 20
            reasons.append(f"⚪ Tendenza agli upset neutra (risk={upset_risk:.2f})")
        elif upset_risk <= 0.20:
            pts_u = 10
            reasons.append(f"⚠️  Il favorito ha qualche tendenza agli upset (risk={upset_risk:.2f})")
        else:
            pts_u = 0
            reasons.append(f"🔴 Il favorito tende a perdere contro avversari peggio piazzati (risk={upset_risk:.2f})")

        # ── Score finale con normalizzazione adattiva ─────────────────────────
        if has_historical:
            # Entrambi i segnali disponibili → score su 100
            raw   = pts_c + pts_u        # max 100
            total = round(min(100, max(0, raw)))
        else:
            # Solo upset tendency → normalizza su 60 max (segnale incompleto)
            total = round(min(60, max(0, pts_u / 40 * 60)))

        if total >= 65:
            label = 'HIGH'
            avoid = False
        elif total >= 35:
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
                'bucket_accuracy': pts_c if pts_c is not None else 'N/A',
                'upset_tendency':  pts_u,
                'has_historical':  has_historical,
            }
        }


def color_for_label(label: str) -> str:
    return {'HIGH': '#00CC96', 'MEDIUM': '#FFA15A', 'LOW': '#EF553B'}.get(label, '#888888')
