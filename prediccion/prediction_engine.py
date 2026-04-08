"""
prediccion/prediction_engine.py
================================
Modulo UNICO per la logica di predizione ATP.
Importato sia da:
  - pages/1_🔮_Predictor_en_Vivo.py  (Streamlit UI)
  - prediccion/predecir_proximas.py   (batch automatico)

In questo modo qualsiasi modifica alla logica si riflette su entrambi.
"""

import numpy as np
import torch
import torch.nn as nn

# ── Costanti ──────────────────────────────────────────────────────────────────

ANN_FEATURES = [
    'log_rank_ratio', 'log_pts_ratio',
    'diff_elo', 'diff_elo_overall',
    'diff_streak', 'diff_recent_form',
    'surface_enc', 'tourney_level', 'round_enc',
    'is_best_of_5',
    'diff_h2h', 'diff_h2h_surface',
    'diff_skill', 'diff_momentum',
    'diff_fatigue', 'diff_days_since_last',
    'diff_weeks_load',
    'diff_ace', 'diff_1st_pct', 'diff_1st_won',
    'diff_2nd_won', 'diff_bp_saved',
    'diff_return_pct', 'diff_bp_conv', 'diff_return_1st',
    'diff_home',
    'diff_opponent_quality',
    'diff_upset_tendency',
    'diff_late_round_wr',
    'level_weight',
]  # 30 feature (v5.1)


SURFACE_MAP = {'Hard': 0, 'Clay': 1, 'Grass': 2}

# Mapping livello torneo → valore numerico per tourney_level feature
LEVEL_LABEL = {
    'Grand Slam': 5, 'Masters 1000': 4,
    'ATP 500': 3, 'ATP 250': 3, 'Challenger': 2,
}

# Moltiplicatore per level_weight feature
LEVEL_MULT_LABEL = {
    'Grand Slam': 2.0, 'Masters 1000': 1.5,
    'ATP 500': 1.0, 'ATP 250': 1.0, 'Challenger': 0.8,
}

# Mapping turno → valore numerico per round_enc feature.
# *** CANONICAL *** — lo scraper deve usare esattamente queste stringhe.
ROUND_MAP_STR = {
    'Finale':                   7,
    'Semifinale':               6,
    'Quarti':                   5,
    'Ottavi di finale (16mi)':  4,
    '32mi':                     3,
    '64mi':                     2,
    '128mi':                    1,
    'Round Robin':              4,
}

BK_OVERROUND = 2 / 1.85   # ~8.1% margine bookmaker (per quote ideali)

DEFAULT_INTERACTION_PAIRS = [
    (4, 12), (0, 15), (4, 15), (12, 15), (0, 1),
    (4, 16), (6, 15), (12, 14), (14, 15), (1, 12),
]
N_INTERACTIONS = len(DEFAULT_INTERACTION_PAIRS)


# ── ANN (Wide & Deep con Residual Connections) ────────────────────────────────

class TennisANNv3(nn.Module):
    """Wide & Deep con Feature Interaction e Residual Connections."""

    def __init__(self, input_dim: int, hidden_layers: list, dropout: float = 0.3,
                 n_interactions: int = N_INTERACTIONS, interaction_pairs: list = None):
        super().__init__()
        self.interaction_pairs = interaction_pairs or DEFAULT_INTERACTION_PAIRS
        self.n_interactions = min(n_interactions, len(self.interaction_pairs))

        self.wide = nn.Linear(input_dim, 1)

        interaction_dim = input_dim + self.n_interactions
        self.deep_layers   = nn.ModuleList()
        self.deep_norms    = nn.ModuleList()
        self.deep_drops    = nn.ModuleList()
        self.residual_projs = nn.ModuleList()

        prev = interaction_dim
        for h in hidden_layers:
            self.deep_layers.append(nn.Linear(prev, h))
            self.deep_norms.append(nn.BatchNorm1d(h))
            self.deep_drops.append(nn.Dropout(dropout))
            self.residual_projs.append(
                nn.Linear(prev, h) if prev != h else nn.Identity()
            )
            prev = h

        self.deep_out = nn.Linear(prev, 1)

    def forward(self, x):
        wide_out = self.wide(x)

        interactions = []
        for i, j in self.interaction_pairs[:self.n_interactions]:
            if i < x.shape[1] and j < x.shape[1]:
                interactions.append(x[:, i] * x[:, j])
            else:
                interactions.append(torch.zeros(x.shape[0], device=x.device))
        inter_t = torch.stack(interactions, dim=1)
        deep_in = torch.cat([x, inter_t], dim=1)

        h = deep_in
        for layer, norm, drop, res_proj in zip(
                self.deep_layers, self.deep_norms,
                self.deep_drops, self.residual_projs):
            identity = res_proj(h)
            h = layer(h)
            h = norm(h)
            h = torch.relu(h)
            h = drop(h)
            h = h + identity

        deep_out = self.deep_out(h)
        return (wide_out + deep_out).squeeze(1)


# ── Funzioni ANN ──────────────────────────────────────────────────────────────

def _build_ann(cfg: dict) -> TennisANNv3:
    """Costruisce un modello ANN da un dict {'state_dict': ..., 'config': ...}."""
    hp     = cfg['config']
    hl     = hp['hidden_layers']
    dr     = hp['dropout']
    ipairs = hp.get('interaction_pairs', DEFAULT_INTERACTION_PAIRS)
    ninter = hp.get('n_interactions', len(ipairs))

    m = TennisANNv3(len(ANN_FEATURES), hl, dr, ninter, ipairs)
    try:
        m.load_state_dict(cfg['state_dict'])
    except RuntimeError:
        # Fallback per modelli legacy con interaction_pairs diverse
        old_ipairs = [
            (4, 12), (0, 15), (4, 15), (12, 14), (0, 1),
            (4, 16), (12, 16), (6, 15), (14, 15), (1, 12),
        ]
        m = TennisANNv3(len(ANN_FEATURES), hl, dr, len(old_ipairs), old_ipairs)
        m.load_state_dict(cfg['state_dict'])
    m.eval()
    return m


def _ann_prob(model: TennisANNv3, input_t: torch.Tensor) -> float:
    """Probabilità del giocatore 1 da un modello ANN (logit → sigmoid, UNA volta)."""
    model.eval()
    with torch.no_grad():
        logit = model(input_t).item()
    return float(1.0 / (1.0 + np.exp(-logit)))


def predici(input_sc, input_t: torch.Tensor, modelo_finale: dict) -> tuple:
    """
    Produce la probabilità di vittoria del giocatore 1 usando la strategia
    salvata in modelo_finale.

    Restituisce: (prob_j1: float, nome_modello: str)
    """
    s = modelo_finale['strategy']

    if s == 'ann_best':
        ann = _build_ann(modelo_finale['ann'])
        return _ann_prob(ann, input_t), modelo_finale['model_name']

    elif s == 'ann_top5':
        anns  = [_build_ann(c) for c in modelo_finale['ann_top5']]
        probs = [_ann_prob(a, input_t) for a in anns]
        return float(np.mean(probs)), modelo_finale['model_name']

    elif s == 'lgb':
        return float(modelo_finale['lgb_model'].predict_proba(input_sc)[:, 1][0]), \
               modelo_finale['model_name']

    elif s == 'xgb':
        return float(modelo_finale['xgb_model'].predict_proba(input_sc)[:, 1][0]), \
               modelo_finale['model_name']

    elif s == 'ensemble_avg':
        ann   = _build_ann(modelo_finale['ann'])
        p_ann = _ann_prob(ann, input_t)
        p_lgb = float(modelo_finale['lgb_model'].predict_proba(input_sc)[:, 1][0])
        p_xgb = float(modelo_finale['xgb_model'].predict_proba(input_sc)[:, 1][0])
        return float(np.mean([p_ann, p_lgb, p_xgb])), modelo_finale['model_name']

    elif s == 'ensemble_avg_top5':
        anns      = [_build_ann(c) for c in modelo_finale['ann_top5']]
        probs_ann = [_ann_prob(a, input_t) for a in anns]
        p_ann = float(np.mean(probs_ann))
        p_lgb = float(modelo_finale['lgb_model'].predict_proba(input_sc)[:, 1][0])
        p_xgb = float(modelo_finale['xgb_model'].predict_proba(input_sc)[:, 1][0])
        return float(np.mean([p_ann, p_lgb, p_xgb])), modelo_finale['model_name']

    elif s == 'ensemble_stacking':
        ann   = _build_ann(modelo_finale['ann'])
        p_ann = _ann_prob(ann, input_t)
        p_lgb = float(modelo_finale['lgb_model'].predict_proba(input_sc)[:, 1][0])
        p_xgb = float(modelo_finale['xgb_model'].predict_proba(input_sc)[:, 1][0])
        meta_input = np.array([[p_ann, p_lgb, p_xgb]])
        return float(modelo_finale['meta_model'].predict_proba(meta_input)[0, 1]), \
               modelo_finale['model_name']

    else:
        raise ValueError(f"Strategia sconosciuta: {s}")


# ── Feature helper ────────────────────────────────────────────────────────────

def calc_oq(history: list, my_rank: float) -> float:
    """
    Score qualità avversari (ultimi 5 match).
    history: [(result 1/0, r_opp), ...]
    """
    if not history:
        return 0.0
    total = 0.0
    r_me  = max(1.0, float(my_rank))
    for result, r_opp in history:
        r_opp = max(1.0, float(r_opp))
        if result == 1:
            contrib = np.log1p(r_opp) / np.log1p(r_me)
        else:
            contrib = -(np.log1p(r_me) / np.log1p(r_opp))
        total += float(np.clip(contrib, -3.0, 3.0))
    return total / len(history)


def days_since_last(lmd) -> float:
    """Giorni dall'ultimo match dato un intero YYYYMMDD. Default 14 se None."""
    import datetime as _dt
    if lmd is None:
        return 14.0
    y, rest = divmod(lmd, 10000)
    m, d    = divmod(rest, 100)
    try:
        last = _dt.date(y, max(1, m), max(1, d))
        return max(0.0, min(180.0, (_dt.date.today() - last).days))
    except Exception:
        return 14.0


def weeks_load(player: str, match_load_dict: dict, today_days: int) -> float:
    """Numero di match nelle ultime 8 settimane (56 giorni)."""
    dates = match_load_dict.get(player, [])
    return float(sum(1 for d in dates if today_days - d <= 56))


def upset_tendency(hist: list) -> float:
    """Frequenza di sconfitte contro giocatori di ranking peggiore (ultimi 20)."""
    vs_lower = [(r, rk) for r, rk in hist if rk < 0]
    if not vs_lower:
        return 0.2
    return sum(1 for r, _ in vs_lower if r == 0) / len(vs_lower)


def late_round_wr(hist: list) -> float:
    """Win rate nei quarti/semifinali/finali."""
    return float(np.mean(hist)) if hist else 0.5
