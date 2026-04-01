"""
uncertainty.py — Stima dell'incertezza per predizioni tennis

Usa il disaccordo tra i top-5 modelli ANN salvati in modelo_finale.pkl
come segnale di incertezza.

Opzionalmente supporta MC Dropout sul singolo modello migliore.

Utilizzo:
    from uncertainty import predict_with_uncertainty, mc_dropout_uncertainty

    result = predict_with_uncertainty(input_t, ann_models)
    # → {'p_mean': 0.74, 'p_std': 0.03, 'ci_low': 0.68, 'ci_high': 0.80,
    #    'high_uncertainty': False, 'n_models': 5, 'all_probs': [...]}
"""

import numpy as np
import torch


def predict_with_uncertainty(input_t: torch.Tensor, ann_models: list) -> dict:
    """
    Calcola media e varianza delle predizioni di N modelli ANN.

    Args:
        input_t: tensore di input (1, n_features), già scalato
        ann_models: lista di modelli TennisANNv3 già costruiti e in eval mode

    Returns:
        dict con:
            p_mean         — probabilità media (la predizione)
            p_std          — deviazione standard tra modelli
            ci_low/ci_high — intervallo di confidenza empirico (±1.96σ)
            high_uncertainty — True se std > 0.12
            uncertain_flag — stringa descrittiva ('HIGH'/'MEDIUM'/'LOW')
            n_models       — quanti modelli usati
            all_probs      — lista di probabilità individuali
    """
    if not ann_models:
        return {
            'p_mean': 0.5, 'p_std': 0.5, 'ci_low': 0.0, 'ci_high': 1.0,
            'high_uncertainty': True, 'uncertain_flag': 'HIGH',
            'n_models': 0, 'all_probs': []
        }

    probs = []
    for model in ann_models:
        model.eval()
        with torch.no_grad():
            logit = model(input_t).item()
        p = float(1 / (1 + np.exp(-logit)))
        probs.append(p)

    p_mean = float(np.mean(probs))
    p_std  = float(np.std(probs))
    ci_low  = float(np.clip(p_mean - 1.96 * p_std, 0.0, 1.0))
    ci_high = float(np.clip(p_mean + 1.96 * p_std, 0.0, 1.0))

    if p_std < 0.05:
        flag = 'LOW'       # bassa incertezza = buona concordanza
    elif p_std < 0.12:
        flag = 'MEDIUM'
    else:
        flag = 'HIGH'

    return {
        'p_mean':           p_mean,
        'p_std':            p_std,
        'ci_low':           ci_low,
        'ci_high':          ci_high,
        'high_uncertainty': p_std > 0.12,
        'uncertain_flag':   flag,
        'n_models':         len(probs),
        'all_probs':        probs,
    }


def mc_dropout_uncertainty(model, input_t: torch.Tensor, n_passes: int = 50) -> dict:
    """
    MC Dropout: esegue N forward pass con dropout attivo per stimare incertezza.
    Segnale secondario, da usare quando top-5 non disponibili.

    Args:
        model: singolo modello TennisANNv3
        input_t: tensore di input (1, n_features)
        n_passes: numero di forward pass (default 50)

    Returns:
        dict con p_mean, p_std, ci_low, ci_high, uncertain_flag
    """
    model.train()  # attiva dropout
    probs = []
    with torch.no_grad():
        for _ in range(n_passes):
            logit = model(input_t).item()
            probs.append(float(1 / (1 + np.exp(-logit))))
    model.eval()

    p_mean = float(np.mean(probs))
    p_std  = float(np.std(probs))

    if p_std < 0.04:
        flag = 'LOW'
    elif p_std < 0.10:
        flag = 'MEDIUM'
    else:
        flag = 'HIGH'

    return {
        'p_mean':         p_mean,
        'p_std':          p_std,
        'ci_low':         float(np.clip(p_mean - 1.96 * p_std, 0, 1)),
        'ci_high':        float(np.clip(p_mean + 1.96 * p_std, 0, 1)),
        'uncertain_flag': flag,
        'n_passes':       n_passes,
    }
