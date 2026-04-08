# prediccion/predecir_proximas.py
"""
Batch prediction per le prossime partite ATP.
Legge:  scraping/proximas_partidas.json
Scrive: prediccion/predicciones_proximas.json
"""

import os, sys, json, datetime
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED_DIR  = os.path.dirname(os.path.abspath(__file__))
SCRAP_DIR = os.path.join(ROOT, "scraping")

INPUT_JSON  = os.path.join(SCRAP_DIR, "proximas_partidas.json")
OUTPUT_JSON = os.path.join(PRED_DIR,  "predicciones_proximas.json")

# ── Costanti (identiche a Predictor_en_Vivo.py) ───────────────────────────────
DEFAULT_INTERACTION_PAIRS = [
    (4, 12), (0, 15), (4, 15), (12, 15), (0, 1),
    (4, 16), (6, 15), (12, 14), (14, 15), (1, 12),
]
N_INTERACTIONS = len(DEFAULT_INTERACTION_PAIRS)

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
]

SURFACE_ENC   = {'Hard': 0, 'Clay': 1, 'Grass': 2}
LEVEL_LABEL   = {'Grand Slam': 5, 'Masters 1000': 4, 'ATP 500': 3, 'ATP 250': 3, 'Challenger': 2}
LEVEL_MULT    = {'Grand Slam': 2.0, 'Masters 1000': 1.5, 'ATP 500': 1.0, 'ATP 250': 1.0, 'Challenger': 0.8}
ROUND_ENC     = {
    'Finale': 7, 'Semifinale': 6, 'Quarti': 5,
    'Ottavi di finale (16mi)': 4, '32mi': 3, '64mi': 2, '128mi': 1,
    'Round Robin': 4,
}
BK_OVERROUND  = 2 / 1.85   # ~8.1% margine bookmaker


# ── ANN (identica a Predictor_en_Vivo.py) ────────────────────────────────────
class TennisANNv3(nn.Module):
    def __init__(self, input_dim, hidden_layers, dropout=0.3,
                 n_interactions=N_INTERACTIONS, interaction_pairs=None):
        super().__init__()
        self.interaction_pairs = interaction_pairs or DEFAULT_INTERACTION_PAIRS
        self.n_interactions = min(n_interactions, len(self.interaction_pairs))
        self.wide = nn.Linear(input_dim, 1)
        interaction_dim = input_dim + self.n_interactions
        self.deep_layers  = nn.ModuleList()
        self.deep_norms   = nn.ModuleList()
        self.deep_drops   = nn.ModuleList()
        self.residual_projs = nn.ModuleList()
        prev = interaction_dim
        for h in hidden_layers:
            self.deep_layers.append(nn.Linear(prev, h))
            self.deep_norms.append(nn.BatchNorm1d(h))
            self.deep_drops.append(nn.Dropout(dropout))
            self.residual_projs.append(nn.Linear(prev, h) if prev != h else nn.Identity())
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
        deep_in = torch.cat([x, torch.stack(interactions, dim=1)], dim=1)
        h = deep_in
        for layer, norm, drop, res in zip(
                self.deep_layers, self.deep_norms, self.deep_drops, self.residual_projs):
            identity = res(h)
            h = drop(torch.relu(norm(layer(h)))) + identity
        return (wide_out + self.deep_out(h)).squeeze(1)


def _build_ann(cfg):
    hp = cfg['config']
    ipairs = hp.get('interaction_pairs', DEFAULT_INTERACTION_PAIRS)
    ninter = hp.get('n_interactions', len(ipairs))
    m = TennisANNv3(len(ANN_FEATURES), hp['hidden_layers'], hp['dropout'], ninter, ipairs)
    try:
        m.load_state_dict(cfg['state_dict'])
    except RuntimeError:
        old = [(4,12),(0,15),(4,15),(12,14),(0,1),(4,16),(12,16),(6,15),(14,15),(1,12)]
        m = TennisANNv3(len(ANN_FEATURES), hp['hidden_layers'], hp['dropout'], len(old), old)
        m.load_state_dict(cfg['state_dict'])
    m.eval()
    return m


def _ann_prob(model, t):
    model.eval()
    with torch.no_grad():
        return float(torch.sigmoid(model(t)).item())


def predici(input_sc, input_t, modelo):
    s = modelo['strategy']
    if s == 'ann_best':
        return _ann_prob(_build_ann(modelo['ann']), input_t), modelo['model_name']
    elif s == 'ann_top5':
        anns = [_build_ann(c) for c in modelo['ann_top5']]
        return float(np.mean([_ann_prob(a, input_t) for a in anns])), modelo['model_name']
    elif s == 'lgb':
        return float(modelo['lgb_model'].predict_proba(input_sc)[:, 1][0]), modelo['model_name']
    elif s == 'xgb':
        return float(modelo['xgb_model'].predict_proba(input_sc)[:, 1][0]), modelo['model_name']
    elif s in ('ensemble_avg', 'ensemble_avg_top5'):
        anns = [_build_ann(c) for c in modelo.get('ann_top5', [modelo['ann']])]
        p_ann = float(np.mean([_ann_prob(a, input_t) for a in anns]))
        p_lgb = float(modelo['lgb_model'].predict_proba(input_sc)[:, 1][0])
        p_xgb = float(modelo['xgb_model'].predict_proba(input_sc)[:, 1][0])
        return float(np.mean([p_ann, p_lgb, p_xgb])), modelo['model_name']
    elif s == 'ensemble_stacking':
        ann = _build_ann(modelo['ann'])
        p_ann = _ann_prob(ann, input_t)
        p_lgb = float(modelo['lgb_model'].predict_proba(input_sc)[:, 1][0])
        p_xgb = float(modelo['xgb_model'].predict_proba(input_sc)[:, 1][0])
        return float(modelo['meta_model'].predict_proba([[p_ann, p_lgb, p_xgb]])[0, 1]), modelo['model_name']
    raise ValueError(f"Strategia sconosciuta: {s}")


# ── Helper feature ─────────────────────────────────────────────────────────────
def _calc_oq(history, my_rank):
    if not history:
        return 0.0
    r_me = max(1.0, float(my_rank))
    total = 0.0
    for result, r_opp in history:
        r_opp = max(1.0, float(r_opp))
        contrib = np.log1p(r_opp) / np.log1p(r_me) if result == 1 \
                  else -(np.log1p(r_me) / np.log1p(r_opp))
        total += float(np.clip(contrib, -3.0, 3.0))
    return total / len(history)


def _days_since(lmd):
    if lmd is None:
        return 14.0
    y, rest = divmod(lmd, 10000)
    m, d = divmod(rest, 100)
    try:
        last = datetime.date(y, max(1, m), max(1, d))
        return max(0.0, min(180.0, (datetime.date.today() - last).days))
    except Exception:
        return 14.0


def _weeks_load(player, match_load_dict, today_days):
    dates = match_load_dict.get(player, [])
    return float(sum(1 for d in dates if today_days - d <= 56))


def _upset_tendency(hist):
    vs_lower = [(r, rk) for r, rk in hist if rk < 0]
    if not vs_lower:
        return 0.2
    return sum(1 for r, _ in vs_lower if r == 0) / len(vs_lower)


def _late_round_wr(hist):
    return float(np.mean(hist)) if hist else 0.5


def _get_skill(stats_dict, player, superficie):
    return stats_dict.get((player, superficie), 0.5)


def _calc_h2h(p1, p2, df_history):
    if df_history.empty:
        return 0, 0
    w1 = len(df_history[(df_history['winner_name'] == p1) & (df_history['loser_name'] == p2)])
    w2 = len(df_history[(df_history['winner_name'] == p2) & (df_history['loser_name'] == p1)])
    return w1, w2


def pp(fname): return os.path.join(PRED_DIR,  fname)
def ps(fname): return os.path.join(SCRAP_DIR, fname)


def load_resources():
    """Carica tutti i pkl necessari per le predizioni."""
    res = {}

    def load(path, default):
        try:
            return joblib.load(path)
        except Exception as e:
            print(f"   ⚠️  {os.path.basename(path)}: {e}")
            return default

    res['modelo']           = joblib.load(pp('modelo_finale.pkl'))
    res['perfiles']         = load(ps('perfiles_jugadores.pkl'), {})
    res['stats_dict']       = load(pp('stats_superficie_v2.pkl'), {})
    res['elo_surface']      = load(pp('elo_surface.pkl'),       {})
    res['elo_overall']      = load(pp('elo_overall.pkl'),       {})
    res['streak_players']   = load(pp('streak_players.pkl'),    {})
    res['momentum_surface'] = load(pp('momentum_surface.pkl'),  {})
    res['h2h_surface']      = load(pp('h2h_surface.pkl'),       {})
    res['last_match_date']  = load(pp('last_match_date.pkl'),   {})
    res['opp_quality']      = load(pp('opp_quality.pkl'),       {})
    res['match_load']       = load(pp('match_load.pkl'),        {})
    res['upset_hist']       = load(pp('upset_hist.pkl'),        {})
    res['late_round_hist']  = load(pp('late_round_hist.pkl'),   {})

    try:
        df_rank = pd.read_csv(ps('ranking_2026.csv'))
        res['ranking_dict'] = dict(zip(df_rank['player_slug'], df_rank['rank']))
    except Exception:
        res['ranking_dict'] = {}

    try:
        res['df_history'] = pd.read_csv(ps('historialTenis.csv'), low_memory=False)
    except Exception:
        res['df_history'] = pd.DataFrame()

    # ── Indice cognome+iniziale → nome completo ──────────────────────────────
    # Gestisce nomi scraper tipo "Lehecka J." → perfiles "Jiri Lehecka"
    name_index = {}  # "lehecka_j" → "Jiri Lehecka"
    for full in res['perfiles']:
        words = full.split()
        if not words:
            continue
        first_initial = words[0][0].lower()
        # Indicizza ogni parola del cognome (tutto tranne la prima parola)
        for w in words[1:]:
            key = w.lower().rstrip('-') + "_" + first_initial
            if key not in name_index:
                name_index[key] = full
    res['name_index'] = name_index

    return res


def _resolve_player(raw: str, res: dict) -> str:
    """
    Converte "Lastname F." o "Lastname1 Lastname2 F." nel nome completo
    del database (es. "Jiri Lehecka"). Restituisce raw se non trova match.
    """
    perfiles   = res['perfiles']
    name_index = res.get('name_index', {})

    if raw in perfiles:
        return raw

    parts = raw.split()
    if len(parts) < 2:
        return raw

    # L'ultimo token è l'iniziale (es. "J." o "J")
    initial = parts[-1].rstrip('.').lower()
    if not initial:
        return raw

    # Prova ogni parola che potrebbe essere il cognome
    for word in parts[:-1]:
        key = word.lower().rstrip('-') + "_" + initial
        if key in name_index:
            return name_index[key]

    return raw


def build_features(p1, p2, superficie, livello, turno, res):
    """Costruisce le 30 feature per una coppia di giocatori."""
    perfiles        = res['perfiles']
    stats_dict      = res['stats_dict']
    elo_surface     = res['elo_surface']
    elo_overall     = res['elo_overall']
    streak_players  = res['streak_players']
    momentum_surface= res['momentum_surface']
    h2h_surface     = res['h2h_surface']
    last_match_date = res['last_match_date']
    opp_quality     = res['opp_quality']
    match_load      = res['match_load']
    upset_hist      = res['upset_hist']
    late_round_hist = res['late_round_hist']
    ranking_dict    = res['ranking_dict']
    df_history      = res['df_history']

    sa1 = perfiles.get(p1, {}); sa2 = perfiles.get(p2, {})
    r1  = ranking_dict.get(p1, int(sa1.get('rank', 500)))
    r2  = ranking_dict.get(p2, int(sa2.get('rank', 500)))
    pts1 = sa1.get('points', 0); pts2 = sa2.get('points', 0)

    ELO_DEFAULT = 1500.0
    elo1  = elo_surface.get((p1, superficie), ELO_DEFAULT)
    elo2  = elo_surface.get((p2, superficie), ELO_DEFAULT)
    elov1 = elo_overall.get(p1, ELO_DEFAULT)
    elov2 = elo_overall.get(p2, ELO_DEFAULT)

    rf1   = res.get('recent_form', {}).get(p1, [])
    rf2   = res.get('recent_form', {}).get(p2, [])
    form1 = np.mean(rf1) if rf1 else 0.5
    form2 = np.mean(rf2) if rf2 else 0.5

    strk1 = streak_players.get(p1, 0)
    strk2 = streak_players.get(p2, 0)

    mom_s1 = momentum_surface.get((p1, superficie), [])
    mom_s2 = momentum_surface.get((p2, superficie), [])
    mom1   = np.mean(mom_s1) if mom_s1 else 0.5
    mom2   = np.mean(mom_s2) if mom_s2 else 0.5

    p1k, p2k = sorted([p1, p2])
    rec_s = h2h_surface.get((p1k, p2k, superficie), [0, 0])
    h2h_s1 = rec_s[0] - rec_s[1] if p1 == p1k else rec_s[1] - rec_s[0]
    h2h_s2 = -h2h_s1

    h2h_w1, h2h_w2 = _calc_h2h(p1, p2, df_history)
    diff_h2h = h2h_w1 - h2h_w2

    days1 = _days_since(last_match_date.get(p1))
    days2 = _days_since(last_match_date.get(p2))

    today = datetime.date.today()
    today_days = today.year * 365 + today.month * 30 + today.day
    wload1 = _weeks_load(p1, match_load, today_days)
    wload2 = _weeks_load(p2, match_load, today_days)

    upt1  = _upset_tendency(upset_hist.get(p1, []))
    upt2  = _upset_tendency(upset_hist.get(p2, []))
    lrwr1 = _late_round_wr(late_round_hist.get(p1, []))
    lrwr2 = _late_round_wr(late_round_hist.get(p2, []))

    skill1 = _get_skill(stats_dict, p1, superficie)
    skill2 = _get_skill(stats_dict, p2, superficie)

    rtn_pct1 = 1.0 - sa1.get('serve_win', 65) / 100
    rtn_pct2 = 1.0 - sa2.get('serve_win', 65) / 100
    bp_conv1 = 1.0 - sa1.get('bp_saved', 60) / 100
    bp_conv2 = 1.0 - sa2.get('bp_saved', 60) / 100

    best_of  = 5 if livello == "Grand Slam" else 3
    lev_w    = LEVEL_MULT.get(livello, 1.0)

    row = {
        'log_rank_ratio':       np.log1p(r2) - np.log1p(r1),
        'log_pts_ratio':        np.log1p(pts1) - np.log1p(pts2),
        'diff_elo':             elo1 - elo2,
        'diff_elo_overall':     elov1 - elov2,
        'diff_streak':          float(strk1 - strk2),
        'diff_recent_form':     form1 - form2,
        'surface_enc':          float(SURFACE_ENC.get(superficie, 0)),
        'tourney_level':        float(LEVEL_LABEL.get(livello, 3)),
        'round_enc':            float(ROUND_ENC.get(turno, 3)),
        'is_best_of_5':         1.0 if best_of == 5 else 0.0,
        'diff_h2h':             float(diff_h2h),
        'diff_h2h_surface':     float(h2h_s1 - h2h_s2),
        'diff_skill':           skill1 - skill2,
        'diff_momentum':        mom1 - mom2,
        'diff_fatigue':         0.0,
        'diff_days_since_last': days1 - days2,
        'diff_weeks_load':      wload1 - wload2,
        'diff_ace':             sa1.get('aces', 0) - sa2.get('aces', 0),
        'diff_1st_pct':         (sa1.get('first_serve_pct', 62) - sa2.get('first_serve_pct', 62)) / 100,
        'diff_1st_won':         (sa1.get('serve_win', 65) - sa2.get('serve_win', 65)) / 100,
        'diff_2nd_won':         (sa1.get('second_serve_win', 50) - sa2.get('second_serve_win', 50)) / 100,
        'diff_bp_saved':        (sa1.get('bp_saved', 60) - sa2.get('bp_saved', 60)) / 100,
        'diff_return_pct':      rtn_pct1 - rtn_pct2,
        'diff_bp_conv':         bp_conv1 - bp_conv2,
        'diff_return_1st':      0.0,
        'diff_home':            0.0,
        'diff_opponent_quality':_calc_oq(opp_quality.get(p1, []), r1)
                                - _calc_oq(opp_quality.get(p2, []), r2),
        'diff_upset_tendency':  upt1 - upt2,
        'diff_late_round_wr':   lrwr1 - lrwr2,
        'level_weight':         lev_w,
    }
    return row, h2h_w1, h2h_w2


def predici_partita(partita: dict, res: dict) -> dict:
    p1  = _resolve_player(partita['p1'], res)
    p2  = _resolve_player(partita['p2'], res)
    perfiles = res['perfiles']

    if p1 not in perfiles or p2 not in perfiles:
        mancanti = [p for p in [p1, p2] if p not in perfiles]
        return {**partita, "skipped": True,
                "motivo": f"Giocatori non nel database: {', '.join(mancanti)}"}

    try:
        row, h2h_w1, h2h_w2 = build_features(
            p1, p2, partita['superficie'], partita['livello'], partita['turno'], res
        )
        modelo   = res['modelo']
        scaler   = modelo['scaler']
        df_row   = pd.DataFrame([row])
        input_sc = scaler.transform(df_row[ANN_FEATURES])
        input_t  = torch.tensor(input_sc.astype(np.float32))
        prob_p1, modello_usato = predici(input_sc, input_t, modelo)

        # Calibrazione
        cal = modelo.get('calibrator')
        if cal is not None:
            try:
                prob_p1 = float(cal.predict([prob_p1])[0])
            except Exception:
                pass

        prob_p1    = float(np.clip(prob_p1, 0.01, 0.99))
        favorito   = p1 if prob_p1 >= 0.5 else p2
        confidenza = prob_p1 if prob_p1 >= 0.5 else 1 - prob_p1
        quota_fav  = round(1 / (confidenza * BK_OVERROUND), 2)
        quota_sfav = round(1 / ((1 - confidenza) * BK_OVERROUND), 2)

        return {
            **partita,
            "prob_p1":    round(prob_p1, 4),
            "prob_p2":    round(1 - prob_p1, 4),
            "favorito":   favorito,
            "confidenza": round(confidenza, 4),
            "h2h_p1":     h2h_w1,
            "h2h_p2":     h2h_w2,
            "quota_fav":  quota_fav,
            "quota_sfav": quota_sfav,
            "modello":    modello_usato,
            "skipped":    False,
        }
    except Exception as e:
        return {**partita, "skipped": True, "motivo": f"Errore predizione: {e}"}


def main():
    print("🔮  Predizioni prossime partite ATP...")

    if not os.path.exists(INPUT_JSON):
        print(f"   ⚠️  {INPUT_JSON} non trovato — salvo JSON vuoto")
        out = {"updated_at": datetime.datetime.utcnow().isoformat(), "partite": []}
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        return

    with open(INPUT_JSON, encoding="utf-8") as f:
        partite = json.load(f)

    print(f"   📋  {len(partite)} partite da processare")

    if not partite:
        out = {"updated_at": datetime.datetime.utcnow().isoformat(), "partite": []}
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print("   ⚠️  Nessuna partita trovata — JSON vuoto salvato")
        return

    print("   📦  Caricamento modelli e pkl...")
    try:
        res = load_resources()
    except Exception as e:
        print(f"   ❌  Impossibile caricare modelo_finale.pkl: {e}")
        return

    # Carica anche recent_form se esiste (field opzionale)
    rf_path = os.path.join(PRED_DIR, 'recent_form.pkl')
    if os.path.exists(rf_path):
        try:
            res['recent_form'] = joblib.load(rf_path)
        except Exception:
            res['recent_form'] = {}
    else:
        res['recent_form'] = {}

    risultati = []
    ok = skip = 0
    for p in partite:
        r = predici_partita(p, res)
        risultati.append(r)
        if r.get('skipped'):
            skip += 1
            print(f"   ⏭️  Skip: {p['p1']} vs {p['p2']} — {r.get('motivo','')}")
        else:
            ok += 1

    out = {
        "updated_at": datetime.datetime.utcnow().isoformat(),
        "partite": risultati,
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"   ✅  {ok} predizioni OK, {skip} skippate → {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
