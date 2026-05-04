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

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRED_DIR  = os.path.dirname(os.path.abspath(__file__))
SCRAP_DIR = os.path.join(ROOT, "scraping")

# Aggiungi prediccion/ al path per importare prediction_engine
sys.path.insert(0, PRED_DIR)
from prediction_engine import (
    ANN_FEATURES, SURFACE_MAP, LEVEL_LABEL, LEVEL_MULT_LABEL, ROUND_MAP_STR,
    BK_OVERROUND, TennisANNv3, _build_ann, _ann_prob, predici,
    calc_oq, days_since_last, weeks_load, upset_tendency, late_round_wr,
    Glicko2Store,
)

INPUT_JSON  = os.path.join(SCRAP_DIR, "proximas_partidas.json")
OUTPUT_JSON = os.path.join(PRED_DIR,  "predicciones_proximas.json")


# ── Helper feature ─────────────────────────────────────────────────────────────

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
    _glicko2_raw = load(pp('glicko2_stores.pkl'), {})
    res['glicko2_surf_store'] = {
        s: Glicko2Store.from_dict(d) for s, d in _glicko2_raw.items()
    } if _glicko2_raw else {}
    res['serve_stats']  = load(pp('serve_stats.pkl'),  {})
    res['return_stats'] = load(pp('return_stats.pkl'), {})
    res['streak_players']   = load(pp('streak_players.pkl'),    {})
    res['momentum_surface'] = load(pp('momentum_surface.pkl'),  {})
    res['recent_form']      = load(pp('recent_form.pkl'),       {})
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

    initial = parts[-1].rstrip('.').lower()
    if not initial:
        return raw

    for word in parts[:-1]:
        key = word.lower().rstrip('-') + "_" + initial
        if key in name_index:
            return name_index[key]

    return raw


def build_features(p1, p2, superficie, livello, turno, res):
    """Costruisce le 32 feature v6 per una coppia di giocatori."""
    perfiles         = res['perfiles']
    stats_dict       = res['stats_dict']
    g2_stores        = res.get('glicko2_surf_store', {})
    serve_stats      = res.get('serve_stats', {})
    return_stats     = res.get('return_stats', {})
    streak_players   = res['streak_players']
    momentum_surface = res['momentum_surface']
    recent_form      = res['recent_form']
    h2h_surface      = res['h2h_surface']
    last_match_date  = res['last_match_date']
    opp_quality      = res['opp_quality']
    match_load       = res['match_load']
    upset_hist       = res['upset_hist']
    late_round_hist  = res['late_round_hist']
    ranking_dict     = res['ranking_dict']
    df_history       = res['df_history']

    sa1 = perfiles.get(p1, {}); sa2 = perfiles.get(p2, {})
    r1  = ranking_dict.get(p1, int(sa1.get('rank', 500)))
    r2  = ranking_dict.get(p2, int(sa2.get('rank', 500)))
    pts1 = sa1.get('points', 0); pts2 = sa2.get('points', 0)

    g2_s   = g2_stores.get(superficie, g2_stores.get('Gen', Glicko2Store()))
    g2_gen = g2_stores.get('Gen', Glicko2Store())
    g2_r1,  g2_rd1,  _ = g2_s.get(p1)
    g2_r2,  g2_rd2,  _ = g2_s.get(p2)
    g2_ov1, g2_ovrd1, _ = g2_gen.get(p1)
    g2_ov2, g2_ovrd2, _ = g2_gen.get(p2)

    rf1   = recent_form.get(p1, [])
    rf2   = recent_form.get(p2, [])
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

    days1 = days_since_last(last_match_date.get(p1))
    days2 = days_since_last(last_match_date.get(p2))

    today      = datetime.date.today()
    today_days = today.year * 365 + today.month * 30 + today.day
    wload1 = weeks_load(p1, match_load, today_days)
    wload2 = weeks_load(p2, match_load, today_days)

    upt1  = upset_tendency(upset_hist.get(p1, []))
    upt2  = upset_tendency(upset_hist.get(p2, []))
    lrwr1 = late_round_wr(late_round_hist.get(p1, []))
    lrwr2 = late_round_wr(late_round_hist.get(p2, []))

    skill1 = _get_skill(stats_dict, p1, superficie)
    skill2 = _get_skill(stats_dict, p2, superficie)

    def _get_serve(player, key, default):
        vals = serve_stats.get(player, {}).get(key, [])
        return float(np.mean(vals)) if vals else default

    def _get_return(player, key, default):
        vals = return_stats.get(player, {}).get(key, [])
        return float(np.mean(vals)) if vals else default

    best_of = 5 if livello == "Grand Slam" else 3
    lev_w   = LEVEL_MULT_LABEL.get(livello, 1.0)

    row = {
        'log_rank_ratio':        np.log1p(r2)   - np.log1p(r1),
        'log_pts_ratio':         np.log1p(pts1) - np.log1p(pts2),
        'diff_glicko':           g2_r1    - g2_r2,
        'diff_glicko_overall':   g2_ov1   - g2_ov2,
        'diff_glicko_rd':        g2_rd1   - g2_rd2,
        'diff_streak':           float(strk1 - strk2),
        'diff_recent_form':      form1 - form2,
        'surface_enc':           float(SURFACE_MAP.get(superficie, 0)),
        'tourney_level':         float(LEVEL_LABEL.get(livello, 3)),
        'round_enc':             float(ROUND_MAP_STR.get(turno, 3)),
        'is_best_of_5':          1.0 if best_of == 5 else 0.0,
        'indoor':                0.0,
        'diff_h2h':              float(diff_h2h),
        'diff_h2h_surface':      float(h2h_s1 - h2h_s2),
        'diff_skill':            skill1 - skill2,
        'diff_momentum':         mom1 - mom2,
        'diff_fatigue':          0.0,
        'diff_days_since_last':  days1 - days2,
        'diff_weeks_load':       wload1 - wload2,
        'diff_ace':              _get_serve(p1, 'ace',     5.0)   - _get_serve(p2, 'ace',     5.0),
        'diff_1st_pct':          _get_serve(p1, '1st_pct', 0.62) - _get_serve(p2, '1st_pct', 0.62),
        'diff_1st_won':          _get_serve(p1, '1st_won', 0.70) - _get_serve(p2, '1st_won', 0.70),
        'diff_2nd_won':          _get_serve(p1, '2nd_won', 0.50) - _get_serve(p2, '2nd_won', 0.50),
        'diff_bp_saved':         _get_serve(p1, 'bp_saved', 0.62)  - _get_serve(p2, 'bp_saved', 0.62),
        'diff_return_pct':       _get_return(p1, 'return_pct', 0.35) - _get_return(p2, 'return_pct', 0.35),
        'diff_bp_conv':          _get_return(p1, 'bp_conv',    0.35) - _get_return(p2, 'bp_conv',    0.35),
        'diff_return_1st':       _get_return(p1, 'return_1st', 0.30) - _get_return(p2, 'return_1st', 0.30),
        'diff_home':             0.0,
        'diff_opponent_quality': calc_oq(opp_quality.get(p1, []), r1)
                                 - calc_oq(opp_quality.get(p2, []), r2),
        'diff_upset_tendency':   upt1 - upt2,
        'diff_late_round_wr':    lrwr1 - lrwr2,
        'level_weight':          lev_w,
    }
    return row, h2h_w1, h2h_w2


def predici_partita(partita: dict, res: dict) -> dict:
    p1 = _resolve_player(partita['p1'], res)
    p2 = _resolve_player(partita['p2'], res)
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

        # Calibrazione base
        cal = modelo.get('calibrator')
        if cal is not None:
            try:
                prob_p1 = float(cal.predict([prob_p1])[0])
            except Exception:
                pass

        # Top-5 uncertainty ensemble (identico al predictor live)
        top5_data = modelo.get('ann_top5_uncertainty') or modelo.get('ann_top5')
        if top5_data:
            try:
                top5_models   = [_build_ann(c) for c in top5_data]
                probs_top5    = [_ann_prob(m, input_t) for m in top5_models]
                raw_top5_mean = float(np.mean(probs_top5))
                if cal is not None:
                    try:
                        prob_p1 = float(cal.predict([raw_top5_mean])[0])
                    except Exception:
                        prob_p1 = raw_top5_mean
                else:
                    prob_p1 = raw_top5_mean
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
        import traceback
        return {**partita, "skipped": True, "motivo": f"Errore predizione: {e}\n{traceback.format_exc()}"}


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
        "partite":    risultati,
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"   ✅  {ok} predizioni OK, {skip} skippate → {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
