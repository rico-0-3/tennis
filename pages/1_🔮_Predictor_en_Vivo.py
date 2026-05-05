import streamlit as st
import pandas as pd
import joblib
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import plotly.graph_objects as go

# ── Import shared prediction engine ──────────────────────────────────────────
_pred_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'prediccion')
sys.path.insert(0, _pred_dir)
from prediction_engine import (
    ANN_FEATURES, SURFACE_MAP, LEVEL_LABEL, LEVEL_MULT_LABEL, ROUND_MAP_STR,
    BK_OVERROUND, TennisANNv3, _build_ann, _ann_prob, predici, _apply_cal,
    calc_oq, days_since_last, weeks_load as weeks_load_fn,
    upset_tendency as upset_tendency_fn, late_round_wr as late_round_wr_fn,
)

# Import court speed helper
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scraping'))
try:
    from court_speed_helper import get_court_stats_latest
except:
    def get_court_stats_latest(name, surf): return (11.5, 1.10) if surf != 'Clay' else (6.5, 0.65)

# Import moduli di supporto opzionali
try:
    from uncertainty import predict_with_uncertainty
    HAS_UNCERTAINTY = True
except ImportError:
    HAS_UNCERTAINTY = False
try:
    from meta_confidence import MetaConfidenceScorer, color_for_label
    HAS_META = True
except ImportError:
    HAS_META = False

st.set_page_config(page_title="ATP Predittore 2026", page_icon="🎾", layout="wide")

# # Carica CSS custom
# css_path = os.path.join(os.path.dirname(__file__), "../assets/style.css")
# if os.path.exists(css_path):
#     with open(css_path, "r", encoding="utf-8") as f:
#         st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


st.title("🎾 ATP Prediction Pro 2026")

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;} /* Oculta los 3 puntitos de arriba a la derecha */
            footer {visibility: hidden;}
            header {visibility: hidden;} /* Oculta el "Made with Streamlit" de abajo */
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

st.markdown("""
Questa applicazione utilizza un modello di **Intelligenza Artificiale** selezionato automaticamente
in fase di addestramento come il **migliore** tra diverse strategie:

| Strategia | Tipo | Come funziona |
|-----------|------|---------------|
| 🧠 **ANN Best** | Rete Neurale (Wide & Deep) | Singola rete neurale con la migliore architettura |
| 🧠×5 **ANN Top-5 Avg** | Media di 5 Reti Neurali | Media delle probabilità di 5 architetture diverse — riduce l'errore individuale |
| 🌲 **LightGBM** | Gradient Boosting ad Albero | Alberi decisionali in sequenza, ognuno corregge gli errori del precedente |
| 🌳 **XGBoost** | Gradient Boosting ad Albero | Simile a LightGBM con regolarizzazione diversa |
| 📊 **Ensemble Avg** | Media ANN+LGB+XGB | Media semplice delle probabilità dei 3 modelli |
| 🎯 **Ensemble Stacking** | Meta-Learner | Regressione logistica che combina i 3 modelli con pesi ottimali |

Il sistema analizza **30 feature** tra cui:
* 📊 **Gerarchia:** Ranking (log), Punti (log), Elo (globale e per superficie).
* ⚔️ **Storico:** Scontri diretti (H2H globale e per superficie), forma recente.
* 🧠 **Momento:** Striscia, momentum, fatica, giorni dall'ultimo match.
* 🎯 **Tecnica:** Ace, servizio (1st pct, 1st won, 2nd won), break point, resa al ritorno.
* 🏆 **Contesto:** Superficie, livello torneo, turno, best-of-3/5.
* 🎯 **Qualità avversari:** Score pesato sugli ultimi 5 match (win vs top = valore alto, loss vs debole = penalità).
""")

st.write("---")


# â"€â"€â"€ Definizione ANN v3 (stessa architettura di train_ann.py) â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
DEFAULT_INTERACTION_PAIRS = [
    (4, 12), (0, 15), (4, 15), (12, 15), (0, 1),
    (4, 16), (6, 15), (12, 14), (14, 15), (1, 12),
]
N_INTERACTIONS = len(DEFAULT_INTERACTION_PAIRS)


class TennisANNv3(nn.Module):
    """Wide & Deep con Feature Interaction e Residual Connections."""
    def __init__(self, input_dim: int, hidden_layers: list, dropout: float = 0.3,
                 n_interactions: int = N_INTERACTIONS,
                 interaction_pairs: list = None):
        super().__init__()
        self.interaction_pairs = interaction_pairs or DEFAULT_INTERACTION_PAIRS
        self.n_interactions = min(n_interactions, len(self.interaction_pairs))

        # === Wide path ===
        self.wide = nn.Linear(input_dim, 1)

        # === Deep path con residual ===
        interaction_dim = input_dim + self.n_interactions
        self.deep_layers = nn.ModuleList()
        self.deep_norms = nn.ModuleList()
        self.deep_drops = nn.ModuleList()
        self.residual_projs = nn.ModuleList()

        prev = interaction_dim
        for h in hidden_layers:
            self.deep_layers.append(nn.Linear(prev, h))
            self.deep_norms.append(nn.BatchNorm1d(h))
            self.deep_drops.append(nn.Dropout(dropout))
            if prev != h:
                self.residual_projs.append(nn.Linear(prev, h))
            else:
                self.residual_projs.append(nn.Identity())
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


SURFACE_MAP   = {'Hard': 0, 'Clay': 1, 'Grass': 2}
LEVEL_MAP     = {'G': 5, 'M': 4, 'A': 3, 'F': 4, 'C': 2, 'S': 1, 'E': 0}
ROUND_MAP_STR = {'Finale': 7, 'Semifinale': 6, 'Quarti': 5,
                 'Ottavi di finale (16mi)': 4, '32mi': 3,
                 '64mi': 2, '128mi': 1, 'Round Robin': 4}
LEVEL_MULT_LABEL = {'Grand Slam': 2.0, 'Masters 1000': 1.5, 'ATP 500': 1.0,
                    'ATP 250': 1.0, 'Challenger': 0.8}



# â"€â"€â"€ Caricamento risorse â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

def get_version():
    try:
        ruta_script = os.path.dirname(os.path.abspath(__file__))
        ruta_proyecto = os.path.dirname(ruta_script)
        ruta_redeploy = os.path.join(ruta_proyecto, ".streamlit_redeploy.py")
        with open(ruta_redeploy) as f:
            return f.read().strip()
    except:
        return "default"

@st.cache_resource
def cargar_todo(_version: str):
    ruta_script = os.path.dirname(os.path.abspath(__file__))
    ruta_proyecto = os.path.dirname(ruta_script)
    ruta_pred   = os.path.join(ruta_proyecto, "prediccion")
    ruta_scrap  = os.path.join(ruta_proyecto, "scraping")

    def pp(f): return os.path.join(ruta_pred,  f)
    def ps(f): return os.path.join(ruta_scrap, f)

    try:
        stats_dict     = joblib.load(pp('stats_superficie_v2.pkl'))
        perfiles       = joblib.load(ps('perfiles_jugadores.pkl'))
    except FileNotFoundError as e:
        st.error(f"Mancano file fondamentali. Errore: {e}")
        st.stop()

    # â"€â"€ Modello finale (selezionato automaticamente in training) â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
    modelo_finale = None
    finale_path = pp('modelo_finale.pkl')
    if os.path.exists(finale_path):
        try:
            # Temporarily patch torch.load to always use CPU mapping
            # This is needed because joblib.load calls torch.load internally for PyTorch objects
            original_torch_load = torch.load
            def patched_torch_load(*args, **kwargs):
                kwargs['map_location'] = torch.device('cpu')
                return original_torch_load(*args, **kwargs)
            
            torch.load = patched_torch_load
            try:
                modelo_finale = joblib.load(finale_path)
            finally:
                # Restore original torch.load
                torch.load = original_torch_load
                
        except Exception as e:
            st.warning(f"modelo_finale.pkl non caricato: {e}")

    # Elo + streak + momentum + elo_overall + recent_form
    elo_surface = {}     # {(player, surface): elo_float}
    streak_players = {}
    momentum_surface = {}  # {(player, surface): [last 10 results]}
    elo_path    = pp('elo_surface.pkl')
    streak_path = pp('streak_players.pkl')
    mom_path    = pp('momentum_surface.pkl')
    elo_ov_path = pp('elo_overall.pkl')
    rf_path     = pp('recent_form.pkl')
    if os.path.exists(elo_path):    elo_surface      = joblib.load(elo_path)
    if os.path.exists(streak_path): streak_players   = joblib.load(streak_path)
    if os.path.exists(mom_path):    momentum_surface = joblib.load(mom_path)

    elo_overall = {}     # {player: elo_float}
    recent_form = {}     # {player: [last 10 results]}
    if os.path.exists(elo_ov_path): elo_overall  = joblib.load(elo_ov_path)
    if os.path.exists(rf_path):     recent_form  = joblib.load(rf_path)

    h2h_surface_dict = {}  # {(p1, p2, surface): [w1, w2]}
    last_match_date_dict = {}  # {player: int (YYYYMMDD)}
    opp_quality_dict = {}  # {player: [(result, r_opp), ...]} last 5 matches
    match_load_dict  = {}  # {player: [days, ...]} — last 80 match days
    upset_hist_dict  = {}  # {player: [(result, rank_diff), ...]} — last 20
    late_round_hist_dict = {}  # {player: [1/0, ...]} — QF/SF/F results
    h2h_s_path  = pp('h2h_surface.pkl')
    lmd_path    = pp('last_match_date.pkl')
    oq_path     = pp('opp_quality.pkl')
    ml_path     = pp('match_load.pkl')
    uh_path     = pp('upset_hist.pkl')
    lrh_path    = pp('late_round_hist.pkl')
    if os.path.exists(h2h_s_path):  h2h_surface_dict     = joblib.load(h2h_s_path)
    if os.path.exists(lmd_path):    last_match_date_dict = joblib.load(lmd_path)
    if os.path.exists(oq_path):     opp_quality_dict     = joblib.load(oq_path)
    if os.path.exists(ml_path):     match_load_dict      = joblib.load(ml_path)
    if os.path.exists(uh_path):     upset_hist_dict      = joblib.load(uh_path)
    if os.path.exists(lrh_path):    late_round_hist_dict = joblib.load(lrh_path)

    # Storico e Ranking
    try:
        df_history = pd.read_csv(ps("historialTenis.csv"), low_memory=False)
    except:
        df_history = pd.DataFrame()

    try:
        df_rank = pd.read_csv(ps("ranking_2026.csv"))
        ranking_dict = dict(zip(df_rank['player_slug'], df_rank['rank']))
    except:
        ranking_dict = {}

    return (stats_dict, perfiles, df_history, ranking_dict,
            modelo_finale,
            elo_surface, streak_players,
            momentum_surface, elo_overall, recent_form,
            h2h_surface_dict, last_match_date_dict, opp_quality_dict,
            match_load_dict, upset_hist_dict, late_round_hist_dict)


(stats_dict, perfiles, df_history, ranking_dict,
 modelo_finale,
 elo_surface, streak_players,
 momentum_surface, elo_overall, recent_form,
 h2h_surface_dict, last_match_date_dict,
 opp_quality_dict,
 match_load_dict, upset_hist_dict,
 late_round_hist_dict) = cargar_todo(_version=get_version())

if modelo_finale is None:
    st.error("❌ Modello non trovato. Assicurati che `modelo_finale.pkl` sia nella cartella `prediccion/`.")
    st.stop()

# Estrai componenti dal modello finale
finale_strategy = modelo_finale['strategy']
finale_scaler   = modelo_finale['scaler']

# Carica MetaConfidenceScorer con dati storici (se disponibili)
_analysis_dir = os.path.join(_pred_dir, 'error_analysis_output')
if HAS_META:
    meta_scorer = MetaConfidenceScorer.from_files(_analysis_dir)
else:
    meta_scorer = None
finale_name     = modelo_finale['model_name']
finale_accuracy = modelo_finale.get('accuracy', 0)
finale_score    = modelo_finale.get('score', 0)
finale_trained  = modelo_finale.get('trained_date', 'N/D')
surf_acc        = modelo_finale.get('surface_accuracy', {})

_surf_lines = " · ".join(
    f"{s[0]}: {v:.1%}" for s, v in [("Hard", surf_acc.get("Hard")),
                                      ("Clay", surf_acc.get("Clay")),
                                      ("Grass", surf_acc.get("Grass"))]
    if v is not None
)
st.sidebar.success(
    f"🎯 Modello attivo: **{finale_name}**\n\n"
    f"Acc: {finale_accuracy:.1%} · Score: {finale_score:.4f}\n\n"
    + (f"H: {surf_acc.get('Hard', 0):.1%} · C: {surf_acc.get('Clay', 0):.1%} · G: {surf_acc.get('Grass', 0):.1%}\n\n" if surf_acc else "")
    + f"📅 Addestrato: {finale_trained}"
)


def get_skill(p, s): return stats_dict.get((p, s), 0.5)

# Wrapper locale: predici() dal shared engine richiede modelo_finale come argomento esplicito
def _predici_locale(input_sc, input_t):
    """Chiama predici() dal shared engine passando il modelo_finale caricato."""
    return predici(input_sc, input_t, modelo_finale)

def mostrar_historial_detallado(lista_partidos):
    if not lista_partidos:
        st.caption("Nessun dato disponibile.")
        return
    for partido in reversed(lista_partidos):
        icono = "✅" if partido['resultado'] == 'W' else "🔴"
        st.markdown(f"{icono} **{partido.get('ronda','??')}**: vs {partido['rival']}")
        st.caption(f"Score: {partido['score']}")
        st.divider()

def calcular_h2h(p1, p2):
    if df_history.empty: return 0, 0
    w1 = len(df_history[(df_history['winner_name'] == p1) & (df_history['loser_name'] == p2)])
    w2 = len(df_history[(df_history['winner_name'] == p2) & (df_history['loser_name'] == p1)])
    return w1, w2

def grafico_radar(j1, j2, perfiles, stats_sup):
    d1 = perfiles.get(j1, {}); d2 = perfiles.get(j2, {})
    aces1 = d1.get('aces', 0); aces2 = d2.get('aces', 0)
    df1   = d1.get('df', 0);   df2   = d2.get('df', 0)
    srv1  = d1.get('serve_win', 65); srv2 = d2.get('serve_win', 65)
    bp1   = d1.get('bp_saved', 60); bp2  = d2.get('bp_saved', 60)
    hold1 = d1.get('service_hold', 75); hold2 = d2.get('service_hold', 75)
    hard1 = stats_sup.get((j1, 'Hard'), 0.5); hard2 = stats_sup.get((j2, 'Hard'), 0.5)
    clay1 = stats_sup.get((j1, 'Clay'), 0.5); clay2 = stats_sup.get((j2, 'Clay'), 0.5)
    grass1 = stats_sup.get((j1, 'Grass'), 0.5); grass2 = stats_sup.get((j2, 'Grass'), 0.5)

    # aces e df sono medie per partita (es. 5.6 ace/match, 2.1 df/match)
    # scala: ace max ~20 (Isner), df max ~8
    sc_a1 = min(1, aces1/20); sc_a2 = min(1, aces2/20)
    sc_d1 = max(0, 1-(df1/8)); sc_d2 = max(0, 1-(df2/8))
    sc_s1 = max(0, min(1,(srv1-60)/25)); sc_s2 = max(0, min(1,(srv2-60)/25))
    sc_b1 = max(0, min(1,(bp1-50)/25)); sc_b2 = max(0, min(1,(bp2-50)/25))
    sc_h1 = max(0, min(1,(hold1-65)/25)); sc_h2 = max(0, min(1,(hold2-65)/25))

    cats = ['Ace', 'Controllo (Pochi DF)', 'Potenza (1° Servizio)',
            'Mentalità (BP Salvati)', 'Solidità (Battuta Vinta)',
            'Hard', 'Terra Rossa', 'Erba', 'Ace']
    v1 = [sc_a1, sc_d1, sc_s1, sc_b1, sc_h1, hard1, clay1, grass1, sc_a1]
    v2 = [sc_a2, sc_d2, sc_s2, sc_b2, sc_h2, hard2, clay2, grass2, sc_a2]
    hv1 = [f"{aces1:.1f} Ace/match", f"{df1:.1f} DF/match", f"{srv1}%", f"{bp1}%", f"{hold1}%",
           f"{hard1:.0%}", f"{clay1:.0%}", f"{grass1:.0%}", f"{aces1:.1f} Ace/match"]
    hv2 = [f"{aces2:.1f} Ace/match", f"{df2:.1f} DF/match", f"{srv2}%", f"{bp2}%", f"{hold2}%",
           f"{hard2:.0%}", f"{clay2:.0%}", f"{grass2:.0%}", f"{aces2:.1f} Ace/match"]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=v1, theta=cats, fill='toself', name=j1,
                                  hovertext=hv1, hoverinfo="text+name", line_color='#00CC96'))
    fig.add_trace(go.Scatterpolar(r=v2, theta=cats, fill='toself', name=j2,
                                  hovertext=hv2, hoverinfo="text+name", line_color='#AB63FA'))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=False, range=[0, 1]), bgcolor='rgba(0,0,0,0)'),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        showlegend=True, height=450,
        font=dict(color='white'),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5)
    )
    return fig


lista_jugadores = sorted(list(perfiles.keys()))

def actualizar_j1():
    nombre = st.session_state.sel_j1
    datos  = perfiles[nombre]
    st.session_state.r1  = ranking_dict.get(nombre, int(datos['rank']))
    st.session_state.a1  = float(datos['age'])
    st.session_state.h1  = int(datos['ht'])
    st.session_state.m1  = int(datos['momentum'] * 100)
    st.session_state.nac1 = datos['ioc']
    st.session_state.l5_1 = datos.get('last_5', [])

def actualizar_j2():
    nombre = st.session_state.sel_j2
    datos  = perfiles[nombre]
    st.session_state.r2  = ranking_dict.get(nombre, int(datos['rank']))
    st.session_state.a2  = float(datos['age'])
    st.session_state.h2  = int(datos['ht'])
    st.session_state.m2  = int(datos['momentum'] * 100)
    st.session_state.nac2 = datos['ioc']
    st.session_state.l5_2 = datos.get('last_5', [])


# â"€â"€â"€ SIDEBAR â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
with st.sidebar:
    st.header("⚙️ Configurazione")

    st.subheader("🧠 Cervello dell'IA")
    st.info("Usando: **ANN v4.1** — Wide&Deep + Residual + GBM Ensemble (30 feature, calibrato) 🔥")

    st.divider()

    superficie    = st.selectbox("Superficie", ["Hard", "Clay", "Grass"])
    
    # Selettore torneo per court speed
    tornei_comuni = [
        "Australian Open", "Roland Garros", "Wimbledon", "Us Open",
        "Indian Wells Masters", "Miami Masters", "Monte Carlo Masters",
        "Madrid Masters", "Rome Masters", "Canada Masters", "Cincinnati Masters",
        "Shanghai Masters", "Paris Masters", "Tour Finals",
        "Dubai", "Barcelona", "Halle", "Queens Club", "Hamburg",
        "Beijing", "Tokyo", "Basel", "Vienna", "Rotterdam",
        "Acapulco", "Rio de Janeiro", "Buenos Aires", "Santiago",
        "Brisbane", "Adelaide", "Auckland", "Doha", "Montpellier",
        "Dallas", "Delray Beach", "Washington", "Winston-Salem",
        "Altro (inserisci manualmente)"
    ]
    torneo_sel = st.selectbox("🏢 Torneo", tornei_comuni)
    if torneo_sel == "Altro (inserisci manualmente)":
        torneo_sel = st.text_input("Nome torneo", "ATP 250")
    
    # Mostra court stats per il torneo selezionato
    court_ace, court_spd = get_court_stats_latest(torneo_sel, superficie)
    st.caption(f"🎾 Ace%: **{court_ace:.1f}%** | Speed: **{court_spd:.2f}**")
    
    pais_torneo   = st.selectbox("Paese Sede", ["NEUTRAL", "ARG", "ESP", "FRA", "USA", "GBR", "AUS"])
    turno         = st.selectbox("Turno", list(ROUND_MAP_STR.keys()), index=0)
    livello       = st.selectbox("Livello Torneo", ["Grand Slam", "Masters 1000", "ATP 500", "ATP 250", "Challenger"],
                                  help="G=Grand Slam | M=Masters | A=500/250 | C=Challenger")
    draw_sz       = st.selectbox("Tabellone", [128, 96, 64, 32, 16], index=2)
    best_of       = st.selectbox("Formato", [3, 5], index=0,
                                  help="3 = best-of-3 (ATP normali) | 5 = best-of-5 (Grand Slam)")

    LEVEL_LABEL = {'Grand Slam': 5, 'Masters 1000': 4, 'ATP 500': 3, 'ATP 250': 3, 'Challenger': 2}


# â"€â"€â"€ GIOCATORI â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 👤 Giocatore 1")
    def_nom1 = "Carlos Alcaraz" if "Carlos Alcaraz" in lista_jugadores else lista_jugadores[0]
    if 'nac1' not in st.session_state:
        d = perfiles[def_nom1]
        st.session_state.r1   = ranking_dict.get(def_nom1, int(d['rank']))
        st.session_state.a1   = float(d['age'])
        st.session_state.h1   = int(d['ht'])
        st.session_state.m1   = int(d['momentum'] * 100)
        st.session_state.nac1 = d['ioc']
        st.session_state.l5_1 = d.get('last_5', [])

    nombre1 = st.selectbox("Seleziona:", lista_jugadores,
                           index=lista_jugadores.index(def_nom1),
                           key="sel_j1", on_change=actualizar_j1)
    nac1    = st.text_input("Paese", disabled=True, key="nac1")

    puntos1 = perfiles.get(nombre1, {}).get('points', 0)
    c_rank1, c_pts1 = st.columns(2)
    with c_rank1: r1 = st.number_input("Ranking (2026)", 1, 5000, key="r1")
    with c_pts1:  st.metric(label="🏆 Punti ATP", value=f"{int(puntos1):,}".replace(",", "."))

    a1   = st.number_input("Età", 15.0, 50.0, step=0.5, key="a1")
    h1   = st.number_input("Altezza (cm)", 150, 230, key="h1")
    seed1 = st.number_input("Testa di serie", 0, 32, 0, help="0 = non è testa di serie")

    st.markdown("##### âš¡ Stato")
    mom1 = st.slider("Momento (%)", 0, 100, key="m1") / 100

    with st.expander("Ultimi 5 partiti"):
        mostrar_historial_detallado(st.session_state.get('l5_1', []))

    fat1 = st.number_input("Fatica (min)", 0, 1000, 0, key="f1")

with col2:
    st.markdown("### 👤 Giocatore 2")
    def_nom2 = "Novak Djokovic" if "Novak Djokovic" in lista_jugadores else lista_jugadores[1]
    if 'nac2' not in st.session_state:
        d = perfiles[def_nom2]
        st.session_state.r2   = ranking_dict.get(def_nom2, int(d['rank']))
        st.session_state.a2   = float(d['age'])
        st.session_state.h2   = int(d['ht'])
        st.session_state.m2   = int(d['momentum'] * 100)
        st.session_state.nac2 = d['ioc']
        st.session_state.l5_2 = d.get('last_5', [])

    nombre2 = st.selectbox("Seleziona:", lista_jugadores,
                           index=lista_jugadores.index(def_nom2),
                           key="sel_j2", on_change=actualizar_j2)
    nac2    = st.text_input("Paese", disabled=True, key="nac2")

    puntos2 = perfiles.get(nombre2, {}).get('points', 0)
    c_rank2, c_pts2 = st.columns(2)
    with c_rank2: r2 = st.number_input("Ranking (2026)", 1, 5000, key="r2")
    with c_pts2:  st.metric(label="🏆 Punti ATP", value=f"{int(puntos2):,}".replace(",", "."))

    a2   = st.number_input("Età", 15.0, 50.0, step=0.5, key="a2")
    h2   = st.number_input("Altezza (cm)", 150, 230, key="h2")
    seed2 = st.number_input("Testa di serie", 0, 32, 0, help="0 = non è testa di serie", key="seed2_input")

    st.markdown("##### âš¡ Stato")
    mom2 = st.slider("Momento (%)", 0, 100, key="m2") / 100

    with st.expander("Ultimi 5 partiti"):
        mostrar_historial_detallado(st.session_state.get('l5_2', []))

    fat2 = st.number_input("Fatica (min)", 0, 1000, 0, key="f2")


# â"€â"€â"€ H2H + RADAR â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
st.divider()
c_h2h, c_radar = st.columns([1, 2])

with c_h2h:
    st.subheader("⚔️ Scontri Diretti")
    wins_p1, wins_p2 = calcular_h2h(nombre1, nombre2)
    st.metric(f"Vittorie {nombre1}", wins_p1)
    st.metric(f"Vittorie {nombre2}", wins_p2)
    st.caption("Partite precedenti nel database.")

with c_radar:
    st.subheader("🕸️ Analisi Tecnica 360°")
    try:
        fig_radar = grafico_radar(nombre1, nombre2, perfiles, stats_dict)
        st.plotly_chart(fig_radar, use_container_width=True)
    except Exception as e:
        st.warning(f"Impossibile generare il radar: {e}")

st.divider()


# â"€â"€â"€ Helper: Opponent Quality Score â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
def _calc_oq(history, my_rank):
    """Media pesata qualità avversari: win vs top vale di più, loss vs scarso penalizza.
    history: [(result 1/0, r_opp), ...] — ultimi 5 match
    my_rank: ranking attuale del giocatore
    """
    if not history:
        return 0.0
    total = 0.0
    r_me = max(1.0, float(my_rank))
    for result, r_opp in history:
        r_opp = max(1.0, float(r_opp))
        if result == 1:  # vittoria
            contrib = np.log1p(r_opp) / np.log1p(r_me)
        else:            # sconfitta
            contrib = -(np.log1p(r_me) / np.log1p(r_opp))
        total += float(np.clip(contrib, -3.0, 3.0))
    return total / len(history)


# â"€â"€â"€ PREDIZIONE â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
if st.button("🔮 PREDICI con ANN v5.1", type="primary", use_container_width=True):

    skill1  = get_skill(nombre1, superficie)
    skill2  = get_skill(nombre2, superficie)
    home1   = 1 if nac1 == pais_torneo else 0
    home2   = 1 if nac2 == pais_torneo else 0
    diff_h2h = wins_p1 - wins_p2

    hand_map_d = {'R': 1, 'L': -1, 'U': 0}
    hand1 = perfiles.get(nombre1, {}).get('hand', 'R')
    hand2 = perfiles.get(nombre2, {}).get('hand', 'R')
    sa1   = perfiles.get(nombre1, {})
    sa2   = perfiles.get(nombre2, {})
    pts1  = sa1.get('points', 0); pts2 = sa2.get('points', 0)
    seed1_eff = seed1 if seed1 > 0 else 33
    seed2_eff = seed2 if seed2 > 0 else 33

    # Elo per superficie
    ELO_DEFAULT = 1500.0
    elo1 = elo_surface.get((nombre1, superficie), ELO_DEFAULT)
    elo2 = elo_surface.get((nombre2, superficie), ELO_DEFAULT)

    # Elo overall (all surfaces)
    elo_ov1 = elo_overall.get(nombre1, ELO_DEFAULT)
    elo_ov2 = elo_overall.get(nombre2, ELO_DEFAULT)

    # Recent form (all surfaces, last 10 matches)
    rf1 = recent_form.get(nombre1, [])
    rf2 = recent_form.get(nombre2, [])
    form1 = np.mean(rf1) if rf1 else 0.5
    form2 = np.mean(rf2) if rf2 else 0.5

    # Striscia attiva (dal pkl, oppure calcolata da last_5)
    def calc_streak_from_last5(last5):
        s = 0
        for m in reversed(last5):
            if m['risultato'] == 'W' if 'risultato' in m else m.get('resultado') == 'W':
                if s >= 0: s += 1
                else: break
            else:
                if s <= 0: s -= 1
                else: break
        return s

    strk1 = streak_players.get(nombre1,
                calc_streak_from_last5(st.session_state.get('l5_1', [])))
    strk2 = streak_players.get(nombre2,
                calc_streak_from_last5(st.session_state.get('l5_2', [])))

    # Momentum per superficie (dal pkl, fallback a slider)
    mom_surf_1 = momentum_surface.get((nombre1, superficie), [])
    mom_surf_2 = momentum_surface.get((nombre2, superficie), [])
    mom1_val = np.mean(mom_surf_1) if mom_surf_1 else mom1
    mom2_val = np.mean(mom_surf_2) if mom_surf_2 else mom2

    # Return stats (dai profili, approssimazione)
    rtn_pct1 = 1.0 - (sa1.get('serve_win', 65) / 100)  # approssimazione
    rtn_pct2 = 1.0 - (sa2.get('serve_win', 65) / 100)
    bp_conv1 = 1.0 - (sa1.get('bp_saved', 60) / 100)   # approssimazione
    bp_conv2 = 1.0 - (sa2.get('bp_saved', 60) / 100)
    rtn_1st1 = 0.30  # default (non disponibile nel profilo)
    rtn_1st2 = 0.30

    # H2H per superficie (dal pkl)
    p1k, p2k = sorted([nombre1, nombre2])
    rec_s = h2h_surface_dict.get((p1k, p2k, superficie), [0, 0])
    if nombre1 == p1k:
        _h2h_s1, _h2h_s2 = rec_s[0] - rec_s[1], rec_s[1] - rec_s[0]
    else:
        _h2h_s1, _h2h_s2 = rec_s[1] - rec_s[0], rec_s[0] - rec_s[1]

    # Days since last match — usa la funzione condivisa dal shared engine
    import datetime as _dt2
    _today      = _dt2.date.today()
    _today_days = _today.year * 365 + _today.month * 30 + _today.day
    _days1 = days_since_last(last_match_date_dict.get(nombre1))
    _days2 = days_since_last(last_match_date_dict.get(nombre2))

    # Weeks load — usa la funzione condivisa dal shared engine
    wload1 = weeks_load_fn(nombre1, match_load_dict, _today_days)
    wload2 = weeks_load_fn(nombre2, match_load_dict, _today_days)

    # Upset tendency — usa la funzione condivisa dal shared engine
    upt1 = upset_tendency_fn(upset_hist_dict.get(nombre1, []))
    upt2 = upset_tendency_fn(upset_hist_dict.get(nombre2, []))

    # Late round win rate — usa la funzione condivisa dal shared engine
    lrwr1 = late_round_wr_fn(late_round_hist_dict.get(nombre1, []))
    lrwr2 = late_round_wr_fn(late_round_hist_dict.get(nombre2, []))

    # Level weight
    lev_w = LEVEL_MULT_LABEL.get(livello, 1.0)

    ann_input = pd.DataFrame([{
        'log_rank_ratio':   np.log1p(r2) - np.log1p(r1),
        'log_pts_ratio':    np.log1p(pts1) - np.log1p(pts2),
        'diff_elo':         elo1 - elo2,
        'diff_elo_overall':  elo_ov1 - elo_ov2,
        'diff_streak':      float(strk1 - strk2),
        'diff_recent_form':  form1 - form2,
        'surface_enc':      float(SURFACE_MAP.get(superficie, 0)),
        'tourney_level':    float(LEVEL_LABEL.get(livello, 3)),
        'round_enc':        float(ROUND_MAP_STR.get(turno, 3)),
        'is_best_of_5':     1.0 if best_of == 5 else 0.0,
        'diff_skill':       skill1 - skill2,
        'diff_home':        home1 - home2,
        'diff_fatigue':     fat1 - fat2,
        'diff_momentum':    mom1_val - mom2_val,
        'diff_h2h':         diff_h2h,
        'diff_h2h_surface': float(_h2h_s1 - _h2h_s2),
        'diff_days_since_last': float(_days1 - _days2),
        'diff_weeks_load':  wload1 - wload2,
        'diff_ace':         sa1.get('aces', 0) - sa2.get('aces', 0),
        'diff_1st_pct':     (sa1.get('first_serve_pct', 62) - sa2.get('first_serve_pct', 62)) / 100,
        'diff_1st_won':     (sa1.get('serve_win', 65) - sa2.get('serve_win', 65)) / 100,
        'diff_2nd_won':     (sa1.get('second_serve_win', 50) - sa2.get('second_serve_win', 50)) / 100,
        'diff_bp_saved':    (sa1.get('bp_saved', 60) - sa2.get('bp_saved', 60)) / 100,
        'diff_return_pct':  rtn_pct1 - rtn_pct2,
        'diff_bp_conv':     bp_conv1 - bp_conv2,
        'diff_return_1st':  rtn_1st1 - rtn_1st2,
        # Opponent Quality Score (ultimi 5 match) — usa calc_oq dal shared engine
        'diff_opponent_quality': calc_oq(opp_quality_dict.get(nombre1, []), r1)
                                - calc_oq(opp_quality_dict.get(nombre2, []), r2),
        'diff_upset_tendency':  upt1 - upt2,
        'diff_late_round_wr':   lrwr1 - lrwr2,
        'level_weight':         lev_w,
    }])

    input_sc = finale_scaler.transform(ann_input[ANN_FEATURES].values)
    input_t  = torch.tensor(input_sc.astype(np.float32))

    # ── Predizione con modello finale (usa wrapper che passa modelo_finale) ──
    prob_j1, modello_usato = _predici_locale(input_sc, input_t)

    # Calibrazione (se calibratore disponibile nel modelo_finale)
    calibrator = modelo_finale.get('calibrator', None)
    prob_j1 = _apply_cal(calibrator, prob_j1)

    # Uncertainty: top-5 ANN ensemble
    uncertainty_result = None
    ann_std = 0.0
    if HAS_UNCERTAINTY:
        top5_data = modelo_finale.get('ann_top5_uncertainty') or modelo_finale.get('ann_top5')
        if top5_data:
            try:
                top5_models = [_build_ann(c) for c in top5_data]
                uncertainty_result = predict_with_uncertainty(input_t, top5_models)
                ann_std = uncertainty_result['p_std']
                raw_top5_mean = uncertainty_result['p_mean']
                prob_j1 = _apply_cal(calibrator, raw_top5_mean) if calibrator is not None else raw_top5_mean
            except Exception:
                pass

    # Meta-Confidence
    meta_result = None
    if meta_scorer is not None:
        try:
            feat_dict = ann_input.iloc[0].to_dict()
            meta_result = meta_scorer.score(
                prob_calibrated=prob_j1,
                ann_std=ann_std,
                features=feat_dict,
            )
        except Exception:
            pass

    # Risultato
    st.divider()
    col_res_izq, col_res_der = st.columns([1, 3])

    with col_res_izq:
        if prob_j1 > 0.5:
            st.markdown("## 🎾")

    with col_res_der:
        if prob_j1 > 0.5:
            st.success(f"🏆 Vincitore: **{nombre1}**")
            st.metric("Confidenza", f"{prob_j1:.1%}", delta=f"Modello: {modello_usato}")
        else:
            st.error(f"🏆 Vincitore: **{nombre2}**")
            st.metric("Confidenza", f"{(1-prob_j1):.1%}", delta=f"Modello: {modello_usato}")

        # Barra probabilità
        prob_display = prob_j1 if prob_j1 > 0.5 else 1 - prob_j1
        vincitore    = nombre1 if prob_j1 > 0.5 else nombre2
        perdente     = nombre2 if prob_j1 > 0.5 else nombre1
        fig_bar = go.Figure(go.Bar(
            x=[prob_display, 1 - prob_display],
            y=[vincitore, perdente],
            orientation='h',
            marker_color=['#00CC96', '#EF553B'],
            text=[f"{prob_display:.1%}", f"{1-prob_display:.1%}"],
            textposition='auto'
        ))
        fig_bar.update_layout(
            height=160, xaxis=dict(range=[0, 1], visible=False),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=10, b=10), showlegend=False
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Quote ideali con margine bookmaker (overround ~8.1%, come 1.85 su 50/50)
        BK_OVERROUND = 2 / 1.85  # ≈ 1.0811
        q_ideale_v = 1 / (prob_display * BK_OVERROUND)
        q_ideale_p = 1 / ((1 - prob_display) * BK_OVERROUND)
        cq1, cq2 = st.columns(2)
        cq1.metric(
            label=f"📊 Quota Ideale — {vincitore}",
            value=f"{q_ideale_v:.2f}",
            help="Quota già scontata del margine bookmaker (~8%). Se trovi una quota superiore, hai un edge."
        )
        cq2.metric(
            label=f"📊 Quota Ideale — {perdente}",
            value=f"{q_ideale_p:.2f}",
            help="Quota già scontata del margine bookmaker (~8%). Se trovi una quota superiore, hai un edge."
        )

        # Uncertainty band
        if uncertainty_result is not None:
            uf = uncertainty_result['uncertain_flag']
            uf_icon = {'LOW': '🟢', 'MEDIUM': '🟡', 'HIGH': '🔴'}.get(uf, '⚪')
            ci_lo = uncertainty_result['ci_low']
            ci_hi = uncertainty_result['ci_high']
            if prob_j1 >= 0.5:
                ci_show = f"{ci_lo:.0%} – {ci_hi:.0%}"
            else:
                ci_show = f"{1-ci_hi:.0%} – {1-ci_lo:.0%}"
            n_mod = uncertainty_result['n_models']
            st.caption(
                f"{uf_icon} **Concordanza modelli** ({n_mod} ANN): "
                f"intervallo {ci_show}  |  std={uncertainty_result['p_std']:.3f}"
            )

        # Meta-Confidence Score
        if meta_result is not None:
            score = meta_result['score']
            label = meta_result['label']
            label_icon = {'HIGH': '🟢', 'MEDIUM': '🟡', 'LOW': '🔴'}.get(label, '⚪')
            with st.expander(f"{label_icon} Meta-Confidence: **{score}/100 — {label}**"):
                for reason in meta_result['reasons']:
                    st.markdown(f"- {reason}")
                bd = meta_result['breakdown']
                hist_str = f"{bd['bucket_accuracy']}/60" if bd['has_historical'] else "N/D"
                st.caption(
                    f"Bucket storico: {hist_str} · "
                    f"Upset tendency: {bd['upset_tendency']}/40"
                )


# SCOMMESSE SPECIALI (VALUE BET) â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
from scipy.stats import poisson
import sys
import __main__

# 1. Mettiamo la classe ESATTAMENTE dove serve
class TennisANN_Reg(nn.Module):
    """Wide & Deep ANN per Regressione (Scommesse Speciali)."""
    def __init__(self, input_dim, hidden_layers=[64, 32], dropout=0.3):
        super().__init__()
        self.wide = nn.Linear(input_dim, 1)
        layers = []
        in_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.Dropout(dropout))
            in_dim = h
        self.deep = nn.Sequential(*layers)
        self.deep_out = nn.Linear(in_dim, 1)
        
    def forward(self, x):
        wide_out = self.wide(x)
        deep_out = self.deep_out(self.deep(x))
        return wide_out + deep_out

# 2. IL TRUCCO MAGICO: leghiamo la classe a tutti i moduli che Streamlit potrebbe usare
setattr(__main__, 'TennisANN_Reg', TennisANN_Reg)
if 'main' in sys.modules:
    setattr(sys.modules['main'], 'TennisANN_Reg', TennisANN_Reg)

st.divider()
st.header("🎲 Value Bet: Ace, Doppi Falli, Break")

special_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'prediccion', 'modelos_special_bets.pkl')

try:
    special_data = joblib.load(special_path)
    hist = special_data['history']
    
    c_bet1, c_bet2, c_bet3 = st.columns(3)
    with c_bet1:
        bet_mercato = st.selectbox("Mercato", ["Totale Ace", "Totale Doppi Falli", "Totale Break"])
    with c_bet2:
        bet_linea = st.number_input("Linea Bookmaker (es. 12.5)", value=12.5, step=0.5)
    with c_bet3:
        bet_quota = st.number_input("Quota OVER offerta", value=1.85, step=0.01)

    if st.button("🧮 Calcola Valore Scommessa", type="secondary", use_container_width=True):
        
        # Costruiamo la chiave dinamica (es. "Totale Ace (Best of 3)")
        target_key = f"{bet_mercato} (Best of {best_of})"
        
        if target_key not in special_data['models']:
            st.warning(f"Modello non addestrato per: {target_key}")
            st.stop()
            
        # Estraiamo il modello vincitore e il suo scaler
        model_info = special_data['models'][target_key]
        spec_scaler = model_info['scaler']
        m_type = model_info['type']
        
        # --- Feature dinamiche per J1 e J2 ---
        def get_special_stats(p):
            # Valori di default se il giocatore è nuovo
            if p not in hist or len(hist[p]['games_played']) == 0:
                return {'ace_rate': 0.05, 'df_rate': 0.02, 'bp_faced_rate': 0.08, 'hold_pct': 0.80, 
                        'ace_allowed_rate': 0.05, 'bp_created_rate': 0.08, 'ret_2nd_win_pct': 0.50}
            
            h = hist[p]
            t_sv = sum(h['games_played']) or 1
            t_ret = sum(h['games_return']) or 1
            t_2nd_ret = sum(h.get('ret_2nd_played', [1])) or 1
            
            return {
                'ace_rate': sum(h['ace_for']) / t_sv,
                'df_rate': sum(h['df_for']) / t_sv,
                'bp_faced_rate': sum(h['bp_faced']) / t_sv,
                'hold_pct': 1.0 - (sum(h.get('bp_lost', [0])) / t_sv),
                'ace_allowed_rate': sum(h['ace_against']) / t_ret,
                'bp_created_rate': sum(h['bp_created']) / t_ret,
                'ret_2nd_win_pct': sum(h.get('ret_2nd_won', [0])) / t_2nd_ret
            }

        s1, s2 = get_special_stats(nombre1), get_special_stats(nombre2)
        exp_games = 22 if best_of == 3 else 38
        
        # Dizionario feature nello STESSO identico ordine del file di addestramento!
        feat_dict = {
            'surface': SURFACE_MAP.get(superficie, 0),
            'court_speed': court_spd,
            'court_ace_pct': court_ace,  # court ace % per il torneo
            'best_of': best_of,
            'sum_ht': h1 + h2,
            'diff_ht': abs(h1 - h2),
            'prob_tb': s1['hold_pct'] * s2['hold_pct'], 
            'avg_pressure': (s1['ret_2nd_win_pct'] + s2['ret_2nd_win_pct']) / 2.0,
            'proj_total_aces': ((s1['ace_rate'] + s2['ace_allowed_rate'])/2.0 + (s2['ace_rate'] + s1['ace_allowed_rate'])/2.0) * (exp_games/2),
            'proj_total_dfs': (s1['df_rate'] + s2['df_rate']) * (exp_games/2),
            'proj_total_bps': ((s1['bp_faced_rate'] + s2['bp_created_rate'])/2.0 + (s2['bp_faced_rate'] + s1['bp_created_rate'])/2.0) * (exp_games/2),
            'diff_ace_rate': abs(s1['ace_rate'] - s2['ace_rate']),
            'diff_def_rate': abs(s1['ace_allowed_rate'] - s2['ace_allowed_rate'])
        }
        
        df_spec = pd.DataFrame([feat_dict])
        x_spec_sc = spec_scaler.transform(df_spec)
        
        # --- Predizione Dinamica Universale (Gestisce Singoli, Avg e Stacking) ---
        def get_single_pred(m, x):
            if isinstance(m, nn.Module):
                m.eval()
                with torch.no_grad():
                    return float(m(torch.tensor(x, dtype=torch.float32)).cpu().numpy()[0])
            else:
                return float(m.predict(x)[0])

        if m_type == 'single':
            lambda_pred = get_single_pred(model_info['model'], x_spec_sc)
        elif m_type == 'avg':
            lambda_pred = float(np.mean([get_single_pred(m, x_spec_sc) for m in model_info['models']]))
        elif m_type == 'stacking':
            base_preds = [get_single_pred(m, x_spec_sc) for m in model_info['models']]
            lambda_pred = float(model_info['meta'].predict(np.array([base_preds]))[0])
            
        # Calcolo Poisson
        k_under = int(np.floor(bet_linea))
        prob_under = poisson.cdf(k_under, max(0.1, lambda_pred))
        prob_over = 1.0 - prob_under
        
        q_over_equa = 1 / prob_over if prob_over > 0.001 else 999.0
        q_under_equa = 1 / prob_under if prob_under > 0.001 else 999.0
        
        # --- Calcolo Affidabilità Storica e Metriche (Dinamico dal file!) ---
        mae_storico = model_info.get('mae', 0)
        media_storica = model_info.get('avg_value', 1)
        devianza_storica = model_info.get('devianza', 0)
        
        err_pct = (mae_storico / media_storica) * 100 if media_storica > 0 else 100
        affidabilita = max(0, 100 - err_pct)
        
        # --- Output visuale ---
        st.markdown(f"### 🎯 Valore Atteso: **{lambda_pred:.1f}** {bet_mercato.split(' ')[-1]} (Modello: *{m_type}*)")
        
        # Mostriamo le metriche
        c_met1, c_met2 = st.columns(2)
        c_met1.metric(
            label="Affidabilità Modello", 
            value=f"{affidabilita:.1f}%", 
            help=f"Calcolata su un errore medio (MAE) di {mae_storico:.2f} rispetto a una media di {media_storica:.1f} eventi."
        )
        c_met2.metric(
            label="Devianza Poisson", 
            value=f"{devianza_storica:.3f}", 
            help="Più si avvicina a 1.0, più il modello è perfetto nel calcolare le probabilità."
        )
        st.write("---")
        
        col_o, col_u = st.columns(2)
        edge = (bet_quota / q_over_equa) - 1
        
        with col_o:
            if edge > 0:
                st.success(f"📈 **OVER {bet_linea}**\n\nProb: **{prob_over*100:.1f}%** | Quota Equa: **{q_over_equa:.2f}**\n\n🔥 **VALUE BET! Edge: +{edge*100:.1f}%**")
            else:
                st.error(f"📉 **OVER {bet_linea}**\n\nProb: **{prob_over*100:.1f}%** | Quota Equa: **{q_over_equa:.2f}**\n\n❌ Nessun Valore")
                
        with col_u:
            st.info(f"🔽 **UNDER {bet_linea}**\n\nProb: **{prob_under*100:.1f}%** | Quota Equa: **{q_under_equa:.2f}**")

except FileNotFoundError:
    st.info("ℹ️ File `modelos_special_bets.pkl` non trovato. Fai il training per abilitare il calcolatore.")
except Exception as e:
    st.error(f"Errore: {e}")
