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

# Import court speed helper
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scraping'))
try:
    from court_speed_helper import get_court_stats_latest
except:
    def get_court_stats_latest(name, surf): return (11.5, 1.10) if surf != 'Clay' else (6.5, 0.65)

st.set_page_config(page_title="ATP Predittore 2026", page_icon="🎾", layout="wide")

st.title("🎾 ATP Prediction Pro 2026")

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;} /* Oculta los 3 puntitos de arriba a la derecha */
            footer {visibility: hidden;} /* Oculta el "Made with Streamlit" de abajo */
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

st.markdown("""
Questa applicazione utilizza un **ANN Top-5 Average** — media delle probabilità di 5 reti neurali con architetture diverse selezionate da Optuna:

| Modello | Tipo | Ruolo |
|---------|------|-------|
| 🧠 **ANN ×5** | Reti Neurali (Wide & Deep) | 5 architetture diverse, ciascuna cattura pattern diversi nei dati |
| 🌲 **LightGBM** | Gradient Boosting ad Albero | Alberi decisionali in sequenza — usato come confronto |
| 🌳 **XGBoost** | Gradient Boosting ad Albero | Simile a LightGBM con regolarizzazione diversa — usato come confronto |

**Come funziona:**
1. **Optuna** testa centinaia di architetture neurali diverse (strati, neuroni, dropout, learning rate...)
2. Le **5 migliori** vengono selezionate e ri-addestrate su tutti i dati
3. In fase di predizione, ognuna produce una probabilità → la **media** delle 5 è il risultato finale

Perché la media? Ogni rete ha punti di forza diversi; mediando si riduce l'errore individuale (**ensemble diversity**).

Il sistema analizza **29 feature** tra cui:
* 📊 **Gerarchia:** Ranking, Punti, Elo (globale e per superficie).
* ⚔️ **Storico:** Scontri diretti (H2H), forma recente.
* 🧠 **Momento:** Striscia, momentum, fatica.
* 🎯 **Tecnica:** Ace, servizio, break point, resa al ritorno.
""")

st.write("---")


# ─── Definizione ANN v3 (stessa architettura di train_ann.py) ────────────────
DEFAULT_INTERACTION_PAIRS = [
    (5, 12), (0, 15), (5, 15), (12, 14), (0, 1),
    (5, 16), (12, 16), (6, 15), (14, 15), (1, 12),
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
    'diff_rank', 'diff_rank_points', 'diff_seed', 'diff_age', 'diff_ht',
    'diff_elo', 'diff_streak',
    'surface_enc', 'tourney_level', 'round_enc', 'draw_size',
    'diff_hand',
    'diff_skill', 'diff_home',
    'diff_fatigue', 'diff_momentum', 'diff_h2h',
    'diff_ace', 'diff_1st_won', 'diff_bp_saved',
    'diff_return_pct', 'diff_bp_conv', 'diff_return_1st',
    'court_ace_pct', 'court_speed',
    'log_rank_ratio', 'log_pts_ratio',
    'diff_elo_overall', 'diff_recent_form',
]  # 29 feature (v3.1)

SURFACE_MAP   = {'Hard': 0, 'Clay': 1, 'Grass': 2}
LEVEL_MAP     = {'G': 5, 'M': 4, 'A': 3, 'F': 4, 'C': 2, 'S': 1, 'E': 0}
ROUND_MAP_STR = {'Finale': 7, 'Semifinale': 6, 'Quarti': 5, '16mi': 4,
                 '32mi': 3, '64mi': 2, '128mi': 1, 'Round Robin': 4}

# ─── Caricamento risorse ─────────────────────────────────────────────────────
@st.cache_resource
def cargar_todo():
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

    # ANN globale v3
    ann_global = None; ann_scaler = None; ann_calibrator = None
    ann_top5_models = []   # lista di modelli ANN per Top-5 averaging
    elo_surface = {}     # {(player, surface): elo_float}
    streak_players = {}
    momentum_surface = {}  # {(player, surface): [last 10 results]}

    ann_base = pp('modelo_ann.pth')
    cfg_base = pp('ann_config.json')
    sca_base = pp('scaler_ann.pkl')

    if os.path.exists(ann_base) and os.path.exists(cfg_base) and os.path.exists(sca_base):
        try:
            with open(cfg_base) as f: cfg = json.load(f)
            n_inter = cfg.get('n_interactions', N_INTERACTIONS)
            i_pairs = cfg.get('interaction_pairs', DEFAULT_INTERACTION_PAIRS)
            ann_global = TennisANNv3(cfg['input_dim'], cfg['hidden_layers'],
                                     cfg['dropout'], n_inter, i_pairs)
            ann_global.load_state_dict(torch.load(ann_base, map_location='cpu'))
            ann_global.eval()
            ann_scaler = joblib.load(sca_base)
        except Exception as e:
            st.warning(f"ANN globale non caricata: {e}")

    # Top-5 ANN models
    top5_path = pp('modelo_ann_top5.pkl')
    if os.path.exists(top5_path) and ann_scaler is not None:
        try:
            top5_data = joblib.load(top5_path)
            for sd, hp_i in zip(top5_data['state_dicts'], top5_data['configs']):
                hl_i  = hp_i['hidden_layers']
                dr_i  = hp_i['dropout']
                ipairs_i = hp_i.get('interaction_pairs', DEFAULT_INTERACTION_PAIRS)
                ninter_i = hp_i.get('n_interactions', len(ipairs_i))
                m = TennisANNv3(len(ANN_FEATURES), hl_i, dr_i, ninter_i, ipairs_i)
                m.load_state_dict(sd)
                m.eval()
                ann_top5_models.append(m)
        except Exception as e:
            st.warning(f"Top-5 ANN non caricati: {e}")

    # Calibratore (Platt scaling)
    cal_path = pp('calibrator_ann.pkl')
    if os.path.exists(cal_path):
        try: ann_calibrator = joblib.load(cal_path)
        except: pass

    # Modelli ensemble: LGB, XGB, Meta-LR
    lgb_model = None; xgb_model = None; meta_model = None
    lgb_path  = pp('modelo_lgb.pkl')
    xgb_path  = pp('modelo_xgb.pkl')
    meta_path = pp('modelo_meta_lr.pkl')
    if os.path.exists(lgb_path):
        try: lgb_model = joblib.load(lgb_path)
        except: pass
    if os.path.exists(xgb_path):
        try: xgb_model = joblib.load(xgb_path)
        except: pass
    if os.path.exists(meta_path):
        try: meta_model = joblib.load(meta_path)
        except: pass

    # Elo + streak + momentum + elo_overall + recent_form
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
            ann_global, ann_scaler, ann_calibrator, ann_top5_models,
            elo_surface, streak_players,
            momentum_surface, elo_overall, recent_form,
            lgb_model, xgb_model, meta_model)


(stats_dict, perfiles, df_history, ranking_dict,
 ann_global, ann_scaler, ann_calibrator, ann_top5_models,
 elo_surface, streak_players,
 momentum_surface, elo_overall, recent_form,
 lgb_model, xgb_model, meta_model) = cargar_todo()

if ann_global is None:
    st.error("❌ Modello ANN non trovato. Assicurati che `modelo_ann.pth`, `ann_config.json` e `scaler_ann.pkl` siano nella cartella `prediccion/`.")
    st.stop()


def get_skill(p, s): return stats_dict.get((p, s), 0.5)

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
        paper_bgcolor='rgba(0,0,0,0)', showlegend=True, height=450,
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


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configurazione")

    st.subheader("🧠 Cervello dell'IA")
    st.info("Usando: **ANN v3.1** — Wide&Deep + Residual + GBM Ensemble (29 feature, calibrato) 🔥")

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
    livello       = st.selectbox("Livello Torneo", ["Grand Slam", "Masters 1000", "ATP 500", "ATP 250"],
                                  help="G=Grand Slam | M=Masters | A=500 | C=250")
    draw_sz       = st.selectbox("Tabellone", [128, 96, 64, 32, 16], index=2)

    LEVEL_LABEL = {'Grand Slam': 5, 'Masters 1000': 4, 'ATP 500': 3, 'ATP 250': 2}


# ─── GIOCATORI ────────────────────────────────────────────────────────────────
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

    st.markdown("##### ⚡ Stato")
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

    st.markdown("##### ⚡ Stato")
    mom2 = st.slider("Momento (%)", 0, 100, key="m2") / 100

    with st.expander("Ultimi 5 partiti"):
        mostrar_historial_detallado(st.session_state.get('l5_2', []))

    fat2 = st.number_input("Fatica (min)", 0, 1000, 0, key="f2")


# ─── H2H + RADAR ─────────────────────────────────────────────────────────────
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


# ─── PREDIZIONE ──────────────────────────────────────────────────────────────
if st.button("🔮 PREDICI con ANN v3", type="primary", use_container_width=True):

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

    ann_input = pd.DataFrame([{
        'diff_rank':        r2 - r1,
        'diff_rank_points': pts1 - pts2,
        'diff_seed':        seed2_eff - seed1_eff,
        'diff_age':         a1 - a2,
        'diff_ht':          h1 - h2,
        'diff_elo':         elo1 - elo2,
        'diff_streak':      float(strk1 - strk2),
        'surface_enc':      float(SURFACE_MAP.get(superficie, 0)),
        'tourney_level':    float(LEVEL_LABEL.get(livello, 3)),
        'round_enc':        float(ROUND_MAP_STR.get(turno, 3)),
        'draw_size':        float(draw_sz),
        'diff_hand':        hand_map_d.get(hand1, 0) - hand_map_d.get(hand2, 0),
        'diff_skill':       skill1 - skill2,
        'diff_home':        home1 - home2,
        'diff_fatigue':     fat1 - fat2,
        'diff_momentum':    mom1_val - mom2_val,
        'diff_h2h':         diff_h2h,
        'diff_ace':         sa1.get('aces', 0) - sa2.get('aces', 0),
        'diff_1st_won':     (sa1.get('serve_win', 65) - sa2.get('serve_win', 65)) / 100,
        'diff_bp_saved':    (sa1.get('bp_saved', 60) - sa2.get('bp_saved', 60)) / 100,
        'diff_return_pct':  rtn_pct1 - rtn_pct2,
        'diff_bp_conv':     bp_conv1 - bp_conv2,
        'diff_return_1st':  rtn_1st1 - rtn_1st2,
        'court_ace_pct':    court_ace,
        'court_speed':      court_spd,
        'log_rank_ratio':   np.log1p(r2) - np.log1p(r1),
        'log_pts_ratio':    np.log1p(pts1) - np.log1p(pts2),
        'diff_elo_overall':  elo_ov1 - elo_ov2,
        'diff_recent_form':  form1 - form2,
    }])

    input_sc = ann_scaler.transform(ann_input[ANN_FEATURES])
    input_t  = torch.tensor(input_sc.astype(np.float32))

    # ─── ANN Top-5 Average (miglior metodo in training) ──────────────────────
    use_top5 = len(ann_top5_models) >= 2
    if use_top5:
        top5_probs = []
        for m in ann_top5_models:
            m.eval()
            with torch.no_grad():
                logit_i = m(input_t).item()
            top5_probs.append(1 / (1 + np.exp(-logit_i)))
        prob_j1 = float(np.mean(top5_probs))
        modello_usato = f"ANN Top-{len(ann_top5_models)} Avg"
    else:
        # Fallback: ANN singola con Platt scaling
        ann_global.eval()
        with torch.no_grad():
            logit = ann_global(input_t).item()
        if ann_calibrator is not None:
            prob_j1 = ann_calibrator.predict_proba(np.array([[logit]]))[0, 1]
        else:
            prob_j1 = 1 / (1 + np.exp(-logit))
        modello_usato = "ANN v3"

    # Probabilità modelli individuali (per dettaglio)
    ann_global.eval()
    with torch.no_grad():
        ann_logit = ann_global(input_t).item()
    ann_prob = 1 / (1 + np.exp(-ann_logit))

    has_gbm = (lgb_model is not None and xgb_model is not None)
    lgb_prob = lgb_model.predict_proba(input_sc)[:, 1][0] if lgb_model is not None else None
    xgb_prob = xgb_model.predict_proba(input_sc)[:, 1][0] if xgb_model is not None else None

    # ─── Risultato ───────────────────────────────────────────────────────────
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

    # ─── Dettaglio modelli individuali ────────────────────────────────────────
    with st.expander("🔍 Dettaglio predizioni per modello"):
        if use_top5:
            st.markdown(f"**Strategia:** Media delle probabilità di **{len(ann_top5_models)} reti neurali** con architetture diverse (Top-5 da Optuna).")
            cols = st.columns(len(ann_top5_models))
            for idx, (col, p) in enumerate(zip(cols, top5_probs)):
                with col:
                    st.metric(f"ANN #{idx+1}", f"{p:.1%}")
            st.caption(f"**Risultato Top-{len(ann_top5_models)} Avg = {prob_j1:.1%}** — media aritmetica delle {len(ann_top5_models)} probabilità.")
        else:
            st.metric("🧠 ANN Best", f"{ann_prob:.1%}", delta="Rete Neurale singola")

        if has_gbm:
            st.divider()
            st.markdown("**Modelli ad albero (informativi):**")
            col_l, col_x = st.columns(2)
            with col_l:
                st.metric("🌲 LightGBM", f"{lgb_prob:.1%}", delta="Gradient Boosting")
            with col_x:
                st.metric("🌳 XGBoost", f"{xgb_prob:.1%}", delta="Gradient Boosting")