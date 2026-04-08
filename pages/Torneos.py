# pages/Torneos.py
"""
Pagina: Predizioni Prossime Partite
Legge prediccion/predicciones_proximas.json e mostra le partite con prediction bar,
%, H2H e quote ideali, filtrabili per livello torneo.
"""

import os
import json
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="Prossime Partite",
    page_icon="🗓️",
    layout="wide",
)

hide_st = """
<style>
#MainMenu {visibility: hidden;}
footer    {visibility: hidden;}
header    {visibility: hidden;}
</style>
"""
st.markdown(hide_st, unsafe_allow_html=True)

# ── Path JSON ─────────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_PATH = os.path.join(ROOT, "prediccion", "predicciones_proximas.json")

# ── Carica dati ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def carica_predizioni():
    if not os.path.exists(JSON_PATH):
        return None, []
    with open(JSON_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("updated_at"), data.get("partite", [])

updated_at, partite = carica_predizioni()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🗓️ Predizioni Prossime Partite")

if updated_at is None:
    st.warning(
        "Nessuna predizione disponibile. "
        "Avvia la pipeline di aggiornamento dalla pagina **Aggiorna Dati**."
    )
    st.stop()

try:
    from datetime import datetime
    dt = datetime.fromisoformat(updated_at)
    st.caption(f"Aggiornato il: {dt.strftime('%d/%m/%Y %H:%M')} UTC")
except Exception:
    st.caption(f"Aggiornato il: {updated_at}")

if not partite:
    st.info("Nessuna partita trovata per questa settimana.")
    st.stop()

# ── Filtro livello ────────────────────────────────────────────────────────────
livelli_presenti = sorted({p["livello"] for p in partite if not p.get("skipped")})
opzioni = ["Tutti"] + livelli_presenti
livello_sel = st.selectbox("🏆 Filtra per livello torneo", opzioni)

partite_ok   = [p for p in partite if not p.get("skipped")]
partite_skip = [p for p in partite if p.get("skipped")]

if livello_sel != "Tutti":
    partite_ok = [p for p in partite_ok if p["livello"] == livello_sel]

if not partite_ok:
    st.info("Nessuna partita con predizione per il livello selezionato.")
else:
    st.markdown(f"**{len(partite_ok)} partite** trovate")

# ── Cards partite ─────────────────────────────────────────────────────────────
COLORI_LIVELLO = {
    "Grand Slam":   "#D4AF37",
    "Masters 1000": "#1565C0",
    "ATP 500":      "#2E7D32",
    "ATP 250":      "#558B2F",
    "Challenger":   "#6A1B9A",
}
SURF_EMOJI = {"Hard": "🔵", "Clay": "🟠", "Grass": "🟢"}

# Raggruppa per torneo
tornei = {}
for p in partite_ok:
    tornei.setdefault(p["torneo"], []).append(p)

for torneo_name, matches in tornei.items():
    livello_t  = matches[0]["livello"]
    colore     = COLORI_LIVELLO.get(livello_t, "#444")
    superficie_t = matches[0]["superficie"]
    surf_ico   = SURF_EMOJI.get(superficie_t, "⚪")

    st.markdown(
        f"<h3 style='color:{colore};'>{surf_ico} {torneo_name} "
        f"<span style='font-size:0.7em;font-weight:normal;'>({livello_t} · {superficie_t})</span></h3>",
        unsafe_allow_html=True
    )

    for p in matches:
        with st.container():
            c1, c2 = st.columns([3, 1])

            with c1:
                prob_p1 = p["prob_p1"]
                fav     = p["favorito"]
                sfav    = p["p2"] if fav == p["p1"] else p["p1"]

                fig = go.Figure(go.Bar(
                    x=[prob_p1, 1 - prob_p1],
                    y=[p["p1"], p["p2"]],
                    orientation="h",
                    marker_color=["#00CC96" if prob_p1 >= 0.5 else "#EF553B",
                                  "#00CC96" if prob_p1 < 0.5  else "#EF553B"],
                    text=[f"{prob_p1:.0%}", f"{1-prob_p1:.0%}"],
                    textposition="auto",
                ))
                fig.update_layout(
                    height=120,
                    margin=dict(l=0, r=0, t=4, b=4),
                    xaxis=dict(range=[0, 1], visible=False),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                st.markdown(f"**Turno:** {p['turno']}")
                st.markdown(f"**Data:** {p['data']}")
                st.markdown(f"**H2H:** {p['p1']} {p['h2h_p1']} – {p['h2h_p2']} {p['p2']}")
                st.markdown(
                    f"**Quote ideali:** {fav} `{p['quota_fav']}` · {sfav} `{p['quota_sfav']}`"
                )
                st.caption(f"Modello: {p.get('modello', 'N/D')} · Confidenza: {p['confidenza']:.0%}")

        st.divider()

# ── Partite skippate ──────────────────────────────────────────────────────────
if partite_skip:
    with st.expander(f"⏭️ Partite senza predizione ({len(partite_skip)})"):
        for p in partite_skip:
            st.markdown(
                f"**{p['p1']} vs {p['p2']}** — {p.get('torneo','?')} — "
                f"_{p.get('motivo', 'motivo sconosciuto')}_"
            )
