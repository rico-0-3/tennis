import streamlit as st
import pandas as pd
import joblib
import sys
import os

ruta_scraping = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scraping'))
if ruta_scraping not in sys.path:
    sys.path.append(ruta_scraping)

st.set_page_config(page_title="Classifica ATP", page_icon="🏆", layout="wide")

st.title("🏆 Classifica ATP & Profili Giocatori")

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# ── Pulsante aggiornamento ────────────────────────────────────────────────────
from actualizador_maestro import ejecutar_pipeline

if st.button("🔄 Cerca Nuove Partite e Aggiorna Classifica", type="primary"):
    with st.spinner("I robot stanno lavorando... Potrebbe richiedere alcuni minuti."):
        exito = ejecutar_pipeline()
        if exito:
            st.success("✅ Database aggiornato con successo!")
            st.rerun()
        else:
            st.error("Si è verificato un errore nell'aggiornamento. Controlla la console.")

st.markdown("---")

# ── Caricamento dati ──────────────────────────────────────────────────────────
ruta_scraping  = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scraping'))
ruta_ranking   = os.path.join(ruta_scraping, "ranking_2026.csv")
ruta_perfiles  = os.path.join(ruta_scraping, "perfiles_jugadores.pkl")

try:
    df_ranking = pd.read_csv(ruta_ranking)
    perfiles   = joblib.load(ruta_perfiles)
except Exception as e:
    st.warning("Dati non trovati nella cartella scraping. Hai già eseguito l'aggiornamento?")
    st.error(f"Errore tecnico: {e}")
    st.stop()

col1, col2 = st.columns([1, 1])

def extraer_nombre_real(url):
    try:
        slug = str(url).split('/')[5]
        return slug.replace('-', ' ').title()
    except:
        return ""

df_ranking['Nome Completo'] = df_ranking['url_perfil'].apply(extraer_nombre_real)

# ── Colonna 1: Classifica ─────────────────────────────────────────────────────
with col1:
    st.subheader("📋 Classifica Mondiale")
    df_mostrar = df_ranking[['rank', 'Nome Completo', 'points']].copy()
    df_mostrar.columns = ['Pos.', 'Giocatore', 'Punti']
    st.dataframe(df_mostrar.set_index('Pos.'), height=600, use_container_width=True)

# ── Colonna 2: Profilo giocatore ──────────────────────────────────────────────
with col2:
    st.subheader("🔍 Analizzatore Profilo")
    lista_jugadores = [j for j in df_ranking['Nome Completo'].tolist() if j != ""]
    jugador_sel = st.selectbox("Cerca giocatore:", lista_jugadores)

    if jugador_sel in perfiles:
        p = perfiles[jugador_sel]

        c1, c2, c3 = st.columns(3)
        c1.metric("Ranking", int(p.get('rank', 0)))
        c2.metric("Punti",   f"{int(p.get('points', 0)):,}".replace(",", "."))
        c3.metric("Paese",   p.get('ioc', 'N/D'))

        st.markdown("#### 🧬 Attributi Fisici")
        c4, c5 = st.columns(2)
        c4.info(f"**Età:** {p.get('age', 'N/D')} anni")
        c5.info(f"**Altezza:** {p.get('ht', 'N/D')} cm")

        st.markdown("#### 🎾 Statistiche Avanzate")
        st.write(f"**1° Servizio Vinto:** {p.get('serve_win', 0)}%")
        st.write(f"**Break Point Salvati:** {p.get('bp_saved', 0)}%")
        st.write(f"**Ace per partita:** {p.get('aces', 0):.1f}")
        st.write(f"**Momento (Ultimi 5):** {int(p.get('momentum', 0) * 100)}%")

    else:
        st.info("Nessuna statistica avanzata disponibile per questo giocatore.")