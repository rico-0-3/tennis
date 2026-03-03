import streamlit as st
import pandas as pd

st.set_page_config(page_title="Próximos Torneos", page_icon="🏆")

st.title("🏆 Próximos Partidos y Resultados")

# Cargar datos recientes
@st.cache_data
def cargar_partidos():
    # Usamos el archivo que tiene los datos de 2026
    try:
        df = pd.read_csv("atp_matches_2025_2026_raw.csv") # O el que uses para 2026
        return df
    except:
        return pd.DataFrame()

df = cargar_partidos()

if not df.empty:
    # Filtros
    torneos = df['tourney_name'].unique()
    torneo_sel = st.selectbox("Selecciona un Torneo:", torneos)
    
    # Filtrar por torneo
    df_t = df[df['tourney_name'] == torneo_sel]
    
    st.write(f"Partidos encontrados: {len(df_t)}")
    
    for index, row in df_t.iterrows():
        with st.expander(f"{row['round']}: {row['winner_name']} vs {row['loser_name']}"):
            st.write(f"**Resultado:** {row['score']}")
            st.write(f"**Superficie:** {row['surface']}")
            st.info("Ve a la pestaña 'Predictor' para analizar este matchup en detalle.")
else:
    st.warning("No se encontraron partidos recientes cargados.")