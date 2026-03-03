import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="Laboratorio IA", page_icon="🧠", layout="wide")

st.title("🧠 Laboratorio Dati")

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;} /* Oculta los 3 puntitos de arriba a la derecha */
            footer {visibility: hidden;} /* Oculta el "Made with Streamlit" de abajo */
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

st.markdown("### ⚔️ La Battaglia dei Modelli")
st.info("Per scegliere il 'cervello' di questa applicazione, abbiamo messo in competizione i vari algoritmi. Ecco i risultati in modo semplice.")

ruta_script = os.path.dirname(os.path.abspath(__file__))
ruta_raiz   = os.path.dirname(ruta_script)
ruta_pred   = os.path.join(ruta_raiz, "prediccion")

# ── Risultati comparazione: ultima versione prima, poi fallback ───────────────
try:
    path_finale = os.path.join(ruta_pred, "resultados_comparacion_finale.csv")
    path_classic= os.path.join(ruta_raiz, "resultados_comparacion.csv")
    path_ann    = os.path.join(ruta_pred, "resultados_ann.csv")

    if os.path.exists(path_finale):
        # File prodotto da train_ann.py — contiene tutto
        df_res = pd.read_csv(path_finale)
        df_res.rename(columns={'Modello': 'Modelo', 'Accuracy': 'Accuracy'}, errors='ignore', inplace=True)
        if 'Accuracy' not in df_res.columns and 'test_acc' in df_res.columns:
            df_res['Accuracy'] = df_res['test_acc']
    else:
        # Costruzione al volo
        df_res = pd.DataFrame()
        if os.path.exists(path_classic):
            df_res = pd.read_csv(path_classic)
        if os.path.exists(path_ann):
            df_ann_tmp = pd.read_csv(path_ann)
            best_acc = df_ann_tmp['test_acc'].max()
            df_res = pd.concat([df_res,
                pd.DataFrame([{'Modelo': 'ANN Globale', 'Accuracy': best_acc}])],
                ignore_index=True)

    if df_res.empty:
        raise FileNotFoundError

    if 'Acc. %' not in df_res.columns:
        df_res['Acc. %'] = (df_res['Accuracy'] * 100).round(2)

    df_res = df_res.sort_values('Accuracy', ascending=False).reset_index(drop=True)

    # Colori: ANN in arancio, classici in blu
    def colore(nome):
        n = str(nome)
        if 'ANN' in n or 'Rete' in n: return '#F97316'
        return '#2563EB'
    df_res['_color'] = df_res['Modelo'].apply(colore)

except FileNotFoundError:
    st.error("⚠️ Risultati mancanti. Esegui prima `aggiorna_tutto.py`.")
    st.stop()

# ── Grafico modello vincitore ─────────────────────────────────────────────────
col_graf, col_tabla = st.columns([2, 1])

with col_graf:
    st.subheader("🏆 Chi ha indovinato di più?")
    import plotly.graph_objects as go
    fig = go.Figure(go.Bar(
        x=df_res['Acc. %'],
        y=df_res['Modelo'],
        orientation='h',
        text=df_res['Acc. %'].apply(lambda v: f"{v:.2f}%"),
        textposition='outside',
        marker_color=df_res['_color'],
    ))
    fig.update_layout(
        xaxis=dict(range=[60, 80], title="Accuratezza (%)"),
        yaxis=dict(autorange='reversed'),
        title="Test Accuracy — tutti i modelli",
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=max(250, len(df_res)*38)
    )
    st.plotly_chart(fig, use_container_width=True)

with col_tabla:
    st.subheader("🥇 Il Vincitore")
    ganador = df_res.iloc[0]
    st.success(f"Il modello **{ganador['Modelo']}** è il migliore.")
    st.markdown(f"""
    Ha previsto correttamente il **{ganador['Acc. %']}%** delle partite di test.

    Per questo è il motore scelto per questa App.
    """)

st.divider()

# ── Spiegazione modelli ───────────────────────────────────────────────────────
st.header("🤓 Come 'pensa' ogni modello?")
st.markdown("Immagina di dover indovinare chi vince una partita. Questi modelli sono come diversi tipi di esperti:")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.image("https://cdn-icons-png.flaticon.com/512/2645/2645897.png", width=70)
    st.subheader("La Bilancia")
    st.caption("(Regressione Logistica)")
    st.write("""
    Somma e sottrae punti come una bilancia.
    * *\"Se il ranking è buono, +10 punti.\"*
    * *\"Se è stanco, -5 punti.\"*

    **Verdetto:** Veloce e logico, ma il tennis è più complesso.
    """)

with c2:
    st.image("https://cdn-icons-png.flaticon.com/512/1534/1534938.png", width=70)
    st.subheader("La Democrazia")
    st.caption("(Random Forest)")
    st.write("""
    Crea **100 mini-esperti** e li fa votare.
    * Esperto 1: *\"Vince Nadal sulla terra.\"*
    * Esperto 2: *\"Vince Federer sull'erba.\"*

    **Verdetto:** Stabile, ma fatica con pattern sottili.
    """)

with c3:
    st.image("https://cdn-icons-png.flaticon.com/512/2083/2083213.png", width=70)
    st.subheader("Il Perfezionista")
    st.caption("(XGBoost)")
    st.write("""
    Impara dagli errori passo dopo passo.

    1. Fa una previsione.
    2. Guarda gli errori.
    3. Crea un mini-modello per correggerli.
    4. Ripete centinaia di volte.

    **Verdetto:** Il più intelligente tra i classici.
    """)

with c4:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103652.png", width=70)
    st.subheader("La Rete Neurale")
    st.caption("(ANN — Deep Learning) 🔥")
    st.write("""
    Ispirata al cervello umano.
    * Strati di neuroni che imparano pattern complessi.
    * **Pesi temporali:** le partite recenti contano di più!
    * Adam optimizer + Dropout per generalizzare meglio.

    **Verdetto:** Il più potente — soprattutto con dati recenti.
    """)

st.divider()

# ── Analisi variabili (ANN o XGBoost) ────────────────────────────────────────
st.subheader("👀 Cosa guarda di più l'IA?")

path_ann = os.path.join(ruta_pred, "resultados_ann.csv")
path_imp = os.path.join(ruta_raiz, "importancia_real.csv")

if os.path.exists(path_ann):
    st.write("Architetture **Rete Neurale** testate durante il random search:")
    df_ann = pd.read_csv(path_ann)
    fig_ann = px.scatter(
        df_ann,
        x='log_loss', y='test_acc',
        color='dropout', size='batch_size',
        hover_data=['hidden_layers', 'lr', 'epochs'],
        title="Random Search ANN — Accuratezza vs Log Loss",
        labels={'test_acc': 'Accuratezza Test', 'log_loss': 'Log Loss'},
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig_ann, use_container_width=True)
    best_row = df_ann.loc[df_ann['test_acc'].idxmax()]
    st.info(f"🏆 Miglior architettura: `{best_row['hidden_layers']}` — Accuratezza: **{best_row['test_acc']:.2%}**")

elif os.path.exists(path_imp):
    st.write("Peso dato dal modello **XGBoost** a ogni variabile:")
    df_imp = pd.read_csv(path_imp)
    df_xgb = df_imp[df_imp['Modelo'] == 'XGBoost'].sort_values('Importancia', ascending=False)
    nomi = {
        'diff_rank_points': 'Differenza Punti ATP',
        'diff_rank':        'Ranking ATP',
        'diff_h2h':         'Storico (H2H)',
        'diff_age':         'Età',
        'diff_skill':       'Efficacia Superficie',
        'diff_fatigue':     'Fatica',
        'diff_momentum':    'Striscia (Momentum)',
        'diff_ht':          'Altezza',
        'diff_home':        'Localía'
    }
    df_xgb['Nome'] = df_xgb['Variable'].map(nomi)
    fig_imp = px.bar(df_xgb, x='Importancia', y='Nome', orientation='h',
                     text_auto='.1f',
                     title="Impatto di ogni variabile sulla decisione finale (%)",
                     color='Importancia', color_continuous_scale='Viridis')
    fig_imp.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig_imp, use_container_width=True)
    top_var = df_xgb.iloc[0]['Nome']
    st.info(f"💡 **Conclusione:** Il modello conferma che **{top_var}** è il fattore più determinante.")

else:
    st.warning("⚠️ Nessun dato di analisi disponibile. Esegui `comparar_modelos.py` o il training ANN.")