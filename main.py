import streamlit as st
from PIL import Image

import torch.nn as nn  # <--- AGGIUNGI QUESTO IMPORT

# ─── INCOLLA QUESTA CLASSE QUI IN CIMA AL TUO main.py ──────────────
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
# ───────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ATP Predictor Pro",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="expanded"
)

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;} /* Oculta los 3 puntitos de arriba a la derecha */
            footer {visibility: hidden;} /* Oculta el "Made with Streamlit" de abajo */
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Titolo
st.markdown("<h1 style='text-align: center; color: #1E3A8A;'>🎾 ATP Match Predictor AI</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #64748B;'>Intelligenza Artificiale applicata al Tenis Professionale</h3>", unsafe_allow_html=True)

st.write("---")

# Metriche
col1, col2, col3 = st.columns(3)
col1.metric("Precisione Modello", "73.53%", "+1.2%")
col2.metric("Partite Analizzate", "+30.000", "2015-2026")
col3.metric("Ranking Aggiornato", "2026", "Live")

st.write("---")

st.markdown("""
### 🚀 Cosa può fare questa App?

Questo strumento utilizza algoritmi di **Machine Learning** e **Reti Neurali** per predire il risultato di partite di tennis ATP.

Analizza variabili complesse come:
* 🧠 **Psicologia:** Storico degli scontri diretti (H2H) e momento mentale.
* 🔋 **Fisico:** Fatica accumulata ed età.
* 📊 **Gerarchia:** Differenza reale di punti ATP (non solo ranking).
* 🎯 **Tecnica:** Statistiche di servizio, ace, doppi falli e resa al servizio.
* 🏟️ **Contesto:** Superficie, livello torneo, turno e localía.

### 👈 Usa il menu a sinistra per navigare
* **🏆 Classifica:** Classifica mondiale e profili giocatori aggiornati.
* **🔮 Predittore:** Simula qualsiasi partita (es: Sinner vs Alcaraz).
* **📊 Analisi:** Confronto modelli e importanza delle variabili.
""")

st.info("💡 Suggerimento: Il modello **Rete Neurale (ANN)** tiene conto del peso temporale — le partite recenti contano di più!")

st.markdown("---")

st.markdown("""
    <div style='text-align: center; color: #666; font-size: 12px;'>
        Database, file e algoritmi di tenis di
        <a href='http://www.tennisabstract.com/' target='_blank'>Jeff Sackmann / Tennis Abstract</a>
        sotto licenza
        <a href='https://creativecommons.org/licenses/by-nc-sa/4.0/' target='_blank'>CC BY-NC-SA 4.0</a>.<br>
        Basato su lavoro disponibile su <a href='https://github.com/JeffSackmann' target='_blank'>github.com/JeffSackmann</a>.
    </div>
    """, unsafe_allow_html=True)