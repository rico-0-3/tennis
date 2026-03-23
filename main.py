import streamlit as st
from PIL import Image
import os

import torch.nn as nn

# Classe ANN per Scommesse Speciali
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

st.set_page_config(
    page_title="ATP Predictor Pro",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Carica CSS custom
css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Header principale con design moderno
st.markdown("""
<div class="tennis-header">
    <h1>🎾 ATP Match Predictor AI</h1>
    <h3>Intelligenza Artificiale applicata al Tennis Professionale</h3>
</div>
""", unsafe_allow_html=True)

# Metriche in stile card moderne
st.markdown('<div class="tennis-card">', unsafe_allow_html=True)
metric_col1, metric_col2, metric_col3 = st.columns(3)
with metric_col1:
    st.metric("Precisione Modello", "73.53%", "+1.2%", help="Accuratezza del modello ANN su dati di test")
with metric_col2:
    st.metric("Partite Analizzate", "+30.000", "2015-2026", help="Database storico completo")
with metric_col3:
    st.metric("Ranking ATP", "2026", "Live", help="Classifica aggiornata automaticamente")

st.markdown('</div>', unsafe_allow_html=True)

# Sezione Cosa può fare questa App
st.markdown("""
<div class="tennis-card">
    <h3>Cosa può fare questa App?</h3>
    <p>Questo strumento utilizza algoritmi di <strong>Machine Learning</strong> e <strong>Reti Neurali</strong> 
    per predire il risultato di partite di tennis ATP con precisione professionale.</p>
    
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-top: 1rem;">
        <div style="background: linear-gradient(135deg, #e3f2fd, #bbdefb); padding: 1rem; border-radius: 12px; text-align: center;">
            <div style="font-size: 2rem;">🧠</div>
            <strong>Psicologia</strong><br>
            <small>H2H, momentum mentale</small>
        </div>
        <div style="background: linear-gradient(135deg, #e8f5e9, #c8e6c9); padding: 1rem; border-radius: 12px; text-align: center;">
            <div style="font-size: 2rem;">🔋</div>
            <strong>Fisico</strong><br>
            <small>Fatica, eta, condizione</small>
        </div>
        <div style="background: linear-gradient(135deg, #fff3e0, #ffe0b2); padding: 1rem; border-radius: 12px; text-align: center;">
            <div style="font-size: 2rem;">📊</div>
            <strong>Gerarchia</strong><br>
            <small>Punti ATP reali</small>
        </div>
        <div style="background: linear-gradient(135deg, #fce4ec, #f8bbd9); padding: 1rem; border-radius: 12px; text-align: center;">
            <div style="font-size: 2rem;">🎯</div>
            <strong>Tecnica</strong><br>
            <small>Servizio, break point</small>
        </div>
        <div style="background: linear-gradient(135deg, #f3e5f5, #e1bee7); padding: 1rem; border-radius: 12px; text-align: center;">
            <div style="font-size: 2rem;">🏟️</div>
            <strong>Contesto</strong><br>
            <small>Superficie, torneo</small>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Menu navigazione
st.markdown("""
<div class="tennis-card">
    <h3>Navigazione Rapida</h3>
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem;">
        <div style="background: #1B5E20; color: white; padding: 1rem; border-radius: 12px; text-align: center;">
            <div style="font-size: 2rem;">🏆</div>
            <strong>Classifica ATP</strong><br>
            <small>Ranking e profili giocatori</small>
        </div>
        <div style="background: #1565C0; color: white; padding: 1rem; border-radius: 12px; text-align: center;">
            <div style="font-size: 2rem;">🔮</div>
            <strong>Predittore Match</strong><br>
            <small>Simula qualsiasi partita</small>
        </div>
        <div style="background: #D84315; color: white; padding: 1rem; border-radius: 12px; text-align: center;">
            <div style="font-size: 2rem;">📊</div>
            <strong>Analisi Modelli</strong><br>
            <small>Confronto e metriche</small>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Info box
st.info("**Suggerimento:** Il modello **Rete Neurale (ANN)** tiene conto del peso temporale — le partite recenti contano di piu!")

# Footer
st.markdown("""
<div class="tennis-footer">
    <p>Database, file e algoritmi di tennis di
    <a href='http://www.tennisabstract.com/' target='_blank'>Jeff Sackmann / Tennis Abstract</a>
    sotto licenza
    <a href='https://creativecommons.org/licenses/by-nc-sa/4.0/' target='_blank'>CC BY-NC-SA 4.0</a>.<br>
    Basato su lavoro disponibile su <a href='https://github.com/JeffSackmann' target='_blank'>github.com/JeffSackmann</a>.</p>
</div>
""", unsafe_allow_html=True)
