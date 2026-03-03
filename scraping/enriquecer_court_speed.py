import pandas as pd
import sys
import os

# Importa la funzione helper
sys.path.append(os.path.dirname(__file__))
from court_speed_helper import get_court_stats

print("🏟️ Arricchendo i dati con statistiche dei campi (Ace% e Speed)...")

# Lista dei file da enrichire
files_to_enrich = [
    'historialTenis.csv',
    'historial_tenis.csv',
    'historial_tenis_COMPLETO.csv',
    'atp_matches_2026_full.csv',
]

for filename in files_to_enrich:
    try:
        print(f"\n📂 Processando: {filename}")
        
        if not os.path.exists(filename):
            print(f"   ⏭️  File non trovato, skip...")
            continue
        
        # Carica il CSV
        df = pd.read_csv(filename, low_memory=False)
        initial_rows = len(df)
        
        # Verifica se le colonne esistono già
        if 'court_ace_pct' in df.columns and 'court_speed' in df.columns:
            print(f"   ℹ️  Colonne già presenti. Ricalcolo...")
        
        # Aggiungi le colonne
        ace_pcts = []
        speeds = []
        
        for idx, row in df.iterrows():
            tourney_name = row.get('tourney_name', '')
            surface = row.get('surface', 'Hard')
            
            # Gestisci NaN nei nomi dei tornei
            if pd.isna(tourney_name):
                tourney_name = ''
            if pd.isna(surface):
                surface = 'Hard'
            
            # Estrai l'anno dalla data del torneo
            year = 2020  # Default
            tourney_date = row.get('tourney_date', '')
            if tourney_date and str(tourney_date) != 'nan':
                try:
                    # La data è nel formato YYYYMMDD
                    date_str = str(int(tourney_date))
                    if len(date_str) >= 4:
                        year = int(date_str[:4])
                except:
                    pass
            
            # Ottieni le statistiche con l'anno
            ace_pct, speed = get_court_stats(tourney_name, surface, year)
            ace_pcts.append(ace_pct)
            speeds.append(speed)
            
            # Mostra progresso ogni 10000 righe
            if (idx + 1) % 10000 == 0:
                print(f"   ... {idx + 1}/{len(df)} righe processate")
        
        # Aggiungi al DataFrame
        df['court_ace_pct'] = ace_pcts
        df['court_speed'] = speeds
        
        # Salva
        df.to_csv(filename, index=False)
        
        print(f"   ✅ Completato! {len(df)} righe enrichite")
        print(f"   📊 Statistiche:")
        print(f"      - Ace% medio: {df['court_ace_pct'].mean():.2f}%")
        print(f"      - Speed medio: {df['court_speed'].mean():.1f}")
        
    except Exception as e:
        print(f"   ❌ Errore: {e}")
        continue

# Ora enrichisci anche i file nella cartella prediccion
print("\n📁 Enrichendo file nella cartella prediccion...")
prediccion_path = os.path.join(os.path.dirname(__file__), '..', 'prediccion')

if os.path.exists(prediccion_path):
    for filename in files_to_enrich:
        filepath = os.path.join(prediccion_path, filename)
        if os.path.exists(filepath):
            print(f"\n📂 Processando: prediccion/{filename}")
            try:
                df = pd.read_csv(filepath, low_memory=False)
                
                ace_pcts = []
                speeds = []
                
                for idx, row in df.iterrows():
                    tourney_name = row.get('tourney_name', '')
                    surface = row.get('surface', 'Hard')
                    
                    # Gestisci NaN
                    if pd.isna(tourney_name):
                        tourney_name = ''
                    if pd.isna(surface):
                        surface = 'Hard'
                    
                    # Estrai l'anno dalla data del torneo
                    year = 2020  # Default
                    tourney_date = row.get('tourney_date','')
                    if tourney_date and str(tourney_date) != 'nan':
                        try:
                            date_str = str(int(tourney_date))
                            if len(date_str) >= 4:
                                year = int(date_str[:4])
                        except:
                            pass
                    
                    ace_pct, speed = get_court_stats(tourney_name, surface, year)
                    ace_pcts.append(ace_pct)
                    speeds.append(speed)
                    
                    if (idx + 1) % 10000 == 0:
                        print(f"   ... {idx + 1}/{len(df)} righe processate")
                
                df['court_ace_pct'] = ace_pcts
                df['court_speed'] = speeds
                df.to_csv(filepath, index=False)
                
                print(f"   ✅ Completato!")
                
            except Exception as e:
                print(f"   ❌ Errore: {e}")

print("\n✅ ENRICHMENT COMPLETATO!")
