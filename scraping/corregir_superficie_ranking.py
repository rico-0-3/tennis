import pandas as pd
import numpy as np

print("🛠️ INICIANDO CORRECCIÓN DE SUPERFICIES Y RANKINGS...")

# --- ARCHIVOS DE ENTRADA ---
# Tu archivo con los partidos de 2025 y 2026 (puede ser el raw o el master)
ARCHIVO_PARTIDOS = "atp_matches_2026_full.csv" 
# Tu archivo con el ranking actual que scrapeamos antes
ARCHIVO_RANKING = "ranking_2026.csv"
# Archivo de salida limpio
ARCHIVO_SALIDA = "atp_matches_2026_corregido.csv" 

try:
    # 1. CARGAR DATOS
    df = pd.read_csv(ARCHIVO_PARTIDOS)
    df_rank = pd.read_csv(ARCHIVO_RANKING)
    print(f"✅ Partidos cargados: {len(df)}")
    print(f"✅ Ranking cargado: {len(df_rank)}")

    # ---------------------------------------------------------
    # PASO 1: ARREGLAR SUPERFICIES (CLAY, GRASS, HARD) 🏟️
    # ---------------------------------------------------------
    print("🌍 Corrigiendo superficies...")

    def detectar_superficie(nombre_torneo):
        nombre = str(nombre_torneo).lower()
        
        # --- PALABRAS CLAVE PARA ARCILLA (CLAY) ---
        clay_keywords = [
            'roland garros', 'madrid', 'rome', 'roma', 'monte carlo', 'barcelona', 
            'rio', 'buenos aires', 'cordoba', 'santiago', 'estoril', 'munich', 
            'geneva', 'lyon', 'hamburg', 'bastad', 'gstaad', 'umag', 'kitzbuhel'
        ]
        # --- PALABRAS CLAVE PARA PASTO (GRASS) ---
        grass_keywords = [
            'wimbledon', 'queen', 'halle', 'mallorca', 'eastbourne', 
            'stuttgart', 'hertogenbosch', 'newport'
        ]
        # (Todo lo demás será Hard por descarte: AO, US Open, Masters grandes, etc.)

        if any(k in nombre for k in clay_keywords):
            return 'Clay'
        elif any(k in nombre for k in grass_keywords):
            return 'Grass'
        else:
            return 'Hard' # Por defecto (Cementos, Indoor, Carpet)

    # Aplicamos la función fila por fila
    df['surface'] = df['tourney_name'].apply(detectar_superficie)
    
    # Reporte rápido
    conteo = df['surface'].value_counts()
    print(f"   📊 Superficies detectadas: Clay={conteo.get('Clay',0)}, Grass={conteo.get('Grass',0)}, Hard={conteo.get('Hard',0)}")

    # ---------------------------------------------------------
    # PASO 2: INYECTAR RANKING ACTUAL A 2026 🏆
    # ---------------------------------------------------------
    print("💉 Inyectando Ranking Actual a los partidos de 2026...")

    # Creamos un diccionario rápido: {'Carlos Alcaraz': 2, 'Jannik Sinner': 1}
    # Usamos 'player_slug' porque el scraper de ranking lo guardó así
    ranking_dict = df_rank.set_index('player')['rank'].to_dict()

    # Función para buscar ranking
    def get_current_rank(nombre, ranking_actual):
        # Si el jugador está en el top 500 actual, devolvemos su rank
        if nombre in ranking_dict:
            return ranking_dict[nombre]
        # Si tiene ranking viejo (del archivo), lo mantenemos. Si no, ponemos 500
        if pd.notna(ranking_actual) and ranking_actual > 0:
            return ranking_actual
        return 500 # Valor por defecto para desconocidos

    # Filtramos solo los partidos de 2026 (o todos si prefieres usar el rank actual para todo)
    # Aquí lo aplicamos a TODO el archivo 2025/2026 para que la IA sepa el nivel "actual" del jugador
    # Si prefieres solo 2026, cambia a: df[df['tourney_date'].astype(str).str.startswith('2026')]
    
    # Actualizamos Winner Rank
    df['winner_rank'] = df.apply(lambda row: get_current_rank(row['winner_name'], row.get('winner_rank', 0)), axis=1)
    
    # Actualizamos Loser Rank
    df['loser_rank'] = df.apply(lambda row: get_current_rank(row['loser_name'], row.get('loser_rank', 0)), axis=1)

    # ---------------------------------------------------------
    # PASO 3: LIMPIEZA FINAL Y GUARDADO 💾
    # ---------------------------------------------------------
    
    # Rellenamos columnas faltantes con ceros para que no falle la fusión final
    cols_zero = ['winner_rank_points', 'loser_rank_points', 'match_num']
    for col in cols_zero:
        if col not in df.columns:
            df[col] = 0
            
    df.to_csv(ARCHIVO_SALIDA, index=False)
    
    print("\n" + "="*50)
    print("🎉 ¡CORRECCIÓN COMPLETADA!")
    print(f"📂 Archivo listo: {ARCHIVO_SALIDA}")
    print("   -> Superficies corregidas.")
    print("   -> Ranking 2026 actualizado.")
    print("="*50)

except Exception as e:
    print(f"❌ Error: {e}")
    print("Asegúrate de tener los archivos 'atp_matches_2025_2026_raw.csv' y 'ranking_actual_2026.csv'")