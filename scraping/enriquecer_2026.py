import pandas as pd
import joblib
import numpy as np

# --- CONFIGURACIÓN ---
ARCHIVO_NUEVO = "atp_matches_2026_indetectable.csv" # Tu CSV flaco (recién bajado)
ARCHIVO_PERFILES = "perfiles_jugadores.pkl"         # Tu diccionario de datos (bajado antes)
ARCHIVO_SALIDA = "atp_matches_2026_full.csv"        # El CSV gordo final

print("💉 INICIANDO ENRIQUECIMIENTO DE DATOS...")

# 1. CARGAR DATOS
try:
    df = pd.read_csv(ARCHIVO_NUEVO)
    perfiles = joblib.load(ARCHIVO_PERFILES)
    print(f"📂 Cargados {len(df)} partidos nuevos.")
    print(f"📂 Cargados {len(perfiles)} perfiles de jugadores.")
except Exception as e:
    print(f"❌ Error cargando archivos: {e}")
    print("Asegúrate de tener 'atp_matches_2026_indetectable.csv' y 'perfiles_jugadores.pkl'")
    exit()

# 2. DICCIONARIO DE SUPERFICIES (Para no dejarlo vacío)
# Agrega aquí los torneos que vayas bajando
mapa_superficies = {
    'Australian Open': 'Hard',
    'Roland Garros': 'Clay',
    'Wimbledon': 'Grass',
    'US Open': 'Hard',
    'Indian Wells': 'Hard',
    'Miami': 'Hard',
    'Monte Carlo': 'Clay',
    'Madrid': 'Clay',
    'Rome': 'Clay',
    'Cincinnati': 'Hard',
    'Canada': 'Hard',
    'Shanghai': 'Hard',
    'Paris': 'Hard',
    'Rotterdam': 'Hard', 
    'perth-sydney': 'Hard',
    'brisbane': 'Hard',
    'hong-kong': 'Hard',
    'adelaide': 'Hard',
    'auckland': 'Hard',
    'montpellier': 'Clay',
    'dallas': 'Hard',
    'buenos-aires': 'Clay',
    'doha': 'Hard',
    'rio-de-janeiro': 'Clay',
    'delray-beach': 'Hard'
}

# 3. LISTAS PARA LOS DATOS NUEVOS
w_ht, w_age, w_rank, w_hand, w_ioc = [], [], [], [], []
l_ht, l_age, l_rank, l_hand, l_ioc = [], [], [], [], []
surfaces = []

print("🔄 Cruzando datos...")

for index, row in df.iterrows():
    w_name = row['winner_name']
    l_name = row['loser_name']
    torneo = row['tourney_name']
    
    # --- A. SUPERFICIE ---
    # Buscamos el torneo en el mapa, si no está, ponemos 'Hard' por defecto
    surf = mapa_superficies.get(torneo, 'Hard')
    # Intento simple de detectar Clay/Grass si está en el nombre
    if 'Clay' in str(torneo): surf = 'Clay'
    if 'Grass' in str(torneo): surf = 'Grass'
    surfaces.append(surf)

    # --- B. DATOS DEL GANADOR ---
    if w_name in perfiles:
        p = perfiles[w_name]
        w_rank.append(p.get('rank', 100)) # Si no tiene rank, ponemos 100
        w_ht.append(p.get('ht', 185))     # Altura promedio 185
        w_age.append(p.get('age', 25) + 1.5) # Sumamos 1.5 años porque es 2026
        w_ioc.append(p.get('ioc', 'UNK'))
        w_hand.append('R') # Asumimos diestro si falta (dato menor)
    else:
        # JUGADOR NUEVO (ROOKIE)
        w_rank.append(150)
        w_ht.append(185)
        w_age.append(22)
        w_ioc.append('UNK')
        w_hand.append('R')

    # --- C. DATOS DEL PERDEDOR ---
    if l_name in perfiles:
        p = perfiles[l_name]
        l_rank.append(p.get('rank', 100))
        l_ht.append(p.get('ht', 185))
        l_age.append(p.get('age', 25) + 1.5)
        l_ioc.append(p.get('ioc', 'UNK'))
        l_hand.append('R')
    else:
        l_rank.append(150)
        l_ht.append(185)
        l_age.append(22)
        l_ioc.append('UNK')
        l_hand.append('R')

# 4. AGREGAR COLUMNAS AL DATAFRAME
df['surface'] = surfaces
df['winner_ht'] = w_ht
df['winner_age'] = w_age
df['winner_rank'] = w_rank
df['winner_hand'] = w_hand
df['winner_ioc'] = w_ioc

df['loser_ht'] = l_ht
df['loser_age'] = l_age
df['loser_rank'] = l_rank
df['loser_hand'] = l_hand
df['loser_ioc'] = l_ioc

# Columnas extra para que coincida con el histórico (relleno con ceros)
cols_extra = ['match_num', 'best_of', 'winner_rank_points', 'loser_rank_points']
for col in cols_extra:
    df[col] = 0

# 5. GUARDAR
df.to_csv(ARCHIVO_SALIDA, index=False)
print("\n" + "="*50)
print(f"✅ ¡ENRIQUECIMIENTO COMPLETADO!")
print(f"📄 Archivo guardado: {ARCHIVO_SALIDA}")
print(f"📊 Ahora tienes Ranking, Altura y Edad estimados para 2026.")
print("="*50)