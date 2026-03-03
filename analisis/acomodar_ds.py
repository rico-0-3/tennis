import pandas as pd
import joblib
import numpy as np
import os

ruta_script = os.path.dirname(os.path.abspath(__file__))
ruta_raiz = os.path.dirname(ruta_script)
    
ruta_raiz = os.path.join(ruta_raiz, "scraping") 

def get_path(archivo):
    return os.path.join(ruta_raiz, archivo)

df = pd.read_csv(get_path("historial_tenis_COMPLETO.csv"))

archivo_destino = "historialTenis.csv"

# ==============================================================================
# 1. NORMALIZACIÓN DE RONDAS (R16, QF, SF, F...)
# ==============================================================================
print("   -> Estandarizando nombres de rondas...")

# Quitamos espacios en blanco al principio y final (ej: "Final " -> "Final")
df['round'] = df['round'].astype(str).str.strip()

# Diccionario de Traducción
mapa_rondas = {
    # Cuadro Principal
    'Round of 128': 'R128',
    'Round of 64':  'R64',
    'Round of 32':  'R32',
    'Round of 16':  'R16',
    'Quarterfinals': 'QF',
    'Semifinals':    'SF',
    'Semifinal':     'SF',
    'Finals':        'F',
    'The Final':     'F',
    'Final':         'F',
    'Winner':        'W',
    'Round Robin':   'RR',
    
    # Clasificación (Qualy) - Las abreviamos como Q1, Q2, Q3
    '1st Round Qualifying': 'Q1',
    '2nd Round Qualifying': 'Q2',
    '3rd Round Qualifying': 'Q3',
    'Qualifying Round':     'Q1' 
}

# Aplicamos el cambio. Si no encuentra la ronda en el mapa, deja la original.
df['round'] = df['round'].replace(mapa_rondas)

# ==============================================================================
# 2. ARREGLAR FECHAS (2026 -> Fecha Real)
# ==============================================================================
print("   -> Reparando fechas de 2026...")

meses_torneos = {
    'australian': '0115', # Enero 15
    'dallas': '0205',
    'rotterdam': '0205',
    'buenos-aires': '0205',
    'indian': '0310',     # Marzo 10
    'miami': '0325',      # Marzo 25
    'monte': '0415',      # Montecarlo Abril
    'madrid': '0501',     # Mayo 1
    'rome': '0515',       # Mayo 15
    'garros': '0528',     # Roland Garros
    'wimbledon': '0701',  # Julio
    'canada': '0807',     # Agosto
    'cincinnati': '0815', 
    'us-open': '0828',    # US Open
    'shanghai': '1005',
    'paris': '1030',
    'finals': '1115'
}

def corregir_fecha(row):
    fecha_actual = row['tourney_date']
    if pd.notna(fecha_actual) and float(fecha_actual) > 19900000:
        return int(fecha_actual)
    
    # Intentar recuperar del ID
    try:
        id_parts = str(row['tourney_id']).split('-')
        year = id_parts[0]
        if year.isdigit() and len(year) == 4:
            resto_id = str(row['tourney_id']).lower()
            mes_dia = "0101" # Default
            for key, val in meses_torneos.items():
                if key in resto_id:
                    mes_dia = val
                    break
            return int(year + mes_dia)
    except: pass
    return 0

df['tourney_date'] = df.apply(corregir_fecha, axis=1)

# ==============================================================================
# 3. LIMPIEZA DE CEROS EN BIO (Edad, Altura, País)
# ==============================================================================
print("   -> Limpiando ceros en datos biográficos...")
cols_bio = ['winner_age', 'winner_ht', 'winner_ioc', 'loser_age', 'loser_ht', 'loser_ioc']

for col in cols_bio:
    if col in df.columns:
        # Reemplazamos 0 y "0" por vacío (NaN) para que no ensucien
        df[col] = df[col].replace([0, 0.0, '0'], np.nan)

# ==============================================================================
# 4. ORDENAMIENTO FINAL Y GUARDADO
# ==============================================================================
# Mapa numérico para ordenar las filas (R128 va antes que F)
orden_ronda = {
    'Q1':1, 'Q2':2, 'Q3':3, 
    'R128':10, 'R64':20, 'R32':30, 'R16':40, 'QF':50, 'SF':60, 'F':70, 'W':80, 
    'RR': 5
}
df['orden_temp'] = df['round'].map(orden_ronda).fillna(0)

# Ordenamos: 1. Fecha, 2. Torneo, 3. Ronda
df = df.sort_values(by=['tourney_date', 'tourney_id', 'orden_temp'])
df = df.drop(columns=['orden_temp']) # Borramos la columna auxiliar

df.to_csv(archivo_destino, index=False)

print("\n✅ ¡LISTO! CSV Estandarizado.")
print(f"   📂 Archivo generado: {archivo_destino}")
print("   👉 Ahora las rondas son: R128, R64, R16, QF, SF, F")