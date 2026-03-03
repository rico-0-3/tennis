import pandas as pd


# 1. CARGA
print("🎾 Cargando el Dataset Maestro...")
try:
    df = pd.read_csv("historial_tenis.csv")
    # Convertimos la fecha a formato datetime (útil para gráficos)
    df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d', errors='coerce')
    print(f"✅ Datos cargados: {len(df)} partidos.")
except FileNotFoundError:
    print("❌ No encontré el CSV. Asegúrate de haber corrido 'descargar_tenis.py' primero.")
    exit()

# 2. TOP 10 GANADORES
print("\n--- 🏆 TOP 10 JUGADORES CON MÁS VICTORIAS (2000-2024) ---")
top_winners = df['winner_name'].value_counts().head(10)
print(top_winners)

# 3. ANÁLISIS DE SUPERFICIE
print("\n--- 🌍 DISTRIBUCIÓN POR SUPERFICIE ---")
superficies = df['surface'].value_counts()
print(superficies)

# 4. DURACIÓN DE PARTIDOS
# Limpiamos nulos en 'minutes'
df_duracion = df.dropna(subset=['minutes'])
promedio = df_duracion['minutes'].mean()
mas_largo = df_duracion.sort_values('minutes', ascending=False).iloc[0]

print(f"\n--- ⏱️ TIEMPOS DE JUEGO ---")
print(f"Duración promedio: {int(promedio)} minutos")
print(f"El partido más largo registrado duró: {int(mas_largo['minutes'])} minutos")
print(f"Fue: {mas_largo['winner_name']} vs {mas_largo['loser_name']} ({mas_largo['tourney_name']} {mas_largo['tourney_date'].year})")