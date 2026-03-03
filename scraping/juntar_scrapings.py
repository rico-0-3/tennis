import pandas as pd
import os

print("🔗 UNIENDO ARCHIVOS DE SCRAPING (2025 y 2026)...")

# ⚠️ Asegúrate de que estos nombres coincidan exactamente con los tuyos
archivo_2025 = "atp_matches_2025.csv" 
archivo_2026 = "atp_matches_2026_corregido.csv" # O el nombre que le hayas puesto
archivo_salida = "atp_matches_2025_2026_unidos.csv"

archivos_a_unir = []

# Cargar 2025
if os.path.exists(archivo_2025):
    df_25 = pd.read_csv(archivo_2025)
    archivos_a_unir.append(df_25)
    print(f"✅ {archivo_2025} cargado ({len(df_25)} partidos).")
else:
    print(f"⚠️ No se encontró {archivo_2025}")

# Cargar 2026
if os.path.exists(archivo_2026):
    df_26 = pd.read_csv(archivo_2026)
    archivos_a_unir.append(df_26)
    print(f"✅ {archivo_2026} cargado ({len(df_26)} partidos).")
else:
    print(f"⚠️ No se encontró {archivo_2026}")

# Unir y guardar
if len(archivos_a_unir) > 0:
    df_final = pd.concat(archivos_a_unir, ignore_index=True)
    df_final.to_csv(archivo_salida, index=False)
    print("\n" + "="*40)
    print(f"🎉 ¡ARCHIVOS UNIDOS CON ÉXITO!")
    print(f"📊 Total de partidos: {len(df_final)}")
    print(f"📁 Guardado como: {archivo_salida}")
    print("="*40)
else:
    print("❌ No se encontró ningún archivo para unir.")