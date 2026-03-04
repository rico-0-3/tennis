import pandas as pd
import os

# --- CONFIGURACIÓN ---
ARCHIVO_HISTORICO  = "historial_tenis.csv"          # Storico originale 2000-2024
ARCHIVO_2025       = "atp_matches_2025.csv"          # Partite 2025
ARCHIVO_2026       = "atp_matches_2026_corregido.csv"  # Partite 2026 corrette (output FASE3)
ARCHIVO_SALIDA     = "historialTenis.csv"

print("🧬 INICIANDO FUSIÓN FINAL...")

# 1. CARGAR ARCHIVOS
if not os.path.exists(ARCHIVO_HISTORICO):
    print(f"❌ Error: No encuentro '{ARCHIVO_HISTORICO}'")
    exit()

# Costruisce df_new da 2025 + 2026 disponibili
partes_nuevas = []
for f, label in [(ARCHIVO_2025, "2025"), (ARCHIVO_2026, "2026")]:
    if os.path.exists(f):
        partes_nuevas.append(pd.read_csv(f))
        print(f"   📂 {label}: {f} caricato ({len(partes_nuevas[-1])} partite)")
    else:
        print(f"   ⚠️  {label}: '{f}' non trovato — saltato")

if not partes_nuevas:
    print("❌ Nessun file 2025/2026 trovato. Impossibile continuare.")
    exit()

try:
    df_hist = pd.read_csv(ARCHIVO_HISTORICO)
    df_new = pd.concat(partes_nuevas, ignore_index=True)
    print(f"   📂 2025+2026 uniti: {len(df_new)} partite totali")
    
    print(f"📂 Histórico: {len(df_hist)} partidos | Columnas: {len(df_hist.columns)}")
    print(f"📂 Nuevo:     {len(df_new)} partidos  | Columnas: {len(df_new.columns)}")

    # 2. NORMALIZAR COLUMNAS (El paso clave)
    # Hacemos que el nuevo tenga EXACTAMENTE las mismas columnas que el histórico
    columnas_hist = df_hist.columns.tolist()
    
    # Verificamos si hay columnas con nombres distintos y tratamos de arreglarlas
    # (A veces el scrape trae 'winner' y el historico 'winner_name')
    mapeo = {
        'winner': 'winner_name',
        'loser': 'loser_name',
        'tourney_id': 'tourney_id', # Asegurar que coincidan
        'surface': 'surface'
    }
    df_new.rename(columns=mapeo, inplace=True)

    # Creamos un DataFrame nuevo solo con las columnas del histórico
    df_new_aligned = pd.DataFrame(columns=columnas_hist)
    
    # Copiamos los datos que SÍ tenemos
    for col in df_new.columns:
        if col in columnas_hist:
            df_new_aligned[col] = df_new[col]
        else:
            print(f"   ⚠️ La columna '{col}' del nuevo archivo se ignorará (no existe en histórico).")
    
    # Rellenamos los datos faltantes (Stats de partido que no scrapeamos)
    # Ej: w_ace, w_df, minutes, etc.
    df_new_aligned.fillna(0, inplace=True)
    df_new_aligned = df_new_aligned.infer_objects(copy=False)
    
    # 3. UNIR (CONCATENAR)
    print("🔄 Uniendo archivos...")
    df_total = pd.concat([df_hist, df_new_aligned], ignore_index=True)
    
    # 4. LIMPIEZA FINALE — le date sono già corrette dallo scraper (YYYYMMDD)
    print("⏳ Pulizia e ordinamento...")
    df_total['tourney_date'] = pd.to_numeric(df_total['tourney_date'], errors='coerce').fillna(0).astype(int)

    # Avvisa se ci sono date nulle o sospette (aiuta il debug)
    date_zero = (df_total['tourney_date'] == 0).sum()
    if date_zero > 0:
        print(f"   ⚠️  {date_zero} partite con tourney_date=0 (dati incompleti in scraping)")

    df_total.sort_values(by=['tourney_date', 'match_num'], inplace=True)

    # 5. GUARDAR
    df_total.to_csv(ARCHIVO_SALIDA, index=False)
    
    print("\n" + "="*50)
    print(f"🎉 ¡FUSIÓN EXITOSA!")
    print(f"📊 Total partite: {len(df_total)}")
    print(f"   Storico: {len(df_hist)} | 2025+2026: {len(df_new)}")
    print(f"💾 Guardado en: {ARCHIVO_SALIDA}")
    print("="*50)

except Exception as e:
    print(f"❌ Error durante la fusión: {e}")
    # Diagnóstico de columnas
    print("\n🔍 DIAGNÓSTICO DE COLUMNAS:")
    print(f"Histórico (Primeras 5): {list(df_hist.columns)[:5]}")
    print(f"Nuevo (Primeras 5):     {list(df_new.columns)[:5]}")