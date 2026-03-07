import pandas as pd
import os
import re

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
    # Usa 0 per colonne numeriche e stringa vuota per colonne stringa
    for col in df_new_aligned.columns:
        if df_new_aligned[col].dtype == object:
            df_new_aligned[col] = df_new_aligned[col].fillna('')
        else:
            df_new_aligned[col] = df_new_aligned[col].fillna(0)
    df_new_aligned = df_new_aligned.infer_objects()
    
    # 3. UNIR (CONCATENAR)
    print("🔄 Uniendo archivos...")
    df_total = pd.concat([df_hist, df_new_aligned], ignore_index=True)
    
    # 4. LIMPIEZA FINALE — le date sono già corrette dallo scraper (YYYYMMDD)
    print("⏳ Pulizia e ordinamento...")
    df_total['tourney_date'] = pd.to_numeric(df_total['tourney_date'], errors='coerce').fillna(0).astype(int)

    # ── Correggi tourney_date=0 usando le date storiche ──────────────────────
    date_zero = (df_total['tourney_date'] == 0).sum()
    if date_zero > 0:
        print(f"   ⚠️  {date_zero} partite con tourney_date=0 — provo a correggere...")

        def _normalize_name(name):
            """Normalizza nome torneo: 'Australian Open' → 'australian-open'."""
            if not isinstance(name, str):
                return ""
            s = name.strip().lower()
            # Rimuovi suffissi numerici tipo "Adelaide 1" → "Adelaide"
            s = re.sub(r'\s+\d+$', '', s)
            # Mapping nomi speciali storico → slug scraping
            special = {
                'canada masters': 'montreal',
                'cincinnati masters': 'cincinnati',
                'shanghai masters': 'shanghai',
                'nitto atp finals': 'nitto-atp-finals',
                'united cup': 'perth-sydney',
                'indian wells masters': 'indian-wells',
                'miami masters': 'miami',
                'monte carlo masters': 'monte-carlo',
                'madrid masters': 'madrid',
                'rome masters': 'rome',
                'paris masters': 'paris',
                "queen's club": 'london',
                'tour finals': 'nitto-atp-finals',
            }
            if s in special:
                s = special[s]
            # Converti spazi e caratteri speciali in trattini
            s = re.sub(r'[^a-z0-9]+', '-', s).strip('-')
            return s

        # Costruisci mappa: nome_normalizzato → data più recente (YYYYMMDD)
        df_con_data = df_total[df_total['tourney_date'] > 0].copy()
        date_map = {}
        for _, row in df_con_data[['tourney_name', 'tourney_date']].drop_duplicates().iterrows():
            norm = _normalize_name(row['tourney_name'])
            if norm and int(row['tourney_date']) > date_map.get(norm, 0):
                date_map[norm] = int(row['tourney_date'])

        fixed = 0
        for idx in df_total[df_total['tourney_date'] == 0].index:
            tname = _normalize_name(df_total.at[idx, 'tourney_name'])
            tid = str(df_total.at[idx, 'tourney_id'])

            # Estrai anno dal tourney_id (es. "2025-brisbane-1" → 2025)
            m = re.match(r'(\d{4})', tid)
            anno_match = int(m.group(1)) if m else 0

            if tname in date_map and anno_match > 0:
                data_ref = date_map[tname]
                # Sostituisci anno nella data storica con anno della partita
                anno_ref = data_ref // 10000
                data_nuova = data_ref - anno_ref * 10000 + anno_match * 10000
                df_total.at[idx, 'tourney_date'] = data_nuova
                fixed += 1

        date_zero_after = (df_total['tourney_date'] == 0).sum()
        print(f"   ✅ Corrette {fixed} partite | Rimaste con date=0: {date_zero_after}")

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