"""
normalizza_challenger_sackmann.py
Importa i file Sackmann atp_matches_qual_chall_YYYY.csv (2022-2024),
filtra solo i Challenger (tourney_level='C'), e li salva come
challenger_matches_sackmann.csv compatibile con historialTenis.csv.

Da eseguire UNA SOLA VOLTA (o ogni volta che aggiungi un nuovo file Sackmann).
Output: challenger_matches_sackmann.csv
"""

import pandas as pd
import os
import glob

ANNI          = list(range(2022, 2025))   # 2022, 2023, 2024
OUTPUT_FILE   = "challenger_matches_sackmann.csv"
PATTERN       = "atp_matches_qual_chall_{anno}.csv"

print("📦 Normalizzazione Challenger Sackmann")
print("=" * 50)

frames = []
for anno in ANNI:
    fname = PATTERN.format(anno=anno)
    if not os.path.exists(fname):
        print(f"   ⚠️  {fname} non trovato — saltato")
        continue

    df = pd.read_csv(fname, low_memory=False)
    n_orig = len(df)

    # Filtra solo Challenger (tourney_level == 'C')
    df = df[df['tourney_level'] == 'C'].copy()
    print(f"   ✅ {fname}: {n_orig} righe totali → {len(df)} Challenger (livello C)")

    frames.append(df)

if not frames:
    print("❌ Nessun file trovato. Verifica che i CSV Sackmann siano in questa cartella.")
    exit()

df_out = pd.concat(frames, ignore_index=True)

# Assicura che tourney_date sia intero YYYYMMDD
df_out['tourney_date'] = pd.to_numeric(df_out['tourney_date'], errors='coerce').fillna(0).astype(int)

# Ordina per data
df_out = df_out.sort_values('tourney_date').reset_index(drop=True)

df_out.to_csv(OUTPUT_FILE, index=False)

print(f"\n💾 Salvato: {OUTPUT_FILE}")
print(f"   → {len(df_out)} partite Challenger | "
      f"anni: {sorted(df_out['tourney_date'].astype(str).str[:4].unique())}")
print(f"   → Tornei unici: {df_out['tourney_id'].nunique()}")
