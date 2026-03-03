import pandas as pd
import joblib
import numpy as np

print("👤 GENERANDO PERFILES (V5.0 - SOLUCIÓN TOTAL)...")

try:
    # 1. Cargar CSV
    df = pd.read_csv("historialTenis.csv")
    
    # --- A. LIMPIEZA Y FORMATO ---
    df['tourney_id'] = df['tourney_id'].astype(str)
    
    # Asegurar que columnas numéricas no tengan texto basura
    cols_check = ['winner_age', 'winner_ht', 'loser_age', 'loser_ht', 'winner_rank', 'loser_rank']
    for col in cols_check:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')


    # --- C. MAPEO DE RONDAS (CORREGIDO SEGÚN TU CSV) ---
    # Usamos .strip() más abajo para quitar espacios extra (ej: "Final ")
    round_map = {
        # Clasificación (Valen menos)
        'Q1': 1,
        'Q2': 2,
        'Q3': 3,
        
        # Cuadro Principal (Valen más)
        'Round of 128': 10,
        'Round of 64': 20,
        'Round of 32': 30,
        'Round of 16': 40,
        'Quarterfinals': 50,
        'Semifinals': 60,
        'Final': 70, # La más importante
        'The Final': 70, # Por si acaso
        
        # Mantenemos las abreviaturas viejas por compatibilidad con años anteriores
        'R128': 10, 'R64': 20, 'R32': 30, 'R16': 40, 'QF': 50, 'SF': 60, 'F': 70, 'W': 80
    }

    # Limpiamos espacios en blanco del nombre de la ronda (ej: "Final " -> "Final")
    df['round_clean'] = df['round'].astype(str).str.strip()
    
    # Mapeamos
    df['round_val'] = df['round_clean'].map(round_map).fillna(0)

    # --- D. ORDENAMIENTO MAESTRO (CRONOLOGÍA ABSOLUTA) ---
    print("⏳ Reparando la línea temporal del circuito...")

    def crear_orden_absoluto(row):
        fecha = pd.to_numeric(row['tourney_date'], errors='coerce')
        
        # 1. Si es un partido viejo real (2000-2024), usamos su fecha multiplicada
        if pd.notna(fecha) and fecha > 19900000 and fecha not in [20250101, 20260101]:
            return fecha * 100 
            
        # 2. Si es de los nuevos scrapeados (2025 o 2026), armamos un código matemático infalible
        try:
            tid = str(row['tourney_id']) # Ej: "2026-rio-de-janeiro-12"
            partes = tid.split('-')
            anio = int(partes[0]) # Saca el 2026
            indice = int(partes[-1]) if partes[-1].isdigit() else 0 # Saca el 12
            
            # Crea un número cronológico perfecto: ej. 2026001200
            return (anio * 1000000) + (indice * 100)
        except:
            return 2026999900 # Si falla, lo manda al fondo

    # Aplicamos la magia
    df['orden_absoluto'] = df.apply(crear_orden_absoluto, axis=1)
    
    # Ordenamos por nuestra columna invencible, y luego por la ronda
    df = df.sort_values(by=['orden_absoluto', 'round_val'])

    # --- E. PROCESAMIENTO CON MEMORIA ---
    perfiles = {}
    racha_tracker = {}
    
    # "Cache" para recordar datos si vienen vacíos
    bio_cache = {} 

    total_partidos = {}

    for index, row in df.iterrows():
        w = row['winner_name']
        l = row['loser_name']
        score = row['score']
        torneo = row['tourney_name']
        ronda = row['round']
        fecha = row['tourney_date']

        total_partidos[w] = total_partidos.get(w, 0) + 1
        total_partidos[l] = total_partidos.get(l, 0) + 1
        
        # --- 1. GESTIÓN DEL HISTORIAL (RACHA) ---
        rw = racha_tracker.get(w, [])
        rl = racha_tracker.get(l, [])
        
        # A. Datos para el GANADOR
        match_w = {
            'resultado': 'W',      # Won
            'rival': l,            # El rival fue el perdedor
            'score': score,        # El score (siempre está visto desde el ganador)
            'torneo': torneo,
            'ronda': ronda
        }
        
        # B. Datos para el PERDEDOR
        match_l = {
            'resultado': 'L',      # Lost
            'rival': w,            # El rival fue el ganador
            'score': score,        # El score
            'torneo': torneo,
            'ronda': ronda
        }
        
        rw.append(match_w)
        rl.append(match_l)
        
        # Mantenemos solo los últimos 5
        if len(rw) > 5: rw.pop(0)
        if len(rl) > 5: rl.pop(0)
        
        racha_tracker[w] = rw
        racha_tracker[l] = rl
        
        # --- 2. DATOS BIO (Igual que antes) ---
        # (Resumido para no ocupar espacio, la lógica es la misma de la V4.0)
        mem_w = bio_cache.get(w, {'age': 25, 'ht': 185, 'ioc': 'UNK', 'points': 0, 'rank': 500})
        if pd.notna(row.get('winner_age')) and row['winner_age'] > 10: mem_w['age'] = row['winner_age']
        if pd.notna(row.get('winner_ht')) and row['winner_ht'] > 100: mem_w['ht'] = row['winner_ht']
        if pd.notna(row.get('winner_ioc')) and str(row['winner_ioc']) != '0': mem_w['ioc'] = row['winner_ioc']
        if pd.notna(row.get('winner_rank')): mem_w['rank'] = row['winner_rank']
        if pd.notna(row.get('winner_rank_points')): mem_w['points'] = row['winner_rank_points']
        bio_cache[w] = mem_w; perfiles[w] = mem_w.copy()

        mem_l = bio_cache.get(l, {'age': 25, 'ht': 185, 'ioc': 'UNK', 'points': 0, 'rank': 500})
        if pd.notna(row.get('loser_age')) and row['loser_age'] > 10: mem_l['age'] = row['loser_age']
        if pd.notna(row.get('loser_ht')) and row['loser_ht'] > 100: mem_l['ht'] = row['loser_ht']
        if pd.notna(row.get('loser_ioc')) and str(row['loser_ioc']) != '0': mem_l['ioc'] = row['loser_ioc']
        if pd.notna(row.get('loser_rank')): mem_l['rank'] = row['loser_rank']
        if pd.notna(row.get('loser_rank_points')): mem_l['points'] = row['loser_rank_points']
        bio_cache[l] = mem_l; perfiles[l] = mem_l.copy()

    # --- F. INYECTAR ESTADÍSTICAS AVANZADAS, RANKING 2026 Y GUARDADO ---
    print("   💉 Inyectando estadísticas avanzadas y Ranking actualizado...")
    
    # Cargar Stats
    try:
        df_stats_adv = pd.read_csv("estadisticas_jugadores_avanzadas.csv")
        stats_dict_adv = df_stats_adv.set_index('player').to_dict(orient='index')
    except:
        print("   ⚠️ No se encontró 'estadisticas_jugadores_avanzadas.csv'.")
        stats_dict_adv = {}

    # Cargar Ranking Nuevo (El que acabas de scrapear)
    try:
        df_ranking = pd.read_csv("ranking_2026.csv")
        # 🧹 BARRER DUPLICADOS
        df_ranking = df_ranking.drop_duplicates(subset=['player'], keep='first')
        
        # 🧠 EL TRUCO MAGISTRAL: Extraer el nombre completo desde la URL
        # URL ej: https://www.atptour.com/en/players/jannik-sinner/s0ag/overview
        def extraer_nombre_real(url):
            try:
                slug = str(url).split('/')[5] # Corta la URL y agarra "jannik-sinner"
                return slug.replace('-', ' ').title() # Lo convierte a "Jannik Sinner"
            except:
                return ""
                
        # Creamos una nueva columna con el nombre perfecto
        df_ranking['player_real'] = df_ranking['url_perfil'].apply(extraer_nombre_real)
        
        # Ahora usamos 'player_real' como llave del diccionario en lugar del abreviado
        ranking_dict_fresco = df_ranking.set_index('player_real').to_dict(orient='index')
        print("   ✅ 'ranking_2026.csv' cargado OK con nombres corregidos.")
    except Exception as e:
        print(f"   ⚠️ ERROR REAL con ranking: {e}")
        ranking_dict_fresco = {}

    # --- TEST DE SUPERVIVENCIA ---
    cerundolo = df[(df['winner_name'] == 'Francisco Cerundolo') | (df['loser_name'] == 'Francisco Cerundolo')]
    print("\n🔍 TEST ANTES DEL BUCLE:")
    print(cerundolo[['tourney_id', 'round']].tail(8))
    print("-----------------------\n")

    for jugador, datos in perfiles.items():
        
        # 1. Actualizar Ranking y Puntos con la info fresca de hoy (Pisa la memoria vieja)
        if jugador in ranking_dict_fresco:
            perfiles[jugador]['rank'] = ranking_dict_fresco[jugador].get('rank', 500)
            perfiles[jugador]['points'] = ranking_dict_fresco[jugador].get('points', 0)

        # 2. Racha y Momentum
        historial = racha_tracker.get(jugador, [])
        victorias = sum(1 for x in historial if x['resultado'] == 'W')
        perfiles[jugador]['momentum'] = victorias / len(historial) if historial else 0.5
        perfiles[jugador]['last_5'] = historial

        # 3. Stats Avanzadas
        partidos_jugados = total_partidos.get(jugador, 1) # Evitar dividir por cero
        
        if jugador in stats_dict_adv:
            datos_extra = stats_dict_adv[jugador]
            perfiles[jugador]['serve_win'] = datos_extra.get('serve_win_pct', 65.0)
            perfiles[jugador]['bp_saved'] = datos_extra.get('bp_saved_pct', 60.0)
            perfiles[jugador]['service_hold'] = datos_extra.get('service_hold_pct', 75.0)
            
            # Ajuste de Aces y Dobles Faltas dividiendo por cantidad de partidos
            perfiles[jugador]['aces'] = datos_extra.get('aces_avg', 0.0) / partidos_jugados
            perfiles[jugador]['df'] = datos_extra.get('df_avg', 0.0) / partidos_jugados
        else:
            perfiles[jugador]['serve_win'] = 65.0 
            perfiles[jugador]['bp_saved'] = 60.0 
            perfiles[jugador]['service_hold'] = 75.0
            perfiles[jugador]['aces'] = 0.0
            perfiles[jugador]['df'] = 0.0

    # Debug Final
    print("\n🔎 VERIFICACIÓN FINAL:")
    if 'Jannik Sinner' in perfiles:
        p = perfiles['Jannik Sinner']
        print(f"   Jannik Sinner -> Ranking: {p.get('rank')} | Puntos: {p.get('points')}")

    joblib.dump(perfiles, 'perfiles_jugadores.pkl')
    print("\n✅ Archivo de perfiles actualizado con Ranking Fresco. ¡Listo para la App!")

except Exception as e:
    print(f"❌ Error: {e}")