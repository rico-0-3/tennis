import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

print("🧪 INICIANDO COMPARACIÓN Y ANÁLISIS DE VARIABLES...")

# --- 1. PREPARACIÓN DE DATOS (Igual que siempre) ---
try:
    df = pd.read_csv("historial_tenis_COMPLETO.csv")
    df['minutes'] = df['minutes'].fillna(100)
    df['tourney_date'] = pd.to_numeric(df['tourney_date'], errors='coerce')
    df = df.sort_values(by=['tourney_date', 'match_num'])
except:
    print("❌ Error cargando CSV")
    exit()

# --- VARIABLES ---
print("   -> Generando variables...")

# 1. SKILLS (EFECTIVIDAD EN SUPERFICIE)
# Contar victorias
wins = df.groupby(['winner_name', 'surface']).size().reset_index(name='wins')
wins = wins.rename(columns={'winner_name': 'player'}) # Estandarizamos nombre

# Contar derrotas
losses = df.groupby(['loser_name', 'surface']).size().reset_index(name='losses')
losses = losses.rename(columns={'loser_name': 'player'}) # Estandarizamos nombre

# Unir todo en una sola tabla
stats = pd.merge(wins, losses, on=['player', 'surface'], how='outer')

# Rellenar ceros (Si alguien nunca perdió, sale NaN, lo pasamos a 0)
stats['wins'] = stats['wins'].fillna(0)
stats['losses'] = stats['losses'].fillna(0)

# Calcular Win Rate
stats['total'] = stats['wins'] + stats['losses']
stats = stats[stats['total'] >= 5] # Filtro de al menos 5 partidos
stats['win_rate'] = stats['wins'] / stats['total']

# Crear Diccionario
stats_dict = stats.set_index(['player', 'surface'])['win_rate'].to_dict()

def get_skill(p, s): return stats_dict.get((p, s), 0.5)

# 2. LOCALÍA
def detectar_pais(nombre):
    t = str(nombre).upper()
    if 'MADRID' in t or 'BARCELONA' in t: return 'ESP'
    if 'PARIS' in t: return 'FRA'
    if 'US OPEN' in t or 'INDIAN' in t: return 'USA'
    if 'WIMBLEDON' in t: return 'GBR'
    if 'AUSTRALIAN' in t: return 'AUS'
    return 'NEUTRAL'
df['tourney_ioc'] = df['tourney_name'].apply(detectar_pais)

# 3. VARIABLES TEMPORALES (Fatiga, Momentum, H2H)
fatiga_tracker = {}
racha_tracker = {}
h2h_tracker = {}
l_fatiga_w, l_fatiga_l = [], []
l_racha_w, l_racha_l = [], []
l_h2h_w, l_h2h_l = [], []

for index, row in df.iterrows():
    tid, w, l, dur = row['tourney_id'], row['winner_name'], row['loser_name'], row['minutes']
    
    # --- Fatiga ---
    f_w = fatiga_tracker.get((tid, w), 0)
    f_l = fatiga_tracker.get((tid, l), 0)
    l_fatiga_w.append(f_w); l_fatiga_l.append(f_l)
    fatiga_tracker[(tid, w)] = f_w + dur
    fatiga_tracker[(tid, l)] = f_l + dur
    
    # --- Momentum (Racha) ---
    hw = racha_tracker.get(w, []); hl = racha_tracker.get(l, [])
    # Promedio de la racha actual
    mw = sum(hw)/len(hw) if hw else 0.5
    ml = sum(hl)/len(hl) if hl else 0.5
    l_racha_w.append(mw); l_racha_l.append(ml)
    # Actualizar para el futuro (Winner 1, Loser 0)
    hw.append(1); hl.append(0)
    if len(hw)>5: hw.pop(0)
    if len(hl)>5: hl.pop(0)
    racha_tracker[w] = hw; racha_tracker[l] = hl
    
    # --- H2H (Historial) ---
    p1, p2 = sorted([w, l])
    key = (p1, p2)
    record = h2h_tracker.get(key, [0, 0]) # [Victorias_P1, Victorias_P2]
    
    if w == p1:
        # P1 es el ganador actual. Su ventaja PREVIA es (Vic_P1 - Vic_P2)
        l_h2h_w.append(record[0] - record[1])
        l_h2h_l.append(record[1] - record[0])
        record[0] += 1 # Sumamos victoria a P1
    else:
        # P2 (w) es el ganador. Su ventaja PREVIA es (Vic_P2 - Vic_P1)
        l_h2h_w.append(record[1] - record[0])
        l_h2h_l.append(record[0] - record[1])
        record[1] += 1 # Sumamos victoria a P2
        
    h2h_tracker[key] = record

df['winner_fatigue'] = l_fatiga_w; df['loser_fatigue'] = l_fatiga_l
df['winner_momentum'] = l_racha_w; df['loser_momentum'] = l_racha_l
df['winner_h2h'] = l_h2h_w; df['loser_h2h'] = l_h2h_l

# --- DATASET FINAL ---
cols = ['winner_rank', 'loser_rank', 'winner_age', 'loser_age', 'winner_ht', 'loser_ht', 'surface', 'winner_ioc', 'loser_ioc', 'winner_fatigue', 'loser_fatigue', 'winner_momentum', 'loser_momentum', 'winner_h2h', 'loser_h2h']
df = df.dropna(subset=cols)

data_rows = []
df_sample = df.sample(frac=1, random_state=42)

for index, row in df_sample.iterrows():
    surf = row['surface']
    t_ioc = row['tourney_ioc']
    skill_w, skill_l = get_skill(row['winner_name'], surf), get_skill(row['loser_name'], surf)
    h_w = 1 if row['winner_ioc'] == t_ioc else 0
    h_l = 1 if row['loser_ioc'] == t_ioc else 0
    
    pts_w = row['winner_rank_points'] if pd.notna(row['winner_rank_points']) else 0
    pts_l = row['loser_rank_points'] if pd.notna(row['loser_rank_points']) else 0
    
    diffs = {
        'diff_rank': row['loser_rank'] - row['winner_rank'],
        'diff_rank_points': pts_w - pts_l,
        'diff_age': row['winner_age'] - row['loser_age'],
        'diff_ht': row['winner_ht'] - row['loser_ht'],
        'diff_skill': skill_w - skill_l,
        'diff_home': h_w - h_l,
        'diff_fatigue': row['winner_fatigue'] - row['loser_fatigue'],
        'diff_h2h': row['winner_h2h'] - row['loser_h2h'],
        'diff_momentum': row['winner_momentum'] - row['loser_momentum']
    }
    
    d1 = diffs.copy(); d1['target'] = 1
    data_rows.append(d1)
    d0 = {k: -v for k, v in diffs.items()}; d0['target'] = 0
    data_rows.append(d0)

df_train = pd.DataFrame(data_rows)
features = ['diff_rank', 'diff_rank_points', 'diff_age', 'diff_ht', 'diff_skill', 'diff_home', 'diff_fatigue', 'diff_momentum', 'diff_h2h']
X = df_train[features]
y = df_train['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================================================================
# 2. ENTRENAMIENTO Y ANÁLISIS DE IMPORTANCIA 📊
# ==============================================================================
modelos = {
    "Regresión Logística": LogisticRegression(C=0.01, max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10),
    "XGBoost": xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5)
}

resultados_acc = []
resultados_imp = [] # Aquí guardaremos la importancia de cada variable

print(f"\n🥊 Entrenando y analizando variables...")

for nombre, modelo in modelos.items():
    modelo.fit(X_train_scaled, y_train)
    
    # 1. Guardar Accuracy
    y_pred = modelo.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    resultados_acc.append({"Modelo": nombre, "Accuracy": acc})
    print(f"   -> {nombre}: {acc:.2%}")
    
    # 2. Calcular Importancia de Variables
    importancia = []
    
    if nombre == "Regresión Logística":
        # En Logística, usamos el valor absoluto de los coeficientes
        importancia = np.abs(modelo.coef_[0])
    else:
        # En Árboles (RF y XGB), usamos feature_importances_
        importancia = modelo.feature_importances_
    
    # Convertir a porcentaje para que sume 100
    importancia = 100.0 * (importancia / importancia.sum())
    
    # Guardar cada variable
    for i, feature in enumerate(features):
        resultados_imp.append({
            "Modelo": nombre,
            "Variable": feature,
            "Importancia": importancia[i]
        })

# Guardar Archivos
pd.DataFrame(resultados_acc).to_csv("resultados_comparacion.csv", index=False)
pd.DataFrame(resultados_imp).to_csv("importancia_real.csv", index=False) # <--- NUEVO ARCHIVO

print("\n✅ ¡Listo! Se generó 'importancia_real.csv' con los datos reales.")