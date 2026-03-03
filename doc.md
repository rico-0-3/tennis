# 1. Crear un entorno virtual (una caja aislada para este proyecto)

python -m venv venv

# 2. Activar el entorno (Verás que aparece un (venv) verde o blanco al inicio de la linea)

# En Windows:

.\venv\Scripts\activate

# (Si te da error de permisos en Windows, avísame, pero suele funcionar).

# 3. Instalar las librerías de scraping

pip install requests beautifulsoup4 pandas

requests: Es el "navegador" de Python. Pide la página web.

beautifulsoup4: Es el "traductor". Toma el código HTML feo y te deja buscar cosas fácil (ej: "dame el precio").

pandas: Ya lo conoces, es tu Excel supervitaminado.

¡Excelente iniciativa! Documentar es la mejor forma de consolidar lo que has aprendido y asegurarte de que el proyecto sea mantenible en el futuro.

He preparado una documentación técnica completa, organizada por el **flujo de trabajo** (desde que se obtienen los datos hasta que se muestran en la pantalla).

---

# 📘 Documentación Técnica: ATP Tennis Predictor

## 1. Flujo General del Proyecto (Pipeline)

El proyecto sigue un flujo de datos lineal, conocido en Ciencia de Datos como **ETL** (Extract, Transform, Load) + **Modelado**:

1. **Scraping:** Obtención de datos crudos de la web (Partidos y Ranking).
2. **Preprocesamiento (ETL):** Limpieza, corrección de fechas, normalización de nombres y fusión de datos históricos con los nuevos.
3. **Entrenamiento:** Generación de variables matemáticas (Feature Engineering) y creación de los modelos de IA.
4. **Backend (Perfiles):** Generación de la "memoria" actual de los jugadores para la App.
5. **Frontend (App):** Visualización y predicción en tiempo real.

---

## 2. Fase de Extracción (Scraping)

Aquí usamos **Selenium** y **BeautifulSoup** para "leer" páginas web como si fuéramos humanos.

### 📄 `scraper_2026_final.py`

**Propósito:** Descargar los resultados de los partidos de 2025/2026 desde la web de la ATP.

- **Librerías Clave:**
- `undetected_chromedriver`: Una versión modificada de Chrome para evitar que las webs detecten que somos un robot (evita bloqueos de Cloudflare).
- `BeautifulSoup`: Sirve para analizar el código HTML y extraer texto limpio (nombres, scores).

- **Lógica:**

1. Lee una lista de URLs de torneos.
2. Abre un navegador real.
3. Hace scroll hasta el final de la página para cargar todos los partidos.
4. Extrae ganador, perdedor, resultado y ronda.
5. Guarda todo en `atp_matches_2025_indetectable.csv`.

### 📄 `scraper_ranking.py`

**Propósito:** Obtener el Top 500 del ranking ATP actual.

- **Función Clave:**
- `slug`: Extrae el identificador único del jugador desde la URL (ej: de `.../players/carlos-alcaraz/...` saca `carlos-alcaraz`) para normalizar nombres.

- **Salida:** `ranking_actual_2026.csv`.

---

## 3. Fase de Limpieza y Transformación

Los datos crudos suelen tener errores (fechas vacías, nombres distintos, etc.). Aquí los arreglamos.

### 📄 `corregir_superficie_ranking.py`

**Propósito:** Rellenar huecos de información en los datos nuevos.

- **¿Qué hace?**

1. **Deducción de Superficie:** La ATP a veces no dice si es "Arcilla" o "Dura". El script busca palabras clave en el nombre del torneo (ej: si dice "Roland Garros" -> pone "Clay").
2. **Inyección de Ranking:** Cruza los partidos con el archivo de ranking descargado para asignar el puesto actual a cada jugador en los partidos de 2026.

### 📄 `acomodar_ds.py`

**Propósito:** Estandarización final antes de entrenar.

- **Funciones Críticas:**
- `mapa_rondas`: Un diccionario que traduce nombres largos ("Quarterfinals") a códigos cortos ("QF"). Esto es vital para que la IA entienda el orden de los partidos.
- `corregir_fecha`: Transforma fechas simuladas o incompletas (ej: "2026") en fechas numéricas reales (ej: `20260115` para 15 de Enero), basándose en el calendario real de torneos.
- **Limpieza de Ceros:** Convierte `0` en `NaN` (Not a Number) para que Pandas sepa que son datos faltantes y no valores reales.

### 📄 `fusionar_historico_final.py`

**Propósito:** Unir el pasado (2000-2024) con el presente (2025-2026).

- **Lógica:**
- Toma el archivo histórico gigante.
- Toma el archivo nuevo limpio.
- Alinea las columnas (se asegura de que tengan el mismo nombre).
- Usa `pd.concat` para pegarlos uno debajo del otro, creando `historial_tenis_COMPLETO.csv`.

---

## 4. Fase de Entrenamiento (Machine Learning)

Aquí es donde la matemática ocurre. Transformamos "nombres de jugadores" en "números" que la IA puede entender.

### 📄 `comparar_modelos.py`

**Propósito:** Laboratorio de pruebas. Entrena varios modelos a la vez para ver cuál es mejor.

- **Feature Engineering (Ingeniería de Variables):**
- Crea variables nuevas que no existen en el Excel:
- `H2H` (Historial entre ellos): Cuenta cuántas veces se ganaron antes.
- `Fatiga`: Suma los minutos jugados en el torneo actual.
- `Momentum`: Calcula el % de victorias en los últimos 5 partidos (ventana deslizante).
- `Skill`: % de victorias en la superficie específica (ej: Nadal en Polvo).

- **Creación del Dataset de Entrenamiento:**
- La IA necesita aprender de la **diferencia**. No le sirve saber "Alcaraz Rank 1, Novak Rank 3".
- Calculamos: `diff_rank = Rank_Perdedor - Rank_Ganador`.

- **Comparación:**
- Prueba `LogisticRegression`, `RandomForest` y `XGBoost`.
- Guarda los resultados en `resultados_comparacion.csv` y la importancia de variables en `importancia_real.csv`.

### 📄 `predict_xgboost.py` (y `predict_LR.py`)

**Propósito:** Entrenar el modelo definitivo y guardarlo para la App.

- **Librerías:**
- `xgboost`: Algoritmo basado en "Gradient Boosting" (árboles de decisión que corrigen sus errores secuencialmente). Es el estándar de oro en competiciones de datos.
- `sklearn.preprocessing.StandardScaler`: **Normalización**. Convierte los datos a una escala común (media 0, desviación 1). Esto es vital porque el Ranking (1-500) y la Altura (180) tienen escalas muy distintas.

- **Salida:** Genera archivos `.pkl` (pickle) que son el "cerebro congelado" de la IA, listos para cargarse en la App.

---

## 5. Fase de Backend para la App

### 📄 `generar_perfiles.py`

**Propósito:** Crear una "foto instantánea" del estado actual de cada jugador.

- **Diferencia con los scripts de entrenamiento:**
- Los scripts de entrenamiento miran el pasado para aprender.
- Este script recorre toda la historia para calcular **cómo llega el jugador HOY**.

- **Memoria Bio (`bio_cache`):**
- Soluciona el problema de datos faltantes. Si en el último partido de 2026 no figura la edad, el script "recuerda" la edad del partido anterior y la rellena.

- **Salida:** `perfiles_jugadores.pkl`. Un diccionario gigante con la info de cada tenista (Racha, Rank, Edad, H2H, etc.).

---

## 6. Fase de Frontend (Visualización)

### 📄 `laboratorio.py`

**Propósito:** Página educativa dentro de la App Streamlit.

- **Librerías:**
- `streamlit`: Convierte scripts de Python en páginas web interactivas.
- `plotly.express`: Crea gráficos interactivos (barras, tortas).

- **Lógica:**
- Lee los CSVs generados por `comparar_modelos.py`.
- Muestra gráficamente qué modelo ganó y qué variables son las más importantes (Ranking, H2H, etc.).

---

## 📚 Glosario de Conceptos y Librerías

Para que tengas a mano si te olvidas qué hace cada cosa:

- **Pandas (`pd`):** El Excel de Python. Maneja tablas de datos (`DataFrames`).
- `pd.read_csv()`: Abre archivos.
- `df.apply()`: Aplica una función a cada fila.
- `pd.to_numeric(errors='coerce')`: Intenta convertir texto a número; si falla, pone `NaN` (vacío).

- **Numpy (`np`):** Matemáticas rápidas.
- `np.nan`: Representación técnica de "dato faltante".

- **Joblib:** La "caja fuerte". Sirve para guardar variables complejas (como un modelo entrenado o un diccionario) en un archivo `.pkl` y recuperarlas después.
- **Scikit-Learn (`sklearn`):** La caja de herramientas de IA clásica.
- `train_test_split`: Divide los datos en "Estudio" (80%) y "Examen" (20%) para verificar que la IA no memorice.
- `GridSearchCV`: Prueba muchas combinaciones de configuraciones (hiperparámetros) automáticamente para encontrar la mejor.

- **XGBoost:** Un algoritmo muy potente tipo "ensamble". Crea cientos de árboles de decisión simples, donde cada uno intenta corregir los errores del anterior.

---

### 💡 Resumen del Flujo de Ejecución

Si quisieras actualizar todo el proyecto desde cero con datos nuevos, el orden de ejecución sería:

1. `scraper_2026_final.py` (Bajar partidos nuevos).
2. `scraper_ranking.py` (Bajar ranking nuevo).
3. `corregir_superficie_ranking.py` (Arreglar datos nuevos).
4. `fusionar_historico_final.py` (Unir con histórico).
5. `acomodar_ds.py` (Limpieza final y estandarización).
6. `generar_perfiles.py` (Crear base de datos para la App).
7. `comparar_modelos.py` (Verificar métricas).
8. `predict_xgboost.py` (Entrenar cerebro final).
9. **Ejecutar la App** (`streamlit run app.py`).
