# 🎾 ATP Tennis Predictor AI

Una aplicación de Machine Learning construida con Python y Streamlit para predecir ganadores de partidos de tenis ATP.

## Modelos
Compara dos algoritmos en tiempo real:
* **XGBoost:** (Modelo Campeón - 72% Precisión)
* **Regresión Logística:** (Modelo Base - 69% Precisión)

## Estructura 
* `scraper_2026_final.py`: Obtención de datos automatizada con Selenium.
* `generar_perfiles.py`: Creación de perfiles de jugadores y cálculo de Momentum/Fatiga.
* `predict_xgboost.py`: Entrenamiento del modelo.
* `app_main.py`: Interfaz gráfica web.
