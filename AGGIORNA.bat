@echo off
chcp 65001 >nul
echo.
echo  ============================================
echo   TENNIS PREDICTOR — Aggiornamento Pipeline
echo  ============================================
echo.
echo  Questo script aggiorna i dati e i modelli.
echo  (Per addestrare l'ANN, imposta ESEGUI_ANN=True in aggiorna_tutto.py)
echo.
pause
echo.
call "%~dp0venv\Scripts\activate.bat"
python "%~dp0aggiorna_tutto.py"
echo.
echo  Pipeline completato. Premi un tasto per chiudere.
pause
