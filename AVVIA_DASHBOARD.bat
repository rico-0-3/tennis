@echo off
chcp 65001 >nul
call "%~dp0venv\Scripts\activate.bat"
streamlit run "%~dp00_🏠_Inicio.py"
