@echo off
chcp 65001 >nul
call "%~dp0venv\Scripts\activate.bat"
streamlit run "%~dp0run.py"
