@echo off
REM Windows launcher: creates a venv on first run, installs deps, starts the app.
cd /d "%~dp0"
if not exist .venv (
    python -m venv .venv || exit /b 1
    REM truststore = use the Windows certificate store (works behind antivirus/proxy SSL inspection)
    .venv\Scripts\python -m pip install --use-feature=truststore -r requirements.txt || exit /b 1
)
.venv\Scripts\python -m streamlit run app.py
