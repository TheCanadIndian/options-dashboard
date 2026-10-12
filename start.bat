@echo off
rem Opens the dashboard. Scanning, alerts and the simulated accounts run in the
rem cloud (cloudflare/ and .github/workflows/scan.yml), so the PC can be off.
cd /d "%~dp0"
set UV_LINK_MODE=copy
rem The scanner runs in the cloud: fetch its latest state first.
if exist .cloud.json uv run python -m options_dashboard.cloudstate pull
uv run streamlit run app.py
