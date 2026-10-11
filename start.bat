@echo off
rem Opens the dashboard. Scanning, alerts and the paper test run from the
rem "Options Dashboard Scanner" scheduled task, so no second window is needed.
cd /d "%~dp0"
set UV_LINK_MODE=copy
rem The scanner runs in the cloud: fetch its latest state first.
if exist .cloud.json uv run python -m options_dashboard.cloudstate pull
uv run streamlit run app.py
