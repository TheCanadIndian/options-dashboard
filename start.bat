@echo off
rem Opens the dashboard. Scanning, alerts and the paper test run from the
rem "Options Dashboard Scanner" scheduled task, so no second window is needed.
cd /d "%~dp0"
set UV_LINK_MODE=copy
uv run streamlit run app.py
