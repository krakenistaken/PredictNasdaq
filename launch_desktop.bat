@echo off
title PredictNasdaq Desktop Application Launcher
echo ========================================================
echo   Starting PredictNasdaq Desktop Application...
echo ========================================================
"%~dp0\.venv\Scripts\python.exe" "%~dp0\gui_app.py"
if %errorlevel% neq 0 (
    echo.
    echo An error occurred. Checking system Python environment...
    python gui_app.py
)
pause
