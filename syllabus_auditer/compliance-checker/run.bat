@echo off
setlocal

echo.
echo ==========================================
echo   AI Compliance Checker — Setup ^& Launch
echo ==========================================
echo.

:: ── Use existing ai-lab environment ──────────────────────────────────────────
set AI_LAB=C:\Users\parminder_local\ai-lab

if not exist "%AI_LAB%\Scripts\activate.bat" (
    echo [ERROR] Could not find environment at %AI_LAB%\Scripts\activate.bat
    echo         Check the path is correct.
    pause
    exit /b 1
)

call "%AI_LAB%\Scripts\activate.bat"

for /f "tokens=*" %%v in ('python --version 2^>^&1') do echo [OK] Using %%v from %AI_LAB%

:: ── Install / upgrade packages ───────────────────────────────────────────────
echo.
echo [SETUP] Installing packages from requirements.txt...
"%AI_LAB%\Scripts\pip3.exe" install -q -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Package installation failed.
    pause
    exit /b 1
)
echo [OK] Packages ready.

:: ── Create .env from example if missing ──────────────────────────────────────
if not exist ".env" (
    echo.
    echo [SETUP] No .env file found. Creating one from .env.example...
    copy ".env.example" ".env" >nul
    echo [ACTION NEEDED] Open .env in Notepad and add your API key, then re-run this file.
    echo.
    notepad .env
    echo.
    echo Press any key once you have saved your API key...
    pause >nul
)

:: ── Launch the app ────────────────────────────────────────────────────────────
echo.
echo [OK] Starting AI Compliance Checker...
echo      Opening browser at http://localhost:8501
echo      Press Ctrl+C in this window to stop the app.
echo.
python -m streamlit run app.py

endlocal
