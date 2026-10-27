@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
chcp 65001 >nul 2>&1
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
title MaizeSight - UAV Maize Yield Intelligence

echo =========================================================
echo   MaizeSight - UAV Maize Yield Intelligence Web Server
echo =========================================================
echo.
echo Starting MaizeSight Web Server...

if exist "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" (
    "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" app.py
    goto FINISH
)

if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" app.py
    goto FINISH
)

if exist ".venv\Scripts\python.exe" (
    .venv\Scripts\python.exe app.py
    goto FINISH
)

where py >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    py app.py
    goto FINISH
)

where python >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    python app.py
    goto FINISH
)

echo.
echo [ERROR] No working Python installation was found.
echo Please ensure Python is installed and added to your PATH.
echo.

:FINISH
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Server stopped with error code %ERRORLEVEL%.
    pause
)


