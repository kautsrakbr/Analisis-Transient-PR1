@echo off
cd /d "%~dp0"

set PYTHON_EXE="C:\My Workspaces\venv.Python\.venv\Scripts\python.exe"

if not exist %PYTHON_EXE% (
    echo [ERROR] Python venv tidak ditemukan di %PYTHON_EXE%
    echo Coba pakai python biasa dari PATH sebagai gantinya...
    set PYTHON_EXE=python
)

echo Menjalankan Gaver-Stehfest PTA solver...
echo.
%PYTHON_EXE% "%~dp0pta_solver.py"

echo.
echo Selesai. Tekan tombol apa saja untuk menutup jendela ini.
pause >nul
