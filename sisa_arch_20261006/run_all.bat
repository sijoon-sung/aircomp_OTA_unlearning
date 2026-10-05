@echo off
chcp 65001 > nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
set PY=python
where python > nul 2> nul
if errorlevel 1 set PY=py -3
%PY% -c "import torch, torchvision, numpy" 2> nul
if errorlevel 1 (
  echo [!] torch / torchvision / numpy not found. See README_KO.md section 2 to install them.
  pause
  exit /b 1
)
%PY% run_all.py %*
echo.
echo Done. Open runs\^<latest folder^>\SUMMARY_KO.md
pause
