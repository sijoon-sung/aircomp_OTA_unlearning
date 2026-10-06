@echo off
chcp 65001 > nul
cd /d "%~dp0"
python check.py
if errorlevel 1 goto end
python run.py %*
:end
pause
