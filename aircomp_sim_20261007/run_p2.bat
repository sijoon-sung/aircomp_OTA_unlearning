@echo off
chcp 65001 > nul
cd /d "%~dp0"
echo [1/2] 무선 식 검증
python check_cdma.py
if errorlevel 1 goto end
echo [2/2] P2 실행 (seed 5, 수십 분~2시간)
python run.py --only p2 %*
:end
pause
