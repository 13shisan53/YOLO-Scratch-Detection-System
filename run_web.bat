@echo off
title MetalScan Web Server
echo ============================================
echo  MetalScan industrial inspection system
echo  http://localhost:8000
echo ============================================
start "" cmd /c "timeout /t 8 /nobreak >nul & start http://localhost:8000"
python web_app.py
pause
