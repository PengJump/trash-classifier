@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================
echo   Trash classification demo - starting...
echo ============================================
echo.

where java >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Java not found. Please install JDK 17 or newer.
    echo         Download: https://adoptium.net/
    echo.
    pause
    exit /b 1
)

if not exist "trashnet.onnx" (
    echo [ERROR] trashnet.onnx not found next to this script.
    echo         Please keep this .bat in the project root folder.
    echo.
    pause
    exit /b 1
)

echo Starting backend (Spring Boot + ONNX Runtime)...
echo Web UI will open at: http://localhost:8080
echo Close this console window to stop the demo.
echo.

start "TrashDemo" cmd /k "java -jar backend\trash-classifier.jar --trash.model.path=trashnet.onnx"

timeout /t 12 /nobreak >nul
start http://localhost:8080

echo.
echo Done. If the page did not open, visit http://localhost:8080
echo.
pause
