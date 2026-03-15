@echo off
REM Start all services: YOLO backend, Elysia backend, Frontend
REM Usage: start-all.bat

set YOLO_DIR=C:\Users\user\utils\annotator-yolo-backend
set ANNOTATOR_DIR=C:\Users\user\utils\annotator-tools
set PYTHON=%YOLO_DIR%\.venv\Scripts\python.exe

REM Default environment variables
if "%DB_URL%"=="" set DB_URL=postgres://postgres:postgres@localhost:5432/annotator-tools
set YOLO_HOST=https://localhost:8002
set BACKEND_HOST=http://localhost:8001

REM Generate SSL certs if missing
if not exist "%YOLO_DIR%\certs\key.pem" (
    echo Generating SSL certificates...
    cd /d "%YOLO_DIR%"
    "%PYTHON%" generate_certs.py
)

echo [1/3] Starting YOLO backend on port 8002...
cd /d "%YOLO_DIR%"
start "YOLO Backend" "%PYTHON%" main.py

echo [2/3] Starting Elysia backend on port 8001...
cd /d "%ANNOTATOR_DIR%\backend-elysia"
start "Elysia Backend" bun run src/index.ts

echo [3/3] Starting frontend on port 5173...
cd /d "%ANNOTATOR_DIR%"
start "Frontend" npm run dev

echo.
echo === All services running ===
echo   Frontend:      http://localhost:5173
echo   Elysia API:    http://localhost:8001
echo   YOLO backend:  https://127.0.0.1:8002
echo.
echo Close the opened terminal windows to stop services.
