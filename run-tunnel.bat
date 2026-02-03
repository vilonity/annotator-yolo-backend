@echo off
echo YOLO Backend - Tunnel Mode
echo.
if "%~1"=="" (
    echo Usage: run-tunnel.bat ^<token^>
    echo.
    echo Example: run-tunnel.bat eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
    echo.
    echo Get your token from the annotator-tools web app settings.
    exit /b 1
)
set TOKEN=%~1
set TUNNEL_URL=wss://pepeshit.ru/api/yolo-tunnel
echo Connecting to tunnel: %TUNNEL_URL%
python main.py --tunnel %TUNNEL_URL% --token %TOKEN%
