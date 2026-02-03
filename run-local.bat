@echo off
echo YOLO Backend - Local HTTPS Mode
echo.
if not exist "certs\cert.pem" (
    if exist "mkcert.exe" (
        echo Generating trusted SSL certificates with mkcert...
        if not exist "certs" mkdir certs
        .\mkcert.exe -install
        .\mkcert.exe -key-file certs\key.pem -cert-file certs\cert.pem localhost 127.0.0.1
    ) else (
        echo Generating self-signed SSL certificates...
        python generate_certs.py
    )
)
echo Starting server at https://127.0.0.1:8002
python main.py



