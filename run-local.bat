@echo off
set "PYTHON=.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo Virtual environment not found: %PYTHON%
    echo Create it first, then run this file again.
    exit /b 1
)

if not exist "certs\cert.pem" (
    if exist "mkcert.exe" (
        echo Generating trusted SSL certificates with mkcert...
        if not exist "certs" mkdir certs
        .\mkcert.exe -install
        .\mkcert.exe -key-file certs\key.pem -cert-file certs\cert.pem localhost 127.0.0.1
    ) else (
        echo Generating self-signed SSL certificates...
        "%PYTHON%" generate_certs.py
    )
)
"%PYTHON%" main.py



