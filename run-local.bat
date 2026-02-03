@echo off
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
python main.py



