@echo off
if not exist "certs\cert.pem" (
    echo Generating SSL certificates...
    python generate_certs.py
)
python main.py



