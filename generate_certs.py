import subprocess
import shutil
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CERTS_DIR = BASE_DIR / "certs"
KEY_PATH = CERTS_DIR / "key.pem"
CERT_PATH = CERTS_DIR / "cert.pem"


def generate_with_openssl():
    openssl = shutil.which("openssl")
    if not openssl:
        return False

    CERTS_DIR.mkdir(exist_ok=True)

    cmd = [
        openssl, "req", "-x509", "-newkey", "rsa:2048",
        "-keyout", str(KEY_PATH),
        "-out", str(CERT_PATH),
        "-days", "365", "-nodes",
        "-subj", "/CN=localhost",
        "-addext", "subjectAltName=DNS:localhost,IP:127.0.0.1",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def generate_with_cryptography():
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from datetime import datetime, timedelta
        import ipaddress
    except ImportError:
        return False

    CERTS_DIR.mkdir(exist_ok=True)

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
    ])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.utcnow())
        .not_valid_after(datetime.utcnow() + timedelta(days=365))
        .add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName("localhost"),
                x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
            ]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )

    with open(KEY_PATH, "wb") as f:
        f.write(key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        ))
    with open(CERT_PATH, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))

    return True


if __name__ == "__main__":
    if KEY_PATH.exists() and CERT_PATH.exists():
        print(f"Certificates already exist in {CERTS_DIR}")
        print("Delete them first if you want to regenerate.")
        sys.exit(0)

    if generate_with_openssl():
        pass
    elif generate_with_cryptography():
        pass
    else:
        print("ERROR: Cannot generate certificates.")
        print("Install openssl or: pip install cryptography")
        sys.exit(1)

    print(f"Generated SSL certificates in {CERTS_DIR}")
    print(f"  - {KEY_PATH}")
    print(f"  - {CERT_PATH}")
    print()
    print("Server will now run on https://127.0.0.1:8002")
    print("Note: Accept the self-signed certificate in your browser first:")
    print("  Open https://127.0.0.1:8002 -> Advanced -> Proceed")
