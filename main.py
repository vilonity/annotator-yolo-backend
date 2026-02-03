from fastapi import FastAPI, Request
from fastapi.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware
from pathlib import Path

from app.routers import health_router, yolo_router, sam3_router

app = FastAPI(title="YOLO & SAM Inference Backend")

class CORSPrivateNetworkMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        origin = request.headers.get("origin", "*")
        
        if request.method == "OPTIONS":
            response = Response(status_code=204)
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "*"
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers["Access-Control-Allow-Private-Network"] = "true"
            response.headers["Access-Control-Max-Age"] = "86400"
            return response
        
        response = await call_next(request)
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Private-Network"] = "true"
        return response

app.add_middleware(CORSPrivateNetworkMiddleware)

app.include_router(health_router)
app.include_router(yolo_router)
app.include_router(sam3_router)

async def run_tunnel_mode(tunnel_url: str, token: str):
    import asyncio
    from tunnel_client import TunnelClient, create_local_request_handler

    handler = await create_local_request_handler(app)
    client = TunnelClient(tunnel_url, token, handler)

    print(f"Starting tunnel mode, connecting to {tunnel_url}")
    await client.connect()


def run_local_mode():
    import uvicorn

    BASE_DIR = Path(__file__).resolve().parent
    ssl_keyfile = BASE_DIR / "certs" / "key.pem"
    ssl_certfile = BASE_DIR / "certs" / "cert.pem"

    if ssl_keyfile.exists() and ssl_certfile.exists():
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8002,
            ssl_keyfile=str(ssl_keyfile),
            ssl_certfile=str(ssl_certfile),
        )
    else:
        print("SSL certificates not found. Run: python generate_certs.py")
        print("Starting without HTTPS...")
        uvicorn.run(app, host="0.0.0.0", port=8002)


if __name__ == "__main__":
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description="YOLO & SAM Inference Backend")
    parser.add_argument("--tunnel", type=str, help="Tunnel WebSocket URL (e.g., wss://pepeshit.ru/api/yolo-tunnel)")
    parser.add_argument("--token", type=str, help="Authentication token for tunnel")
    args = parser.parse_args()

    if args.tunnel and args.token:
        asyncio.run(run_tunnel_mode(args.tunnel, args.token))
    elif args.tunnel or args.token:
        print("Error: Both --tunnel and --token are required for tunnel mode")
        exit(1)
    else:
        run_local_mode()
