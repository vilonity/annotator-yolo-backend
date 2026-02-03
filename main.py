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

async def login_and_get_token(api_url: str, username: str, password: str) -> str:
    import aiohttp
    
    login_url = f"{api_url}/api/auth/login"
    async with aiohttp.ClientSession() as session:
        async with session.post(login_url, json={"username": username, "password": password}) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise Exception(f"Login failed: {error_text}")
            data = await resp.json()
            return data["access_token"]


async def run_tunnel_mode(tunnel_url: str, token: str):
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


async def run_tunnel_with_login(server_url: str, username: str, password: str):
    print(f"Logging in as {username}...")
    token = await login_and_get_token(server_url, username, password)
    print("Login successful!")
    
    tunnel_url = server_url.replace("https://", "wss://").replace("http://", "ws://") + "/api/yolo-tunnel"
    await run_tunnel_mode(tunnel_url, token)


if __name__ == "__main__":
    import argparse
    import asyncio
    import getpass

    parser = argparse.ArgumentParser(description="YOLO & SAM Inference Backend")
    parser.add_argument("--tunnel", action="store_true", help="Run in tunnel mode")
    parser.add_argument("--server", type=str, default="https://pepeshit.ru", help="Server URL (default: https://pepeshit.ru)")
    parser.add_argument("--username", "-u", type=str, help="Username for authentication")
    parser.add_argument("--password", "-p", type=str, help="Password for authentication")
    parser.add_argument("--token", type=str, help="Use existing JWT token instead of login")
    args = parser.parse_args()

    if args.tunnel:
        if args.token:
            tunnel_url = args.server.replace("https://", "wss://").replace("http://", "ws://") + "/api/yolo-tunnel"
            asyncio.run(run_tunnel_mode(tunnel_url, args.token))
        else:
            username = args.username or input("Username: ")
            password = args.password or getpass.getpass("Password: ")
            asyncio.run(run_tunnel_with_login(args.server, username, password))
    else:
        run_local_mode()
