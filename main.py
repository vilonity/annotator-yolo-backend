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

if __name__ == "__main__":
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
