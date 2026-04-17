from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from pathlib import Path

from app.routers import health_router, yolo_router, sam3_router, training_router
from app.services.training_service import TrainingService


@asynccontextmanager
async def lifespan(_: FastAPI):
    TrainingService.initialize()
    yield


app = FastAPI(title="YOLO & SAM Inference Backend", lifespan=lifespan)

class PrivateNetworkMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["Access-Control-Allow-Private-Network"] = "true"
        return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(PrivateNetworkMiddleware)

app.include_router(health_router)
app.include_router(yolo_router)
app.include_router(sam3_router)
app.include_router(training_router)

if __name__ == "__main__":
    import uvicorn
    
    BASE_DIR = Path(__file__).resolve().parent
    ssl_keyfile = BASE_DIR / "certs" / "key.pem"
    ssl_certfile = BASE_DIR / "certs" / "cert.pem"
    
    if ssl_keyfile.exists() and ssl_certfile.exists():
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8002,
            ssl_keyfile=str(ssl_keyfile),
            ssl_certfile=str(ssl_certfile),
            reload=True,
            reload_dirs=[str(BASE_DIR / "app"), str(BASE_DIR)],
            reload_includes=["*.py"],
        )
    else:
        print("SSL certificates not found. Run: python generate_certs.py")
        print("Starting without HTTPS...")
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8002,
            reload=True,
            reload_dirs=[str(BASE_DIR / "app"), str(BASE_DIR)],
            reload_includes=["*.py"],
        )
