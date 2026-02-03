from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import health_router, yolo_router, sam3_router

app = FastAPI(title="YOLO & SAM Inference Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(yolo_router)
app.include_router(sam3_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
