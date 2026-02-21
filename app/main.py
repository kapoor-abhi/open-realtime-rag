from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import get_settings
from app.core.dependencies import init_services, close_services

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    await init_services(settings)
    yield
    await close_services()

app = FastAPI(title="OpenMultiRAG API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}