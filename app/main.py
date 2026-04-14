from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router
from app.services.vector_db import vector_db
from app.services.embeddings import embedding_service
from app.models import HealthResponse
from app.config import get_settings

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await vector_db.ensure_collection()
    _ = embedding_service.dimension
    yield
    await vector_db.close()


app = FastAPI(
    title="Actian VectorAI DB - RAG API",
    description="FastAPI application with RAG capabilities using Actian VectorAI DB",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1", tags=["RAG"])


@app.get("/health", response_model=HealthResponse)
async def health_check():
    try:
        info = await vector_db.health_check()
        collection_exists = await vector_db.collection_exists()
        return HealthResponse(
            status="healthy",
            vector_db_version=info.get("version"),
            embedding_model=settings.embedding_model,
            collection_exists=collection_exists
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            embedding_model=settings.embedding_model,
            collection_exists=False
        )


@app.get("/")
async def root():
    return {
        "name": "Actian VectorAI DB - RAG API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=True
    )
