from fastapi import FastAPI
from contextlib import asynccontextmanager
from core.logging import setup_logging
from api.routes import ingest, chat, stream, collections  # NEW: added stream, collections
from services.vector_store import ensure_collection
import logging

setup_logging()
logger = logging.getLogger("rag_agent")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting RAG Agent...")
    ensure_collection()  # Create default collection
    yield
    logger.info("Shutting down RAG Agent.")


app = FastAPI(
    title="RAG AI Chat Agent",
    description="Production-grade RAG pipeline with LangChain, Qdrant, and OpenRouter",
    version="2.0.0",  # NEW version
    lifespan=lifespan,
)

app.include_router(ingest.router, prefix="/api/v1", tags=["Ingest"])
app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])
app.include_router(stream.router, prefix="/api/v1", tags=["Streaming"])  # NEW
app.include_router(collections.router, prefix="/api/v1", tags=["Collections"])  # NEW


@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0"}