from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.core.logging import setup_logging
from app.api.routes import ingest, chat
from app.services.vector_store import ensure_collection
import logging

setup_logging()
logger = logging.getLogger("rag_agent")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting RAG Agent...")
    ensure_collection()
    yield
    logger.info("Shutting down RAG Agent.")


app = FastAPI(
    title="RAG AI Chat Agent",
    description="Production-grade RAG pipeline with LangChain, Qdrant, and OpenRouter",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(ingest.router, prefix="/api/v1", tags=["Ingest"])
app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])


@app.get("/health")
def health():
    return {"status": "ok"}