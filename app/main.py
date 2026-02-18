from fastapi import FastAPI
from contextlib import asynccontextmanager
from core.logging import setup_logging
from core.tracing import setup_langsmith_tracing  # NEW
from api.routes import ingest, chat, stream, collections, sessions, metrics  # NEW: metrics
from services.vector_store import ensure_collection
import logging

setup_logging()
setup_langsmith_tracing()  # NEW
logger = logging.getLogger("rag_agent")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting RAG Agent v2.2.0...")
    ensure_collection()
    yield
    logger.info("Shutting down RAG Agent.")


app = FastAPI(
    title="RAG AI Chat Agent",
    description="Production RAG: Streaming | Multi-Collection | Reranking | Memory | Observability",
    version="2.2.0",  # FINAL
    lifespan=lifespan,
)

app.include_router(ingest.router, prefix="/api/v1", tags=["Ingest"])
app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])
app.include_router(stream.router, prefix="/api/v1", tags=["Streaming"])
app.include_router(collections.router, prefix="/api/v1", tags=["Collections"])
app.include_router(sessions.router, prefix="/api/v1", tags=["Sessions"])
app.include_router(metrics.router, prefix="/api/v1", tags=["Metrics"])  # NEW


@app.get("/health")
def health():
    return {"status": "ok", "version": "2.2.0"}