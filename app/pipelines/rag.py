from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from services.embedder import embed_query
from services.vector_store import search
from services.reranker import rerank_documents  # NEW
from core.config import get_settings
import logging

logger = logging.getLogger("rag_agent")

PROMPT_TEMPLATE = """You are a helpful assistant. Answer the question based ONLY on the context provided.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""


def format_context(docs: list[dict]) -> str:
    return "\n\n---\n\n".join(
        [f"[Source: {d['filename']} | Score: {d.get('rerank_score', d['score']):.3f}]\n{d['text']}" for d in docs]
    )


def run_rag(question: str, collection_name: str = None, use_reranking: bool = True) -> dict:
    settings = get_settings()

    # 1. Embed query
    query_vector = embed_query(question)

    # 2. Retrieve (get more for reranking)
    initial_top_k = settings.top_k * 2 if use_reranking else settings.top_k
    docs = search(query_vector, top_k=initial_top_k, collection_name=collection_name)
    
    if not docs:
        return {
            "answer": "No relevant documents found. Please ingest documents first.",
            "sources": [],
            "collection_name": collection_name or settings.qdrant_collection,
        }

    # 3. Rerank (NEW)
    if use_reranking and len(docs) > settings.rerank_top_k:
        docs = rerank_documents(question, docs)
        logger.info(f"Reranked to top {len(docs)} documents")

    # 4. Build LLM chain
    llm = ChatOpenAI(
        model=settings.llm_model,
        openai_api_key=settings.openrouter_api_key,
        openai_api_base=settings.openrouter_base_url,
        temperature=0.2,
        max_tokens=1024,
    )

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = prompt | llm | StrOutputParser()

    # 5. Generate
    context = format_context(docs)
    answer = chain.invoke({"context": context, "question": question})

    return {
        "answer": answer,
        "sources": [
            {
                "filename": d["filename"],
                "score": d["score"],
                "rerank_score": d.get("rerank_score"),
            }
            for d in docs
        ],
        "collection_name": collection_name or settings.qdrant_collection,
    }


def run_rag_streaming(question: str, collection_name: str = None, use_reranking: bool = True):
    """Streaming version - yields tokens as they're generated."""
    settings = get_settings()

    # Retrieval (same as above)
    query_vector = embed_query(question)
    initial_top_k = settings.top_k * 2 if use_reranking else settings.top_k
    docs = search(query_vector, top_k=initial_top_k, collection_name=collection_name)
    
    if not docs:
        yield "No relevant documents found. Please ingest documents first."
        return

    if use_reranking and len(docs) > settings.rerank_top_k:
        docs = rerank_documents(question, docs)

    # Build streaming LLM
    llm = ChatOpenAI(
        model=settings.llm_model,
        openai_api_key=settings.openrouter_api_key,
        openai_api_base=settings.openrouter_base_url,
        temperature=0.2,
        max_tokens=1024,
        streaming=True,  # NEW
    )

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = prompt | llm | StrOutputParser()

    context = format_context(docs)
    
    # Stream tokens
    for chunk in chain.stream({"context": context, "question": question}):
        yield chunk