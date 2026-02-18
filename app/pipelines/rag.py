from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from services.embedder import embed_query
from services.vector_store import search
from services.reranker import rerank_documents
from services.memory import get_or_create_session, add_to_memory
from core.config import get_settings
from core.metrics import get_metrics_collector, QueryMetrics  # NEW
import logging
import time

logger = logging.getLogger("rag_agent")

PROMPT_TEMPLATE_WITH_HISTORY = """You are a helpful assistant. Answer the question based ONLY on the context provided and the conversation history.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Answer the question considering the conversation history above."""


def format_context(docs: list[dict]) -> str:
    return "\n\n---\n\n".join(
        [f"[Source: {d['filename']} | Score: {d.get('rerank_score', d['score']):.3f}]\n{d['text']}" for d in docs]
    )


def run_rag(
    question: str,
    collection_name: str = None,
    session_id: str = None,
    use_reranking: bool = True
) -> dict:
    settings = get_settings()
    metrics_collector = get_metrics_collector()
    
    # Initialize metrics
    metrics = QueryMetrics(
        session_id=session_id or "none",
        question=question,
        collection_name=collection_name or settings.qdrant_collection,
    )
    start_time = time.time()
    
    try:
        # Get or create session
        session_id, memory = get_or_create_session(session_id)

        # 1. Embed query (with timing)
        embed_start = time.time()
        query_vector = embed_query(question)
        metrics.embedding_latency_ms = (time.time() - embed_start) * 1000

        # 2. Retrieve (with timing)
        retrieval_start = time.time()
        initial_top_k = settings.top_k * 2 if use_reranking else settings.top_k
        docs = search(query_vector, top_k=initial_top_k, collection_name=collection_name)
        metrics.retrieval_latency_ms = (time.time() - retrieval_start) * 1000
        metrics.num_docs_retrieved = len(docs)
        
        if not docs:
            metrics.total_latency_ms = (time.time() - start_time) * 1000
            metrics_collector.record_query(metrics)
            return {
                "answer": "No relevant documents found. Please ingest documents first.",
                "sources": [],
                "collection_name": collection_name or settings.qdrant_collection,
                "session_id": session_id,
            }
        
        metrics.avg_retrieval_score = sum(d["score"] for d in docs) / len(docs)

        # 3. Rerank (with timing)
        if use_reranking and len(docs) > settings.rerank_top_k:
            rerank_start = time.time()
            docs = rerank_documents(question, docs)
            metrics.rerank_latency_ms = (time.time() - rerank_start) * 1000
            metrics.num_docs_after_rerank = len(docs)
            metrics.avg_rerank_score = sum(d.get("rerank_score", 0) for d in docs) / len(docs)
        else:
            metrics.num_docs_after_rerank = len(docs)

        # 4. Build LLM chain
        llm = ChatOpenAI(
            model=settings.llm_model,
            openai_api_key=settings.openrouter_api_key,
            openai_api_base=settings.openrouter_base_url,
            temperature=0.2,
            max_tokens=1024,
        )

        prompt = ChatPromptTemplate.from_messages([
            # Commented out system prompt for now as it's not supported by Gemma
            # ("system", "You are a helpful assistant."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", PROMPT_TEMPLATE_WITH_HISTORY),
        ])

        chain = prompt | llm | StrOutputParser()

        # 5. Generate (with timing)
        llm_start = time.time()
        context = format_context(docs)
        chat_history = memory.load_memory_variables({})["chat_history"]
        
        answer = chain.invoke({
            "context": context,
            "question": question,
            "chat_history": chat_history,
        })
        metrics.llm_latency_ms = (time.time() - llm_start) * 1000
        metrics.tokens_used = len(answer.split())  # Rough estimate

        # 6. Save to memory
        add_to_memory(session_id, question, answer)

        # Record metrics
        metrics.total_latency_ms = (time.time() - start_time) * 1000
        metrics.success = True
        metrics_collector.record_query(metrics)

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
            "session_id": session_id,
        }
    
    except Exception as e:
        metrics.total_latency_ms = (time.time() - start_time) * 1000
        metrics.success = False
        metrics.error = str(e)
        metrics_collector.record_query(metrics)
        raise


def run_rag_streaming(
    question: str,
    collection_name: str = None,
    session_id: str = None,
    use_reranking: bool = True
):
    """Streaming version with metrics."""
    settings = get_settings()
    metrics_collector = get_metrics_collector()
    
    metrics = QueryMetrics(
        session_id=session_id or "none",
        question=question,
        collection_name=collection_name or settings.qdrant_collection,
    )
    start_time = time.time()
    
    try:
        session_id, memory = get_or_create_session(session_id)

        # Retrieval (with timing)
        embed_start = time.time()
        query_vector = embed_query(question)
        metrics.embedding_latency_ms = (time.time() - embed_start) * 1000
        
        retrieval_start = time.time()
        initial_top_k = settings.top_k * 2 if use_reranking else settings.top_k
        docs = search(query_vector, top_k=initial_top_k, collection_name=collection_name)
        metrics.retrieval_latency_ms = (time.time() - retrieval_start) * 1000
        metrics.num_docs_retrieved = len(docs)
        
        if not docs:
            metrics.total_latency_ms = (time.time() - start_time) * 1000
            metrics_collector.record_query(metrics)
            yield "No relevant documents found. Please ingest documents first."
            return
        
        if use_reranking and len(docs) > settings.rerank_top_k:
            rerank_start = time.time()
            docs = rerank_documents(question, docs)
            metrics.rerank_latency_ms = (time.time() - rerank_start) * 1000
            metrics.num_docs_after_rerank = len(docs)

        llm = ChatOpenAI(
            model=settings.llm_model,
            openai_api_key=settings.openrouter_api_key,
            openai_api_base=settings.openrouter_base_url,
            temperature=0.2,
            max_tokens=1024,
            streaming=True,
        )

        prompt = ChatPromptTemplate.from_messages([
            # Commented out system prompt for now as it's not supported by Gemma
            # ("system", "You are a helpful assistant."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", PROMPT_TEMPLATE_WITH_HISTORY),
        ])

        chain = prompt | llm | StrOutputParser()

        context = format_context(docs)
        chat_history = memory.load_memory_variables({})["chat_history"]
        
        # Stream and collect
        llm_start = time.time()
        full_answer = ""
        for chunk in chain.stream({
            "context": context,
            "question": question,
            "chat_history": chat_history,
        }):
            full_answer += chunk
            yield chunk
        
        metrics.llm_latency_ms = (time.time() - llm_start) * 1000
        metrics.total_latency_ms = (time.time() - start_time) * 1000
        metrics.tokens_used = len(full_answer.split())
        metrics.success = True
        metrics_collector.record_query(metrics)
        
        add_to_memory(session_id, question, full_answer)
    
    except Exception as e:
        metrics.total_latency_ms = (time.time() - start_time) * 1000
        metrics.success = False
        metrics.error = str(e)
        metrics_collector.record_query(metrics)
        raise