from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from services.embedder import embed_query
from services.vector_store import search
from services.reranker import rerank_documents
from services.memory import get_or_create_session, add_to_memory  # NEW
from core.config import get_settings
import logging

logger = logging.getLogger("rag_agent")

# NEW: Updated prompt with chat history
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
    session_id: str = None,  # NEW
    use_reranking: bool = True
) -> dict:
    settings = get_settings()

    # 1. Get or create session memory (NEW)
    session_id, memory = get_or_create_session(session_id)

    # 2. Embed query
    query_vector = embed_query(question)

    # 3. Retrieve
    initial_top_k = settings.top_k * 2 if use_reranking else settings.top_k
    docs = search(query_vector, top_k=initial_top_k, collection_name=collection_name)
    
    if not docs:
        return {
            "answer": "No relevant documents found. Please ingest documents first.",
            "sources": [],
            "collection_name": collection_name or settings.qdrant_collection,
            "session_id": session_id,
        }

    # 4. Rerank
    if use_reranking and len(docs) > settings.rerank_top_k:
        docs = rerank_documents(question, docs)

    # 5. Build LLM chain with memory (NEW)
    llm = ChatOpenAI(
        model=settings.llm_model,
        openai_api_key=settings.openrouter_api_key,
        openai_api_base=settings.openrouter_base_url,
        temperature=0.2,
        max_tokens=1024,
    )

    # NEW: Prompt with chat history placeholder
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", PROMPT_TEMPLATE_WITH_HISTORY),
    ])

    chain = prompt | llm | StrOutputParser()

    # 6. Generate with history
    context = format_context(docs)
    chat_history = memory.load_memory_variables({})["chat_history"]
    
    answer = chain.invoke({
        "context": context,
        "question": question,
        "chat_history": chat_history,
    })

    # 7. Save to memory (NEW)
    add_to_memory(session_id, question, answer)

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
        "session_id": session_id,  # NEW
    }


def run_rag_streaming(
    question: str,
    collection_name: str = None,
    session_id: str = None,  # NEW
    use_reranking: bool = True
):
    """Streaming version with conversation memory."""
    settings = get_settings()

    # Get or create session
    session_id, memory = get_or_create_session(session_id)

    # Retrieval (same as before)
    query_vector = embed_query(question)
    initial_top_k = settings.top_k * 2 if use_reranking else settings.top_k
    docs = search(query_vector, top_k=initial_top_k, collection_name=collection_name)
    
    if not docs:
        yield "No relevant documents found. Please ingest documents first."
        return

    if use_reranking and len(docs) > settings.rerank_top_k:
        docs = rerank_documents(question, docs)

    # Build streaming LLM with memory
    llm = ChatOpenAI(
        model=settings.llm_model,
        openai_api_key=settings.openrouter_api_key,
        openai_api_base=settings.openrouter_base_url,
        temperature=0.2,
        max_tokens=1024,
        streaming=True,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", PROMPT_TEMPLATE_WITH_HISTORY),
    ])

    chain = prompt | llm | StrOutputParser()

    context = format_context(docs)
    chat_history = memory.load_memory_variables({})["chat_history"]
    
    # Stream and collect for memory
    full_answer = ""
    for chunk in chain.stream({
        "context": context,
        "question": question,
        "chat_history": chat_history,
    }):
        full_answer += chunk
        yield chunk
    
    # Save to memory after streaming completes
    add_to_memory(session_id, question, full_answer)