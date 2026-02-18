from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from app.services.embedder import embed_query
from app.services.vector_store import search
from app.core.config import get_settings
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
        [f"[Source: {d['filename']} | Score: {d['score']:.3f}]\n{d['text']}" for d in docs]
    )


def run_rag(question: str) -> dict:
    settings = get_settings()

    # 1. Embed query
    query_vector = embed_query(question)

    # 2. Retrieve
    docs = search(query_vector, top_k=settings.top_k)
    if not docs:
        return {"answer": "No relevant documents found. Please ingest documents first.", "sources": []}

    # 3. Build LLM chain
    llm = ChatOpenAI(
        model=settings.llm_model,
        openai_api_key=settings.openrouter_api_key,
        openai_api_base=settings.openrouter_base_url,
        temperature=0.2,
        max_tokens=1024,
    )

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    chain = prompt | llm | StrOutputParser()

    # 4. Generate
    context = format_context(docs)
    answer = chain.invoke({"context": context, "question": question})

    return {
        "answer": answer,
        "sources": [{"filename": d["filename"], "score": d["score"]} for d in docs],
    }