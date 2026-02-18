import os
from core.config import get_settings


def setup_langsmith_tracing():
    """Configure LangSmith tracing if enabled."""
    settings = get_settings()
    
    if settings.langsmith_tracing and settings.langsmith_api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
        
        import logging
        logger = logging.getLogger("rag_agent")
        logger.info(f"LangSmith tracing enabled for project: {settings.langsmith_project}")
    else:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"