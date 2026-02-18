from typing import Dict, List
from dataclasses import dataclass, field
from datetime import datetime
import time
import logging

logger = logging.getLogger("rag_agent")


@dataclass
class QueryMetrics:
    """Metrics for a single query."""
    session_id: str
    question: str
    collection_name: str
    
    # Timing
    total_latency_ms: float = 0.0
    embedding_latency_ms: float = 0.0
    retrieval_latency_ms: float = 0.0
    rerank_latency_ms: float = 0.0
    llm_latency_ms: float = 0.0
    
    # Retrieval
    num_docs_retrieved: int = 0
    num_docs_after_rerank: int = 0
    avg_retrieval_score: float = 0.0
    avg_rerank_score: float = 0.0
    
    # LLM
    tokens_used: int = 0  # Estimated
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    success: bool = True
    error: str = ""


class MetricsCollector:
    """In-memory metrics collector."""
    
    def __init__(self):
        self.queries: List[QueryMetrics] = []
        self._max_queries = 1000  # Keep last 1000 queries
    
    def record_query(self, metrics: QueryMetrics):
        """Record query metrics."""
        self.queries.append(metrics)
        
        # Keep only recent queries
        if len(self.queries) > self._max_queries:
            self.queries = self.queries[-self._max_queries:]
        
        logger.info(
            f"Query metrics | Latency: {metrics.total_latency_ms:.0f}ms | "
            f"Docs: {metrics.num_docs_retrieved}→{metrics.num_docs_after_rerank} | "
            f"Success: {metrics.success}"
        )
    
    def get_summary(self) -> Dict:
        """Get aggregated metrics."""
        if not self.queries:
            return {"total_queries": 0}
        
        successful = [q for q in self.queries if q.success]
        failed = [q for q in self.queries if not q.success]
        
        return {
            "total_queries": len(self.queries),
            "successful_queries": len(successful),
            "failed_queries": len(failed),
            "avg_latency_ms": sum(q.total_latency_ms for q in successful) / len(successful) if successful else 0,
            "avg_docs_retrieved": sum(q.num_docs_retrieved for q in successful) / len(successful) if successful else 0,
            "avg_docs_after_rerank": sum(q.num_docs_after_rerank for q in successful) / len(successful) if successful else 0,
            "p95_latency_ms": self._percentile([q.total_latency_ms for q in successful], 95) if successful else 0,
            "p99_latency_ms": self._percentile([q.total_latency_ms for q in successful], 99) if successful else 0,
        }
    
    def get_recent_queries(self, limit: int = 20) -> List[Dict]:
        """Get recent query details."""
        return [
            {
                "timestamp": q.timestamp.isoformat(),
                "question": q.question[:100],  # Truncate for privacy
                "latency_ms": q.total_latency_ms,
                "success": q.success,
                "collection": q.collection_name,
            }
            for q in self.queries[-limit:]
        ]
    
    @staticmethod
    def _percentile(values: List[float], percentile: int) -> float:
        """Calculate percentile."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]


# Global metrics collector
_metrics_collector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    return _metrics_collector