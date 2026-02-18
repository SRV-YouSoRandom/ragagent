from fastapi import APIRouter
from core.metrics import get_metrics_collector
from typing import Dict, List

router = APIRouter()


@router.get("/metrics/summary")
async def get_metrics_summary() -> Dict:
    """Get aggregated metrics."""
    collector = get_metrics_collector()
    return collector.get_summary()


@router.get("/metrics/recent")
async def get_recent_queries(limit: int = 20) -> List[Dict]:
    """Get recent query details."""
    collector = get_metrics_collector()
    return collector.get_recent_queries(limit)