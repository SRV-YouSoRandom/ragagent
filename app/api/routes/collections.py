from fastapi import APIRouter, HTTPException
from services.vector_store import (
    ensure_collection,
    list_collections,
    delete_collection,
    get_collection_stats,
)
from schemas.collections import CollectionCreate, CollectionStats, CollectionList

router = APIRouter()


@router.post("/collections", status_code=201)
async def create_collection(request: CollectionCreate):
    """Create a new collection."""
    try:
        ensure_collection(request.name)
        return {"message": f"Collection '{request.name}' created successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/collections", response_model=CollectionList)
async def get_collections():
    """List all collections."""
    try:
        collections = list_collections()
        return CollectionList(collections=collections)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/collections/{name}", response_model=CollectionStats)
async def get_collection_info(name: str):
    """Get collection statistics."""
    try:
        stats = get_collection_stats(name)
        return CollectionStats(**stats)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Collection not found: {name}")


@router.delete("/collections/{name}")
async def remove_collection(name: str):
    """Delete a collection."""
    try:
        delete_collection(name)
        return {"message": f"Collection '{name}' deleted successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))