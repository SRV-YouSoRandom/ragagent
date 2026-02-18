from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from pipelines.ingest import run_ingest
from schemas.ingest import IngestResponse
from core.config import get_settings

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    collection_name: str = Form(None)  # NEW: Optional collection
):
    settings = get_settings()

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    file_bytes = await file.read()
    size_mb = len(file_bytes) / (1024 * 1024)

    if size_mb > settings.max_file_size_mb:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size: {settings.max_file_size_mb}MB"
        )

    try:
        result = run_ingest(file_bytes, file.filename, collection_name)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingest failed: {str(e)}")

    return IngestResponse(**result, message="Document ingested successfully.")