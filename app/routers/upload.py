from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from app.services.data_ingestion import DataIngestionService
from app.models.schemas import DataUploadResponse
from app.core.config import Settings
import os

router = APIRouter()
settings = Settings()
data_service = DataIngestionService(settings.upload_dir)

@router.post("/", response_model=DataUploadResponse)
async def upload_dataset(file: UploadFile = File(...)):
    """Upload and analyze a dataset"""
    
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected")
    
    # Check file extension
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in settings.allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format. Allowed: {settings.allowed_extensions}"
        )
    
    # Check file size
    contents = await file.read()
    if len(contents) > settings.max_file_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {settings.max_file_size / (1024*1024):.1f}MB"
        )
    
    try:
        # Process upload
        upload_id, dataset_info = await data_service.process_upload(contents, file.filename)
        
        return DataUploadResponse(
            message="Dataset uploaded and analyzed successfully",
            dataset_info=dataset_info,
            upload_id=upload_id
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@router.get("/info/{upload_id}")
async def get_dataset_info(upload_id: str):
    """Get information about an uploaded dataset"""
    try:
        _, dataset_info = data_service.load_dataset(upload_id)
        return {"dataset_info": dataset_info}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))