from fastapi import APIRouter, HTTPException, BackgroundTasks
from app.services.data_ingestion import DataIngestionService
from app.services.data_preprocessing import DataPreprocessingService
from app.services.ml_training import MLTrainingService
from app.models.schemas import TrainingRequest, TrainingResponse, EDAResponse
from app.core.config import Settings
import asyncio
from typing import Dict, Any

router = APIRouter()
settings = Settings()

# Initialize services
data_service = DataIngestionService(settings.upload_dir)
preprocessing_service = DataPreprocessingService()
training_service = MLTrainingService(settings.models_dir)

# Store for background training tasks
training_tasks = {}

@router.get("/eda/{upload_id}", response_model=EDAResponse)
async def generate_eda(upload_id: str):
    """Generate Exploratory Data Analysis for uploaded dataset"""
    try:
        # Load dataset
        df, dataset_info = data_service.load_dataset(upload_id)
        
        # Generate EDA
        eda_result = preprocessing_service.generate_eda(
            df, upload_id, dataset_info["column_types"]
        )
        
        return eda_result
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating EDA: {str(e)}")

@router.post("/start", response_model=TrainingResponse)
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Start automated ML training"""
    try:
        # Load and validate dataset
        df, dataset_info = data_service.load_dataset(request.upload_id)
        
        if request.target_column not in df.columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Target column '{request.target_column}' not found in dataset"
            )
        
        # Preprocess data with enhanced quality checks
        preprocessed_data = preprocessing_service.preprocess_data(
            df, 
            request.target_column,
            request.problem_type.value,
            dataset_info["column_types"],
            request.test_size,
            dataset_info  # Pass full dataset info for quality checks
        )
        
        # Start training
        training_results = training_service.train_models(
            preprocessed_data,
            request.problem_type.value,
            request.cv_folds,
            request.models_to_train
        )
        
        # Create response
        response = TrainingResponse(
            message="Training completed successfully",
            training_id=training_results["training_id"],
            problem_type=training_results["problem_type"],
            best_model=training_results["best_model"],
            model_performances=training_results["model_performances"],
            training_summary=training_results["training_summary"]
        )
        
        return response
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")

@router.get("/status/{training_id}")
async def get_training_status(training_id: str):
    """Get training status (for background training)"""
    if training_id in training_tasks:
        task = training_tasks[training_id]
        if task.done():
            try:
                result = await task
                return {"status": "completed", "result": result}
            except Exception as e:
                return {"status": "failed", "error": str(e)}
        else:
            return {"status": "running"}
    else:
        # Check if model exists (completed training)
        try:
            artifacts = training_service.load_model(training_id)
            return {"status": "completed", "model_exists": True}
        except:
            raise HTTPException(status_code=404, detail="Training ID not found")

@router.get("/results/{training_id}")
async def get_training_results(training_id: str):
    """Get detailed training results"""
    try:
        artifacts = training_service.load_model(training_id)
        metadata = artifacts["metadata"]
        
        return {
            "training_id": training_id,
            "problem_type": metadata["problem_type"],
            "best_model": metadata["best_model_name"],
            "model_performances": metadata["model_performances"],
            "feature_names": metadata.get("feature_names", []),
            "training_timestamp": metadata.get("timestamp")
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))