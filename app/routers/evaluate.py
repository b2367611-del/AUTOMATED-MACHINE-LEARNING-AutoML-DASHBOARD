from fastapi import APIRouter, HTTPException
from app.services.ml_training import MLTrainingService
from app.models.schemas import PredictionRequest, PredictionResponse
from app.core.config import Settings
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder
from typing import Dict, Any

router = APIRouter()
settings = Settings()
training_service = MLTrainingService(settings.models_dir)

@router.get("/metrics/{training_id}")
async def get_model_metrics(training_id: str):
    """Get detailed metrics for a trained model"""
    try:
        artifacts = training_service.load_model(training_id)
        metadata = artifacts["metadata"]
        
        # Get model performances
        model_performances = metadata["model_performances"]
        
        # Create comparison chart
        model_names = [perf["model_name"] for perf in model_performances]
        
        if metadata["problem_type"] == "classification":
            primary_metric = "accuracy"
            metric_values = [perf["metrics"].get(primary_metric, 0) for perf in model_performances]
        else:
            primary_metric = "r2_score"
            metric_values = [perf["metrics"].get(primary_metric, 0) for perf in model_performances]
        
        # Create comparison chart
        fig = px.bar(
            x=model_names,
            y=metric_values,
            title=f"Model Comparison ({primary_metric.title()})",
            labels={"x": "Models", "y": primary_metric.title()}
        )
        fig.update_xaxis(tickangle=45)
        fig.update_layout(
            height=400,
            margin=dict(t=50, b=100, l=50, r=50)
        )
        
        comparison_chart = json.dumps(fig, cls=PlotlyJSONEncoder)
        
        return {
            "training_id": training_id,
            "problem_type": metadata["problem_type"],
            "model_performances": model_performances,
            "comparison_chart": comparison_chart,
            "best_model": metadata["best_model_name"]
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/feature-importance/{training_id}")
async def get_feature_importance(training_id: str):
    """Get feature importance for the trained model"""
    try:
        importance_data = training_service.get_feature_importance(training_id)
        
        if importance_data["feature_importance"]:
            # Create feature importance chart
            features = list(importance_data["feature_importance"].keys())[:15]  # Top 15
            importances = list(importance_data["feature_importance"].values())[:15]
            
            fig = px.bar(
                x=importances,
                y=features,
                orientation='h',
                title=f"Feature Importance - {importance_data['model_name']}",
                labels={"x": "Importance", "y": "Features"}
            )
            fig.update_layout(
                yaxis={'categoryorder':'total ascending'},
                height=500,
                margin=dict(t=50, b=50, l=100, r=50)
            )
            
            chart = json.dumps(fig, cls=PlotlyJSONEncoder)
            
            return {
                "training_id": training_id,
                "model_name": importance_data["model_name"],
                "feature_importance": importance_data["feature_importance"],
                "chart": chart
            }
        else:
            return {
                "training_id": training_id,
                "model_name": importance_data["model_name"],
                "feature_importance": {},
                "chart": None,
                "message": "Feature importance not available for this model type"
            }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict/{training_id}", response_model=PredictionResponse)
async def make_predictions(training_id: str, request: PredictionRequest):
    """Make predictions using a trained model"""
    try:
        # Convert data to DataFrame
        df = pd.DataFrame(request.data)
        
        # Make predictions
        result = training_service.predict(training_id, df)
        
        return PredictionResponse(
            predictions=result["predictions"],
            model_info=result["model_info"]
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/confusion-matrix/{training_id}")
async def get_confusion_matrix(training_id: str):
    """Get confusion matrix for classification models"""
    try:
        artifacts = training_service.load_model(training_id)
        metadata = artifacts["metadata"]
        
        if metadata["problem_type"] != "classification":
            raise HTTPException(
                status_code=400, 
                detail="Confusion matrix only available for classification models"
            )
        
        # This would require storing the test predictions during training
        # For now, return a placeholder message
        return {
            "training_id": training_id,
            "message": "Confusion matrix visualization would be implemented with stored test predictions"
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))