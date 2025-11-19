from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum

class ProblemType(str, Enum):
    """ML Problem types"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    AUTO_DETECT = "auto_detect"

class ColumnType(str, Enum):
    """Column data types"""
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    TEXT = "text"

class DatasetInfo(BaseModel):
    """Dataset information response with quality checks"""
    filename: str
    shape: tuple
    columns: List[str]
    column_types: Dict[str, str]
    missing_values: Dict[str, int]
    target_suggestions: List[str]
    problem_type: Optional[str] = None
    quality_warnings: Optional[List[str]] = []
    columns_to_drop: Optional[List[str]] = []
    columns_to_hash_encode: Optional[List[str]] = []

class DataUploadResponse(BaseModel):
    """Response after uploading data"""
    message: str
    dataset_info: DatasetInfo
    upload_id: str

class TrainingRequest(BaseModel):
    """Training configuration request"""
    upload_id: str
    target_column: str
    problem_type: ProblemType
    test_size: float = Field(default=0.2, ge=0.1, le=0.5)
    cv_folds: int = Field(default=5, ge=3, le=10)
    models_to_train: Optional[List[str]] = None

class ModelPerformance(BaseModel):
    """Individual model performance"""
    model_name: str
    metrics: Dict[str, float]
    training_time: float
    is_best: bool = False
    
    model_config = {"protected_namespaces": ()}

class TrainingResponse(BaseModel):
    """Training results response"""
    message: str
    training_id: str
    problem_type: str
    best_model: str
    model_performances: List[ModelPerformance]
    training_summary: Dict[str, Any]
    
    model_config = {"protected_namespaces": ()}

class PredictionRequest(BaseModel):
    """Prediction request"""
    model_id: str
    data: List[Dict[str, Any]]
    
    model_config = {"protected_namespaces": ()}

class PredictionResponse(BaseModel):
    """Prediction response"""
    predictions: List[Any]
    model_info: Dict[str, Any]
    
    model_config = {"protected_namespaces": ()}

class EDAResponse(BaseModel):
    """Exploratory Data Analysis response"""
    upload_id: str
    summary_stats: Dict[str, Any]
    correlation_matrix: Optional[Dict[str, Any]] = None
    missing_data_info: Dict[str, Any]
    visualizations: List[Dict[str, Any]]