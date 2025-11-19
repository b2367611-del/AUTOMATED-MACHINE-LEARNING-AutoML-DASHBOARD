try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    """Application settings"""
    
    # App settings
    app_name: str = "AutoML Platform"
    debug: bool = True
    
    # File upload settings
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    allowed_extensions: list = [".csv", ".xlsx", ".xls"]
    upload_dir: str = "uploads"
    models_dir: str = "trained_models"
    
    # ML settings
    test_size: float = 0.2
    cv_folds: int = 5
    random_state: int = 42
    
    # Model settings
    max_models_to_train: int = 10
    
    class Config:
        env_file = ".env"