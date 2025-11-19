from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from jinja2 import Environment, FileSystemLoader
import json
from pathlib import Path
from app.services.data_ingestion import DataIngestionService
from app.services.data_preprocessing import DataPreprocessingService
import os

router = APIRouter(tags=["Data Quality"])

# Setup Jinja2 templates
template_dir = Path(__file__).parent.parent / "templates"
jinja_env = Environment(loader=FileSystemLoader(template_dir))

@router.get("/report/{upload_id}", response_class=HTMLResponse)
async def get_quality_report(upload_id: str):
    """Generate and serve HTML data quality report for uploaded dataset"""
    
    try:
        # Get dataset info and data
        data_service = DataIngestionService()
        df, dataset_info = data_service.load_dataset(upload_id)
        
        # Get preprocessing service and generate quality report
        preprocessing_service = DataPreprocessingService()
        
        # Generate a sample preprocessing for quality analysis
        column_types = dataset_info.get("column_types", {})
        
        # Generate quality report without full preprocessing
        quality_report = preprocessing_service.quality_reporter.generate_data_summary(
            df=df,
            column_types=column_types,
            preprocessing_info={
                'outlier_info': {},
                'scaling_info': {},
                'encoding_info': {},
                'engineered_features': []
            }
        )
        
        # Prepare template data
        template_data = {
            "dataset_name": dataset_info.get("filename", f"Dataset {upload_id}"),
            "generated_at": quality_report["generated_at"],
            "quality_score": quality_report["data_quality_score"],
            "overview": quality_report["dataset_overview"],
            "missing_data": quality_report["missing_data_analysis"],
            "column_analysis": quality_report["column_analysis"],
            "outlier_analysis": quality_report["outlier_analysis"],
            "preprocessing_actions": quality_report["preprocessing_actions"],
            "recommendations": quality_report["recommendations"]
        }
        
        # Add Jinja2 filters
        def number_format(value):
            if isinstance(value, (int, float)):
                return f"{value:,}"
            return value
        
        jinja_env.filters['number_format'] = number_format
        
        # Render template
        template = jinja_env.get_template("data_quality_report.html")
        html_content = template.render(**template_data)
        
        return HTMLResponse(content=html_content)
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating quality report: {str(e)}")

@router.get("/report/{upload_id}/json")
async def get_quality_report_json(upload_id: str):
    """Get data quality report as JSON for API consumption"""
    
    try:
        # Get dataset info and data
        data_service = DataIngestionService()
        df, dataset_info = data_service.load_dataset(upload_id)
        
        # Get preprocessing service and generate quality report
        preprocessing_service = DataPreprocessingService()
        
        # Generate quality report
        column_types = dataset_info.get("column_types", {})
        quality_report = preprocessing_service.quality_reporter.generate_data_summary(
            df=df,
            column_types=column_types,
            preprocessing_info={
                'outlier_info': {},
                'scaling_info': {},
                'encoding_info': {},
                'engineered_features': []
            }
        )
        
        return quality_report
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating quality report: {str(e)}")