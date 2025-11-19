import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import os
import uuid
from pathlib import Path
import json
from app.models.schemas import DatasetInfo, ColumnType, ProblemType

class DataIngestionService:
    """Service for handling data upload and analysis"""
    
    def __init__(self, upload_dir: str = "uploads"):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(exist_ok=True)
    
    async def process_upload(self, file_content: bytes, filename: str) -> Tuple[str, DatasetInfo]:
        """Process uploaded file and return dataset information"""
        
        # Generate unique upload ID
        upload_id = str(uuid.uuid4())
        
        # Save file
        file_path = self.upload_dir / f"{upload_id}_{filename}"
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        # Read and analyze data
        df = self._read_file(file_path)
        dataset_info = self._analyze_dataset(df, filename)
        
        # Save dataset info
        info_path = self.upload_dir / f"{upload_id}_info.json"
        with open(info_path, "w") as f:
            json.dump({
                "dataset_info": dataset_info.dict(),
                "file_path": str(file_path)
            }, f)
        
        return upload_id, dataset_info
    
    def _read_file(self, file_path: Path) -> pd.DataFrame:
        """Read CSV or Excel file"""
        if file_path.suffix.lower() == '.csv':
            try:
                # Try different encodings
                for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
                    try:
                        return pd.read_csv(file_path, encoding=encoding)
                    except UnicodeDecodeError:
                        continue
                raise ValueError("Could not decode CSV file")
            except Exception as e:
                raise ValueError(f"Error reading CSV: {str(e)}")
        
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            try:
                return pd.read_excel(file_path, engine='openpyxl')
            except Exception as e:
                raise ValueError(f"Error reading Excel: {str(e)}")
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def _analyze_dataset(self, df: pd.DataFrame, filename: str) -> DatasetInfo:
        """Analyze dataset with enhanced data quality checks"""
        
        # Basic info
        shape = df.shape
        columns = df.columns.tolist()
        
        # Enhanced column analysis with quality checks
        column_types = {}
        quality_warnings = []
        columns_to_drop = []
        columns_to_hash_encode = []
        
        for col in columns:
            # Basic type inference
            column_types[col] = self._infer_column_type(df[col])
            
            # Data quality checks
            quality_issues = self._check_column_quality(df[col], col)
            quality_warnings.extend(quality_issues['warnings'])
            
            if quality_issues['should_drop']:
                columns_to_drop.append(col)
            elif quality_issues['should_hash_encode']:
                columns_to_hash_encode.append(col)
        
        # Missing values
        missing_values = df.isnull().sum().to_dict()
        
        # Target suggestions
        target_suggestions = self._suggest_target_columns(df, column_types)
        
        # Problem type suggestion
        problem_type = self._suggest_problem_type(df, target_suggestions)
        
        return DatasetInfo(
            filename=filename,
            shape=shape,
            columns=columns,
            column_types=column_types,
            missing_values=missing_values,
            target_suggestions=target_suggestions,
            problem_type=problem_type,
            quality_warnings=quality_warnings,
            columns_to_drop=columns_to_drop,
            columns_to_hash_encode=columns_to_hash_encode
        )
    
    def _infer_column_type(self, series: pd.Series) -> str:
        """Enhanced column type inference with intelligent mostly-numeric handling"""
        
        # Check if datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            return ColumnType.DATETIME.value
        
        # Check if numeric
        if pd.api.types.is_numeric_dtype(series):
            return ColumnType.NUMERICAL.value
        
        # Enhanced logic for object (string) columns
        if series.dtype == 'object':
            # Try to convert to numeric
            numeric_series = pd.to_numeric(series, errors='coerce')
            if not numeric_series.isna().all():
                # Calculate conversion rate (excluding original nulls)
                original_non_null = series.notna().sum()
                converted_non_null = numeric_series.notna().sum()
                
                if original_non_null > 0:
                    conversion_rate = converted_non_null / original_non_null
                    
                    # 80-100% numeric: Treat as mostly numeric (clean and convert)
                    if conversion_rate >= 0.8:
                        return ColumnType.NUMERICAL.value  # Will be cleaned in preprocessing
                    
                    # 50-80% mixed: Analyze context for mixed-type handling
                    elif conversion_rate >= 0.5:
                        # Check if text values look like missing data markers
                        non_numeric_values = series[pd.to_numeric(series, errors='coerce').isna() & series.notna()]
                        if len(non_numeric_values) > 0:
                            common_missing_markers = {'na', 'n/a', 'null', 'none', 'missing', '', 'nan', 'unknown', 'undefined'}
                            text_values_lower = set(str(v).lower().strip() for v in non_numeric_values.unique()[:10])  # Sample first 10
                            
                            # If most text values are missing markers, treat as numeric
                            missing_marker_overlap = len(text_values_lower.intersection(common_missing_markers))
                            if missing_marker_overlap > 0 or len(text_values_lower) <= 3:  # Few distinct text values
                                return ColumnType.NUMERICAL.value  # Clean and convert
                        
                        # Otherwise, treat as categorical for mixed content
                        return ColumnType.CATEGORICAL.value
        
        # Check if categorical (low cardinality relative to dataset size)
        unique_ratio = series.nunique() / len(series)
        if unique_ratio < 0.1 or series.nunique() < 20:
            return ColumnType.CATEGORICAL.value
        
        # Default to text
        return ColumnType.TEXT.value
    
    def _check_column_quality(self, series: pd.Series, col_name: str) -> dict:
        """Check data quality and suggest actions for a column"""
        warnings = []
        should_drop = False
        should_hash_encode = False
        
        total_rows = len(series)
        
        # Check 1: >50% missing values
        missing_count = series.isnull().sum()
        missing_pct = (missing_count / total_rows) * 100 if total_rows > 0 else 0
        
        if missing_pct > 50:
            warnings.append(f"Column '{col_name}': {missing_pct:.1f}% missing values - RECOMMEND DROPPING")
            should_drop = True
        elif missing_pct > 30:
            warnings.append(f"Column '{col_name}': {missing_pct:.1f}% missing values - consider dropping")
        
        # Check 2: >95% same value (near-constant)
        if not series.empty:
            value_counts = series.value_counts(dropna=False)
            if len(value_counts) > 0:
                most_common_count = value_counts.iloc[0]
                most_common_pct = (most_common_count / total_rows) * 100
                
                if most_common_pct > 95:
                    most_common_value = value_counts.index[0]
                    warnings.append(f"Column '{col_name}': {most_common_pct:.1f}% same value ('{most_common_value}') - RECOMMEND DROPPING")
                    should_drop = True
                elif most_common_pct > 85:
                    most_common_value = value_counts.index[0]
                    warnings.append(f"Column '{col_name}': {most_common_pct:.1f}% same value ('{most_common_value}') - low variance")
        
        # Check 3: >100 unique categories (for categorical columns)
        if series.dtype == 'object' and not should_drop:
            unique_count = series.nunique()
            unique_ratio = unique_count / total_rows if total_rows > 0 else 0
            
            if unique_count > 100 and unique_ratio > 0.8:  # High cardinality
                warnings.append(f"Column '{col_name}': {unique_count} unique categories - RECOMMEND HASH ENCODING")
                should_hash_encode = True
            elif unique_count > 50 and unique_ratio > 0.5:
                warnings.append(f"Column '{col_name}': {unique_count} unique categories - consider hash encoding")
        
        # Check 4: Mixed type quality (numbers + strings)
        if series.dtype == 'object' and not should_drop:
            numeric_series = pd.to_numeric(series, errors='coerce')
            numeric_count = numeric_series.notna().sum()
            non_null_count = series.notna().sum()
            
            if non_null_count > 0:
                numeric_ratio = numeric_count / non_null_count
                if 0.2 < numeric_ratio < 0.8:  # Truly mixed (not mostly one type)
                    warnings.append(f"Column '{col_name}': Mixed numeric/text content ({numeric_ratio:.1%} numeric)")
        
        return {
            'warnings': warnings,
            'should_drop': should_drop,
            'should_hash_encode': should_hash_encode,
            'missing_pct': missing_pct,
            'unique_count': series.nunique() if not should_drop else 0
        }
    
    def _suggest_target_columns(self, df: pd.DataFrame, column_types: Dict[str, str]) -> List[str]:
        """Suggest possible target columns"""
        suggestions = []
        
        # Look for common target column names
        target_keywords = [
            'target', 'label', 'class', 'category', 'outcome', 'result',
            'y', 'dependent', 'response', 'price', 'value', 'amount',
            'prediction', 'predict', 'output'
        ]
        
        for col in df.columns:
            col_lower = col.lower()
            
            # Check for keyword matches
            if any(keyword in col_lower for keyword in target_keywords):
                suggestions.append(col)
                continue
            
            # Check if it's the last column (common convention)
            if col == df.columns[-1]:
                suggestions.append(col)
                continue
            
            # For numerical columns, check if it might be a target
            if column_types[col] == ColumnType.NUMERICAL.value:
                # Check if it has reasonable variance
                if df[col].nunique() > 1 and df[col].nunique() < len(df) * 0.9:
                    suggestions.append(col)
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(suggestions))
    
    def _suggest_problem_type(self, df: pd.DataFrame, target_suggestions: List[str]) -> str:
        """Suggest problem type based on target columns"""
        if not target_suggestions:
            return ProblemType.AUTO_DETECT.value
        
        # Check the first suggested target column
        target_col = target_suggestions[0]
        target_data = df[target_col]
        
        # Check if the column is numeric
        if not pd.api.types.is_numeric_dtype(target_data):
            return ProblemType.CLASSIFICATION.value
        
        # Special case: Binary classification detection (0/1 or True/False)
        unique_values = target_data.dropna().unique()
        if len(unique_values) == 2:
            # Check if values are 0/1 or similar binary pattern
            sorted_values = sorted(unique_values)
            if (sorted_values == [0, 1] or 
                sorted_values == [0.0, 1.0] or
                set(unique_values) == {True, False} or
                set(str(v).lower() for v in unique_values) == {'true', 'false'} or
                set(str(v).lower() for v in unique_values) == {'yes', 'no'}):
                return ProblemType.CLASSIFICATION.value
        
        # For numeric columns, use more sophisticated logic
        unique_count = target_data.nunique()
        total_values = len(df)
        unique_ratio = unique_count / total_values
        
        # If very few unique values relative to dataset size, likely classification
        # Conservative threshold: only classify as classification if <= 10 unique values AND < 5% ratio
        if unique_count <= 10 and unique_ratio < 0.05:
            return ProblemType.CLASSIFICATION.value
        
        # Special check: if all values are integers but there are many unique values,
        # and the range is large, it's likely regression (like revenue, prices, etc.)
        if pd.api.types.is_integer_dtype(target_data):
            value_range = target_data.max() - target_data.min()
            # If range is large (> 1000) and many unique values (> 20), likely regression
            if value_range > 1000 and unique_count > 20:
                return ProblemType.REGRESSION.value
        
        # Default logic: if many unique values or high ratio, it's regression
        if unique_count > 20 or unique_ratio > 0.1:
            return ProblemType.REGRESSION.value
        else:
            return ProblemType.CLASSIFICATION.value
    
    def load_dataset(self, upload_id: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Load dataset by upload ID"""
        info_path = self.upload_dir / f"{upload_id}_info.json"
        
        if not info_path.exists():
            raise ValueError(f"Upload ID {upload_id} not found")
        
        with open(info_path, "r") as f:
            info = json.load(f)
        
        file_path = Path(info["file_path"])
        df = self._read_file(file_path)
        
        return df, info["dataset_info"]