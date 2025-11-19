from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import json

class DataQualityReporter:
    """Generate comprehensive data quality reports"""
    
    def __init__(self):
        self.report_data = {}
        
    def generate_data_summary(self, df: pd.DataFrame, column_types: Dict[str, str], 
                            preprocessing_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive data quality summary"""
        
        summary = {
            "dataset_overview": self._get_dataset_overview(df),
            "column_analysis": self._analyze_columns(df, column_types),
            "missing_data_analysis": self._analyze_missing_data(df),
            "outlier_analysis": self._analyze_outliers(preprocessing_info),
            "preprocessing_actions": self._get_preprocessing_actions(preprocessing_info),
            "data_quality_score": self._calculate_quality_score(df),
            "recommendations": self._generate_recommendations(df, preprocessing_info),
            "generated_at": datetime.now().isoformat()
        }
        
        return summary
    
    def _get_dataset_overview(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic dataset statistics"""
        return {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
            "duplicate_rows": df.duplicated().sum(),
            "duplicate_percentage": round((df.duplicated().sum() / len(df)) * 100, 2)
        }
    
    def _analyze_columns(self, df: pd.DataFrame, column_types: Dict[str, str]) -> List[Dict[str, Any]]:
        """Analyze each column in detail"""
        column_analysis = []
        
        for col in df.columns:
            col_data = df[col]
            
            analysis = {
                "column_name": col,
                "detected_type": column_types.get(col, "unknown"),
                "pandas_dtype": str(col_data.dtype),
                "unique_values": col_data.nunique(),
                "unique_percentage": round((col_data.nunique() / len(col_data)) * 100, 2),
                "missing_count": col_data.isnull().sum(),
                "missing_percentage": round((col_data.isnull().sum() / len(col_data)) * 100, 2),
                "cardinality": self._get_cardinality_category(col_data.nunique()),
                "sample_values": self._get_sample_values(col_data),
                "data_quality_issues": self._identify_quality_issues(col_data)
            }
            
            # Add type-specific analysis
            if pd.api.types.is_numeric_dtype(col_data):
                analysis.update(self._analyze_numeric_column(col_data))
            elif pd.api.types.is_object_dtype(col_data):
                analysis.update(self._analyze_text_column(col_data))
                
            column_analysis.append(analysis)
            
        return column_analysis
    
    def _analyze_missing_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive missing data analysis"""
        missing_summary = df.isnull().sum()
        missing_percentage = (missing_summary / len(df)) * 100
        
        return {
            "total_missing_values": missing_summary.sum(),
            "columns_with_missing": (missing_summary > 0).sum(),
            "missing_data_percentage": round((missing_summary.sum() / df.size) * 100, 2),
            "columns_missing_analysis": [
                {
                    "column": col,
                    "missing_count": int(missing_summary[col]),
                    "missing_percentage": round(missing_percentage[col], 2),
                    "severity": self._get_missing_severity(missing_percentage[col])
                }
                for col in df.columns if missing_summary[col] > 0
            ],
            "missing_pattern": self._analyze_missing_patterns(df)
        }
    
    def _analyze_outliers(self, preprocessing_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze outlier detection results"""
        outlier_info = preprocessing_info.get('outlier_info', {})
        
        if not outlier_info:
            return {"outliers_detected": False, "summary": "No outlier analysis performed"}
        
        outlier_analysis = {
            "outliers_detected": True,
            "columns_with_outliers": len(outlier_info),
            "outlier_details": []
        }
        
        for col, info in outlier_info.items():
            outlier_analysis["outlier_details"].append({
                "column": col,
                "outliers_count": info.get('outliers_detected', 0),
                "outlier_percentage": round(info.get('outlier_percentage', 0), 2),
                "severity": self._get_outlier_severity(info.get('outlier_percentage', 0))
            })
            
        return outlier_analysis
    
    def _get_preprocessing_actions(self, preprocessing_info: Dict[str, Any]) -> Dict[str, List[str]]:
        """Extract all preprocessing actions performed"""
        actions = {
            "outlier_treatment": [],
            "scaling_applied": [],
            "encoding_applied": [],
            "feature_engineering": [],
            "feature_selection": [],
            "data_cleaning": []
        }
        
        # Outlier treatment actions
        outlier_info = preprocessing_info.get('outlier_info', {})
        if outlier_info:
            for col, info in outlier_info.items():
                if info.get('outliers_detected', 0) > 0:
                    actions["outlier_treatment"].append(
                        f"Applied IQR outlier clipping to '{col}' ({info.get('outliers_detected', 0)} outliers)"
                    )
        
        # Scaling actions
        scaling_info = preprocessing_info.get('scaling_info', {})
        if scaling_info:
            for col, method in scaling_info.items():
                actions["scaling_applied"].append(f"Applied {method} scaling to '{col}'")
        
        # Encoding actions
        encoding_info = preprocessing_info.get('encoding_info', {})
        if encoding_info:
            for col, method in encoding_info.items():
                actions["encoding_applied"].append(f"Applied {method} encoding to '{col}'")
        
        # Feature engineering
        engineered_features = preprocessing_info.get('engineered_features', [])
        if engineered_features:
            actions["feature_engineering"].append(f"Created {len(engineered_features)} new features")
            actions["feature_engineering"].extend([f"Generated feature: '{feat}'" for feat in engineered_features[:5]])
            if len(engineered_features) > 5:
                actions["feature_engineering"].append(f"... and {len(engineered_features) - 5} more")
        
        # Feature selection
        feature_selection_info = preprocessing_info.get('feature_selection_info', {})
        if feature_selection_info:
            removed_constant = feature_selection_info.get('removed_constant', [])
            removed_correlated = feature_selection_info.get('removed_correlated', [])
            removed_by_model = feature_selection_info.get('removed_by_model', [])
            
            if removed_constant:
                actions["feature_selection"].append(f"Removed {len(removed_constant)} constant features")
            if removed_correlated:
                actions["feature_selection"].append(f"Removed {len(removed_correlated)} highly correlated features")
            if removed_by_model:
                actions["feature_selection"].append(f"Removed {len(removed_by_model)} low-importance features")
        
        return actions
    
    def _calculate_quality_score(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate overall data quality score"""
        missing_percentage = (df.isnull().sum().sum() / df.size) * 100
        duplicate_percentage = (df.duplicated().sum() / len(df)) * 100
        
        # Quality score calculation (0-100)
        quality_score = 100
        quality_score -= min(missing_percentage * 2, 50)  # Penalize missing data
        quality_score -= min(duplicate_percentage * 3, 30)  # Penalize duplicates
        
        return {
            "overall_score": max(0, round(quality_score, 1)),
            "grade": self._get_quality_grade(quality_score),
            "factors": {
                "missing_data_impact": -min(missing_percentage * 2, 50),
                "duplicate_impact": -min(duplicate_percentage * 3, 30)
            }
        }
    
    def _generate_recommendations(self, df: pd.DataFrame, preprocessing_info: Dict[str, Any]) -> List[str]:
        """Generate data quality recommendations"""
        recommendations = []
        
        # Missing data recommendations
        missing_percentage = (df.isnull().sum() / len(df)) * 100
        high_missing_cols = missing_percentage[missing_percentage > 50].index.tolist()
        if high_missing_cols:
            recommendations.append(f"Consider removing columns with >50% missing data: {', '.join(high_missing_cols)}")
        
        # Duplicate data recommendation
        if df.duplicated().sum() > 0:
            recommendations.append(f"Remove {df.duplicated().sum()} duplicate rows to improve data quality")
        
        # High cardinality recommendations
        for col in df.columns:
            if df[col].nunique() > len(df) * 0.8:
                recommendations.append(f"Column '{col}' has very high cardinality - consider feature engineering")
        
        # Memory optimization
        if df.memory_usage(deep=True).sum() / 1024**2 > 100:  # >100MB
            recommendations.append("Consider data type optimization for large dataset")
        
        return recommendations
    
    # Helper methods
    def _get_cardinality_category(self, unique_count: int) -> str:
        """Categorize cardinality"""
        if unique_count <= 10:
            return "Low"
        elif unique_count <= 1000:
            return "Medium"
        else:
            return "High"
    
    def _get_sample_values(self, series: pd.Series, n: int = 5) -> List[str]:
        """Get sample values from series"""
        return [str(val) for val in series.dropna().unique()[:n]]
    
    def _identify_quality_issues(self, series: pd.Series) -> List[str]:
        """Identify data quality issues"""
        issues = []
        
        missing_pct = (series.isnull().sum() / len(series)) * 100
        if missing_pct > 50:
            issues.append("High missing data (>50%)")
        elif missing_pct > 20:
            issues.append("Moderate missing data (>20%)")
            
        if series.nunique() == 1:
            issues.append("Constant values")
        elif series.nunique() == len(series):
            issues.append("All unique values")
            
        return issues
    
    def _analyze_numeric_column(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze numeric column"""
        return {
            "min_value": float(series.min()) if not series.isnull().all() else None,
            "max_value": float(series.max()) if not series.isnull().all() else None,
            "mean_value": float(series.mean()) if not series.isnull().all() else None,
            "std_value": float(series.std()) if not series.isnull().all() else None,
            "zeros_count": (series == 0).sum(),
            "negative_count": (series < 0).sum()
        }
    
    def _analyze_text_column(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze text column"""
        non_null = series.dropna()
        return {
            "avg_length": round(non_null.astype(str).str.len().mean(), 1) if len(non_null) > 0 else 0,
            "max_length": non_null.astype(str).str.len().max() if len(non_null) > 0 else 0,
            "empty_strings": (series == "").sum(),
            "whitespace_only": series.astype(str).str.strip().eq("").sum()
        }
    
    def _get_missing_severity(self, percentage: float) -> str:
        """Get severity level for missing data"""
        if percentage > 50:
            return "Critical"
        elif percentage > 20:
            return "High"
        elif percentage > 5:
            return "Medium"
        else:
            return "Low"
    
    def _get_outlier_severity(self, percentage: float) -> str:
        """Get severity level for outliers"""
        if percentage > 15:
            return "High"
        elif percentage > 5:
            return "Medium"
        else:
            return "Low"
    
    def _analyze_missing_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze patterns in missing data"""
        # Simple pattern analysis
        total_missing = df.isnull().sum().sum()
        if total_missing == 0:
            return {"pattern": "No missing data"}
        
        # Check if missing data is random or systematic
        missing_per_row = df.isnull().sum(axis=1)
        rows_with_missing = (missing_per_row > 0).sum()
        
        return {
            "pattern": "Systematic" if rows_with_missing < len(df) * 0.5 else "Random",
            "rows_affected": int(rows_with_missing),
            "rows_affected_percentage": round((rows_with_missing / len(df)) * 100, 2)
        }
    
    def _get_quality_grade(self, score: float) -> str:
        """Convert quality score to grade"""
        if score >= 90:
            return "Excellent"
        elif score >= 80:
            return "Good"
        elif score >= 70:
            return "Fair"
        elif score >= 60:
            return "Poor"
        else:
            return "Critical"