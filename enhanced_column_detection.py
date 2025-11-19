"""
Enhanced column type detection for mostly-numeric columns with text errors
"""

def _infer_column_type_enhanced(self, series: pd.Series) -> str:
    """Enhanced version that handles mostly-numeric columns properly"""
    
    # Check if datetime
    if pd.api.types.is_datetime64_any_dtype(series):
        return ColumnType.DATETIME.value
    
    # Check if numeric
    if pd.api.types.is_numeric_dtype(series):
        return ColumnType.NUMERICAL.value
    
    # Check if it's mostly numeric but stored as string
    if series.dtype == 'object':
        # Try to convert to numeric
        numeric_series = pd.to_numeric(series, errors='coerce')
        if not numeric_series.isna().all():
            # Count how many values successfully convert
            total_non_null = series.notna().sum()
            converted_non_null = numeric_series.notna().sum()
            conversion_rate = converted_non_null / total_non_null if total_non_null > 0 else 0
            
            # NEW: If >80% converts to numeric, treat as "mostly_numeric" 
            if conversion_rate >= 0.8:
                return "mostly_numeric"  # New type for preprocessing
            
            # If ALL convert (100%), treat as pure numerical
            if conversion_rate == 1.0:
                return ColumnType.NUMERICAL.value
    
    # Check if categorical (low cardinality relative to dataset size)
    unique_ratio = series.nunique() / len(series)
    if unique_ratio < 0.1 or series.nunique() < 20:
        return ColumnType.CATEGORICAL.value
    
    # Default to text
    return ColumnType.TEXT.value