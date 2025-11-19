#!/usr/bin/env python3
"""
Analysis of different approaches to handle columns with mostly numeric data but some text errors
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import requests

def analyze_preprocessing_approaches():
    """Compare different preprocessing approaches for mixed numeric/text columns"""
    
    print("🔍 ANALYZING PREPROCESSING APPROACHES FOR MIXED NUMERIC/TEXT COLUMNS")
    print("=" * 80)
    
    # Create sample data - mostly numeric with some text errors
    np.random.seed(42)
    data = []
    for i in range(100):
        if i % 20 == 0:  # 5% bad data
            data.append("error_data")
        else:  # 95% good numeric data
            data.append(round(np.random.normal(100, 15), 2))
    
    print(f"Sample column: 95% numeric, 5% text errors")
    print(f"First 10 values: {data[:10]}")
    
    # Approach 1: Current platform approach - treat as categorical
    print(f"\n📊 APPROACH 1: TREAT AS CATEGORICAL (Current Platform)")
    print("-" * 50)
    
    # Convert everything to string first (this is what happens in real preprocessing)
    categorical_data = pd.Series([str(x) for x in data])
    categorical_imputer = SimpleImputer(strategy='constant', fill_value='missing')
    categorical_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    # This creates many one-hot columns (one for each unique value)
    categorical_reshaped = categorical_data.values.reshape(-1, 1)
    categorical_imputed = categorical_imputer.fit_transform(categorical_reshaped)
    categorical_encoded = categorical_encoder.fit_transform(categorical_imputed)
    
    print(f"Result shape: {categorical_encoded.shape} (creates {categorical_encoded.shape[1]} one-hot columns)")
    print(f"Memory usage: High (one column per unique value)")
    print(f"Information preserved: All original values kept as separate categories")
    print(f"ML performance: Poor (high dimensionality, treats '100.5' and '101.2' as completely different)")
    print(f"Problem: {categorical_encoded.shape[1]} features for what should be 1 numeric feature!")
    
    # Approach 2: Clean and treat as numeric
    print(f"\n🧹 APPROACH 2: CLEAN TEXT ERRORS & TREAT AS NUMERIC")
    print("-" * 50)
    
    numeric_series = pd.to_numeric(pd.Series(data), errors='coerce')  # Convert text to NaN
    numeric_imputer = SimpleImputer(strategy='median')
    numeric_scaler = StandardScaler()
    
    # Clean approach
    numeric_cleaned = numeric_imputer.fit_transform(numeric_series.values.reshape(-1, 1))
    numeric_scaled = numeric_scaler.fit_transform(numeric_cleaned)
    
    print(f"Result shape: {numeric_scaled.shape} (single numeric column)")
    print(f"Memory usage: Low")
    print(f"Information preserved: Numeric relationships maintained, text errors replaced with median")
    print(f"ML performance: Good (preserves numeric relationships)")
    
    # Approach 3: Hybrid approach - separate numeric and error indicator
    print(f"\n🔀 APPROACH 3: HYBRID - NUMERIC + ERROR INDICATOR")
    print("-" * 50)
    
    # Create numeric column with imputation
    numeric_clean = pd.to_numeric(pd.Series(data), errors='coerce')
    numeric_imputed = SimpleImputer(strategy='median').fit_transform(numeric_clean.values.reshape(-1, 1))
    
    # Create binary indicator for errors
    error_indicator = pd.Series(data).apply(lambda x: 1 if isinstance(x, str) else 0)
    
    hybrid_result = np.column_stack([numeric_imputed, error_indicator])
    
    print(f"Result shape: {hybrid_result.shape} (numeric column + error flag)")
    print(f"Memory usage: Low")
    print(f"Information preserved: Numeric relationships + knowledge of which values were errors")
    print(f"ML performance: Best (numeric relationships + error pattern information)")
    
    return data

def demonstrate_ml_impact():
    """Show how different preprocessing affects ML performance"""
    
    print(f"\n🤖 ML PERFORMANCE COMPARISON")
    print("=" * 50)
    
    # Create synthetic dataset with target correlation
    np.random.seed(42)
    n_samples = 1000
    
    # Create feature with mostly numeric values, some text errors
    X_clean = np.random.normal(100, 15, n_samples)
    y = X_clean * 2 + np.random.normal(0, 5, n_samples)  # Target correlated with clean values
    
    # Introduce text errors (5%)
    X_with_errors = []
    for i, val in enumerate(X_clean):
        if i % 20 == 0:  # 5% errors
            X_with_errors.append(f"error_{i}")
        else:
            X_with_errors.append(val)
    
    # Test different approaches
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        np.array(X_with_errors), y, test_size=0.2, random_state=42
    )
    
    # Approach 1: One-hot encoding (current platform)
    print("Approach 1 (One-hot): Too many features to test effectively")
    
    # Approach 2: Clean numeric
    X_train_numeric = pd.to_numeric(X_train, errors='coerce')
    X_test_numeric = pd.to_numeric(X_test, errors='coerce')
    
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    
    X_train_clean = scaler.fit_transform(imputer.fit_transform(X_train_numeric.reshape(-1, 1)))
    X_test_clean = scaler.transform(imputer.transform(X_test_numeric.reshape(-1, 1)))
    
    model = LinearRegression()
    model.fit(X_train_clean, y_train)
    y_pred_clean = model.predict(X_test_clean)
    r2_clean = r2_score(y_test, y_pred_clean)
    
    print(f"Approach 2 (Clean numeric): R² = {r2_clean:.4f}")
    
    # Approach 3: Hybrid
    X_train_hybrid_numeric = imputer.fit_transform(pd.to_numeric(X_train, errors='coerce').reshape(-1, 1))
    X_train_hybrid_errors = np.array([1 if isinstance(x, str) else 0 for x in X_train]).reshape(-1, 1)
    X_train_hybrid = np.column_stack([
        scaler.fit_transform(X_train_hybrid_numeric),
        X_train_hybrid_errors
    ])
    
    X_test_hybrid_numeric = imputer.transform(pd.to_numeric(X_test, errors='coerce').reshape(-1, 1))
    X_test_hybrid_errors = np.array([1 if isinstance(x, str) else 0 for x in X_test]).reshape(-1, 1)
    X_test_hybrid = np.column_stack([
        scaler.transform(X_test_hybrid_numeric),
        X_test_hybrid_errors
    ])
    
    model_hybrid = LinearRegression()
    model_hybrid.fit(X_train_hybrid, y_train)
    y_pred_hybrid = model_hybrid.predict(X_test_hybrid)
    r2_hybrid = r2_score(y_test, y_pred_hybrid)
    
    print(f"Approach 3 (Hybrid): R² = {r2_hybrid:.4f}")
    
    return r2_clean, r2_hybrid

def get_current_platform_approach():
    """Show what the current platform does"""
    
    print(f"\n🔧 CURRENT PLATFORM APPROACH")
    print("=" * 50)
    print("When a column has 95% numeric + 5% text:")
    print("✓ Detects mixed content")
    print("✓ Classifies as 'text' or 'categorical'")
    print("✓ Uses OneHotEncoder preprocessing")
    print("✗ Creates high-dimensional sparse representation")
    print("✗ Loses numeric relationships between values")
    print("✗ Poor ML performance for regression tasks")

if __name__ == "__main__":
    data = analyze_preprocessing_approaches()
    r2_clean, r2_hybrid = demonstrate_ml_impact()
    get_current_platform_approach()
    
    print(f"\n🎯 RECOMMENDATIONS")
    print("=" * 50)
    print("For columns with >80% numeric data but some text errors:")
    print("1. DETECT: Identify as 'mostly_numeric_with_errors'")
    print("2. CLEAN: Convert text to NaN, impute with median/mean")
    print("3. OPTIONAL: Add error indicator feature")
    print("4. SCALE: Apply standard scaling")
    print("\nThis preserves numeric relationships while handling errors gracefully.")