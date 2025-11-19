import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder, RobustScaler
from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold, SelectKBest, SelectFromModel, f_classif, f_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from scipy import stats
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from app.services.data_quality_reporter import DataQualityReporter
import plotly.express as px
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder
import json
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import base64
import io

from app.models.schemas import ColumnType, ProblemType, EDAResponse

class NumericCleaner(BaseEstimator, TransformerMixin):
    """Clean mostly-numeric columns by converting text to NaN"""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_clean = X.copy()
        for col in X_clean.columns:
            # Convert to numeric, coercing text to NaN
            X_clean[col] = pd.to_numeric(X_clean[col], errors='coerce')
        return X_clean

class OutlierDetector(BaseEstimator, TransformerMixin):
    """Detect and treat outliers using IQR and Z-score methods"""
    def __init__(self, method='iqr', threshold=1.5, z_threshold=3.0, clip_outliers=True):
        self.method = method  # 'iqr', 'zscore', or 'both'
        self.threshold = threshold  # IQR multiplier (1.5 standard)
        self.z_threshold = z_threshold  # Z-score threshold (3.0 standard)
        self.clip_outliers = clip_outliers  # If True, clip; if False, remove
        self.bounds_ = {}
        self.outlier_info_ = {}
        self.feature_names_ = []
        
    def _ensure_dataframe(self, X):
        """Convert numpy array to DataFrame if needed"""
        if isinstance(X, np.ndarray):
            if len(self.feature_names_) == X.shape[1]:
                return pd.DataFrame(X, columns=self.feature_names_)
            else:
                return pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        return X
        
    def fit(self, X, y=None):
        """Fit outlier detector and calculate bounds"""
        # Convert to DataFrame if needed and store feature names
        X_df = self._ensure_dataframe(X)
        self.feature_names_ = list(X_df.columns)
        
        self.bounds_ = {}
        self.outlier_info_ = {}
        
        for col in X_df.columns:
            if pd.api.types.is_numeric_dtype(X_df[col]):
                outlier_count = 0
                
                if self.method in ['iqr', 'both']:
                    # IQR method
                    Q1 = X_df[col].quantile(0.25)
                    Q3 = X_df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - self.threshold * IQR
                    upper_bound = Q3 + self.threshold * IQR
                    
                    iqr_outliers = ((X_df[col] < lower_bound) | (X_df[col] > upper_bound)).sum()
                    self.bounds_[f'{col}_iqr'] = (lower_bound, upper_bound)
                    outlier_count += iqr_outliers
                
                if self.method in ['zscore', 'both']:
                    # Z-score method (for normally distributed data)
                    z_scores = np.abs(stats.zscore(X[col].dropna()))
                    zscore_outliers = (z_scores > self.z_threshold).sum()
                    
                    # Calculate bounds based on mean ± z_threshold * std
                    mean_val = X[col].mean()
                    std_val = X[col].std()
                    lower_bound = mean_val - self.z_threshold * std_val
                    upper_bound = mean_val + self.z_threshold * std_val
                    
                    self.bounds_[f'{col}_zscore'] = (lower_bound, upper_bound)
                    outlier_count += zscore_outliers
                
                # Store outlier information
                total_outliers = min(outlier_count, len(X[col]))  # Avoid double counting if both methods used
                self.outlier_info_[col] = {
                    'outliers_detected': total_outliers,
                    'outlier_percentage': (total_outliers / len(X[col])) * 100 if len(X[col]) > 0 else 0
                }
        
        return self
        
    def transform(self, X):
        """Transform data by treating outliers"""
        # Convert to DataFrame if needed
        X_df = self._ensure_dataframe(X)
        X_treated = X_df.copy()
        
        for col in X_df.columns:
            if pd.api.types.is_numeric_dtype(X_df[col]) and col in self.outlier_info_:
                
                if self.method in ['iqr', 'both'] and f'{col}_iqr' in self.bounds_:
                    lower_bound, upper_bound = self.bounds_[f'{col}_iqr']
                    if self.clip_outliers:
                        # Clip outliers to boundary values
                        X_treated[col] = np.clip(X_treated[col], lower_bound, upper_bound)
                    else:
                        # Remove outliers (set to NaN for later imputation)
                        mask = (X_treated[col] < lower_bound) | (X_treated[col] > upper_bound)
                        X_treated.loc[mask, col] = np.nan
                
                elif self.method in ['zscore', 'both'] and f'{col}_zscore' in self.bounds_:
                    lower_bound, upper_bound = self.bounds_[f'{col}_zscore']
                    if self.clip_outliers:
                        X_treated[col] = np.clip(X_treated[col], lower_bound, upper_bound)
                    else:
                        mask = (X_treated[col] < lower_bound) | (X_treated[col] > upper_bound)
                        X_treated.loc[mask, col] = np.nan
        
        # Return numpy array for sklearn compatibility
        return X_treated.values
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names for sklearn compatibility"""
        return input_features or self.feature_names_
    
    def get_outlier_summary(self):
        """Get summary of detected outliers"""
        return self.outlier_info_

class SmartScaler(BaseEstimator, TransformerMixin):
    """Intelligently choose scaling method based on data distribution"""
    def __init__(self, auto_detect=True):
        self.auto_detect = auto_detect
        self.scalers_ = {}
        self.scaling_methods_ = {}
        self.feature_names_ = []
        
    def _ensure_dataframe(self, X):
        """Convert numpy array to DataFrame if needed"""
        if isinstance(X, np.ndarray):
            if len(self.feature_names_) == X.shape[1]:
                return pd.DataFrame(X, columns=self.feature_names_)
            else:
                return pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        return X
        
    def _detect_best_scaler(self, series):
        """Detect the best scaling method for a series"""
        # Remove NaN values for analysis
        clean_series = series.dropna()
        
        if len(clean_series) < 10:  # Not enough data
            return StandardScaler()
        
        # Check for normality (Shapiro-Wilk test for small samples)
        if len(clean_series) <= 5000:
            try:
                _, p_value = stats.shapiro(clean_series.sample(min(5000, len(clean_series))))
                is_normal = p_value > 0.05
            except:
                is_normal = False
        else:
            # Use skewness for large samples
            skewness = abs(stats.skew(clean_series))
            is_normal = skewness < 1.0
        
        # Check for outliers (using IQR method)
        Q1 = clean_series.quantile(0.25)
        Q3 = clean_series.quantile(0.75)
        IQR = Q3 - Q1
        outlier_count = ((clean_series < (Q1 - 1.5 * IQR)) | (clean_series > (Q3 + 1.5 * IQR))).sum()
        outlier_percentage = (outlier_count / len(clean_series)) * 100
        
        # Check if data is bounded (e.g., percentages, probabilities)
        is_bounded = clean_series.min() >= 0 and clean_series.max() <= 1
        has_small_range = (clean_series.max() - clean_series.min()) < 10
        
        # Decision logic
        if outlier_percentage > 10:  # High outlier percentage
            return RobustScaler()
        elif is_bounded or has_small_range:  # Bounded data
            return MinMaxScaler()
        elif is_normal:  # Normal distribution
            return StandardScaler()
        else:  # Skewed or unknown distribution
            return RobustScaler()
    
    def fit(self, X, y=None):
        """Fit appropriate scalers for each column"""
        # Convert to DataFrame if needed and store feature names
        X_df = self._ensure_dataframe(X)
        self.feature_names_ = list(X_df.columns)
        
        for col in X_df.columns:
            if pd.api.types.is_numeric_dtype(X_df[col]):
                if self.auto_detect:
                    scaler = self._detect_best_scaler(X_df[col])
                else:
                    scaler = StandardScaler()  # Default
                
                # Fit the scaler
                scaler.fit(X_df[[col]])
                self.scalers_[col] = scaler
                self.scaling_methods_[col] = type(scaler).__name__
        
        return self
    
    def transform(self, X):
        """Transform using fitted scalers"""
        # Convert to DataFrame if needed
        X_df = self._ensure_dataframe(X)
        X_scaled = X_df.copy()
        
        for col in X_df.columns:
            if col in self.scalers_:
                X_scaled[col] = self.scalers_[col].transform(X_df[[col]]).flatten()
        
        # Return numpy array for sklearn compatibility
        return X_scaled.values
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names for sklearn compatibility"""
        return input_features or self.feature_names_
    
    def get_scaling_info(self):
        """Get information about scaling methods used"""
        return self.scaling_methods_

class FeatureSelector(BaseEstimator, TransformerMixin):
    """Comprehensive feature selection with quality checks"""
    def __init__(self, problem_type='classification', max_features=None, correlation_threshold=0.9):
        self.problem_type = problem_type
        self.max_features = max_features
        self.correlation_threshold = correlation_threshold
        self.selected_features_ = None
        self.feature_selection_info_ = {}
        self.feature_names_ = []
        
    def _ensure_dataframe(self, X):
        """Convert numpy array to DataFrame if needed"""
        if isinstance(X, np.ndarray):
            if len(self.feature_names_) == X.shape[1]:
                return pd.DataFrame(X, columns=self.feature_names_)
            else:
                return pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        return X
        
    def fit(self, X, y=None):
        """Fit feature selector with comprehensive quality checks"""
        # Convert to DataFrame if needed and store feature names
        X_df = self._ensure_dataframe(X)
        self.feature_names_ = list(X_df.columns)
        
        feature_info = {}
        features_to_keep = list(X_df.columns)
        
        # Step 1: Remove near-constant features (VarianceThreshold)
        variance_selector = VarianceThreshold(threshold=0.01)  # Remove features with <1% variance
        try:
            variance_selector.fit(X_df)
            constant_features = [col for i, col in enumerate(X_df.columns) 
                               if not variance_selector.get_support()[i]]
            if constant_features:
                feature_info['removed_constant'] = constant_features
                features_to_keep = [f for f in features_to_keep if f not in constant_features]
        except:
            feature_info['removed_constant'] = []
        
        if not features_to_keep:
            self.selected_features_ = []
            self.feature_selection_info_ = feature_info
            return self
            
        X_filtered = X[features_to_keep]
        
        # Step 2: Remove highly correlated features (correlation filter)
        if len(features_to_keep) > 1:
            try:
                # Calculate correlation matrix for numeric features only
                numeric_features = X_filtered.select_dtypes(include=[np.number]).columns
                if len(numeric_features) > 1:
                    corr_matrix = X_filtered[numeric_features].corr().abs()
                    
                    # Find highly correlated feature pairs
                    high_corr_pairs = []
                    for i in range(len(corr_matrix.columns)):
                        for j in range(i+1, len(corr_matrix.columns)):
                            if corr_matrix.iloc[i, j] > self.correlation_threshold:
                                high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
                    
                    # Remove one feature from each highly correlated pair
                    correlated_features_to_remove = []
                    for feat1, feat2 in high_corr_pairs:
                        if feat2 not in correlated_features_to_remove:
                            correlated_features_to_remove.append(feat2)
                    
                    if correlated_features_to_remove:
                        feature_info['removed_correlated'] = correlated_features_to_remove
                        features_to_keep = [f for f in features_to_keep if f not in correlated_features_to_remove]
                    else:
                        feature_info['removed_correlated'] = []
                else:
                    feature_info['removed_correlated'] = []
            except:
                feature_info['removed_correlated'] = []
        
        if not features_to_keep:
            self.selected_features_ = []
            self.feature_selection_info_ = feature_info
            return self
            
        X_filtered = X[features_to_keep]
        
        # Step 3: Model-based feature selection (if target provided and max_features specified)
        if y is not None and self.max_features and len(features_to_keep) > self.max_features:
            try:
                # Use appropriate model based on problem type
                if self.problem_type == 'classification':
                    model = RandomForestClassifier(n_estimators=50, random_state=42)
                else:
                    model = RandomForestRegressor(n_estimators=50, random_state=42)
                
                # Ensure we have numeric data for model-based selection
                X_numeric = X_filtered.select_dtypes(include=[np.number])
                if len(X_numeric.columns) >= self.max_features:
                    selector = SelectFromModel(model, max_features=self.max_features)
                    selector.fit(X_numeric, y)
                    
                    selected_numeric_features = [col for i, col in enumerate(X_numeric.columns) 
                                               if selector.get_support()[i]]
                    
                    # Keep selected numeric features + all categorical features
                    categorical_features = [col for col in features_to_keep if col not in X_numeric.columns]
                    features_to_keep = selected_numeric_features + categorical_features
                    
                    removed_by_model = [col for col in X_numeric.columns if col not in selected_numeric_features]
                    feature_info['removed_by_model'] = removed_by_model
                else:
                    feature_info['removed_by_model'] = []
            except:
                feature_info['removed_by_model'] = []
        
        self.selected_features_ = features_to_keep
        self.feature_selection_info_ = feature_info
        return self
        
    def transform(self, X):
        """Transform data to selected features only"""
        if self.selected_features_ is None:
            raise ValueError("FeatureSelector not fitted yet")
        
        # Convert to DataFrame if needed
        X_df = self._ensure_dataframe(X)
        
        if not self.selected_features_:
            # Return empty array
            return np.array([]).reshape(len(X_df), 0)
            
        # Return numpy array for sklearn compatibility
        return X_df[self.selected_features_].values
    
    def get_feature_names_out(self, input_features=None):
        """Get selected feature names"""
        return self.selected_features_ if self.selected_features_ else []

class SmartCategoricalEncoder(BaseEstimator, TransformerMixin):
    """Intelligently encode categorical variables based on cardinality"""
    def __init__(self, low_cardinality_threshold=10, high_cardinality_threshold=1000):
        self.low_cardinality_threshold = low_cardinality_threshold
        self.high_cardinality_threshold = high_cardinality_threshold
        self.encoders_ = {}
        self.encoding_methods_ = {}
        self.feature_names_ = []
        self.frequency_maps_ = {}  # Store frequency mappings
        
    def _ensure_dataframe(self, X):
        """Convert numpy array to DataFrame if needed"""
        if isinstance(X, np.ndarray):
            if len(self.feature_names_) == X.shape[1]:
                return pd.DataFrame(X, columns=self.feature_names_)
            else:
                return pd.DataFrame(X, columns=[f'cat_feature_{i}' for i in range(X.shape[1])])
        return X
        
    def fit(self, X, y=None):
        """Fit appropriate encoders based on cardinality"""
        # Convert to DataFrame if needed and store feature names
        X_df = self._ensure_dataframe(X)
        self.feature_names_ = list(X_df.columns)
        
        for col in X_df.columns:
            unique_count = X_df[col].nunique()
            
            if unique_count <= self.low_cardinality_threshold:
                # Low cardinality: Use OneHot Encoding
                encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')
                self.encoding_methods_[col] = f'OneHot (cardinality: {unique_count})'
                
            elif unique_count <= self.high_cardinality_threshold:
                # Medium cardinality: Use Target/Mean Encoding (simplified with frequency encoding)
                # Note: For proper target encoding, we'd need the target variable
                encoder = 'frequency'  # Will implement frequency encoding
                self.encoding_methods_[col] = f'Frequency (cardinality: {unique_count})'
                
            else:
                # High cardinality: Use Hash Encoding
                encoder = FeatureHasher(n_features=min(32, unique_count//10), input_type='string')
                self.encoding_methods_[col] = f'Hash (cardinality: {unique_count})'
            
            if encoder != 'frequency':
                try:
                    if hasattr(encoder, 'fit'):
                        encoder.fit(X_df[[col]].astype(str))
                    self.encoders_[col] = encoder
                except Exception as e:
                    # Fallback to frequency encoding
                    self.encoders_[col] = 'frequency'
                    self.encoding_methods_[col] = f'Frequency-fallback (cardinality: {unique_count})'
                    # Store frequency mapping
                    self.frequency_maps_[col] = X_df[col].value_counts().to_dict()
            else:
                self.encoders_[col] = 'frequency'
                # Store frequency mapping for frequency encoding
                self.frequency_maps_[col] = X_df[col].value_counts().to_dict()
        
        return self
    
    def transform(self, X):
        """Transform using appropriate encoders"""
        # Convert to DataFrame if needed
        X_df = self._ensure_dataframe(X)
        encoded_dfs = []
        
        for col in X_df.columns:
            if col in self.encoders_:
                encoder = self.encoders_[col]
                
                if encoder == 'frequency':
                    # Use stored frequency mapping
                    if col in self.frequency_maps_:
                        freq_map = self.frequency_maps_[col]
                        encoded_col = X_df[col].map(freq_map).fillna(0)
                        encoded_df = pd.DataFrame({f'{col}_freq': encoded_col})
                    
                elif isinstance(encoder, OneHotEncoder):
                    # OneHot encoding
                    encoded_array = encoder.transform(X_df[[col]].astype(str))
                    feature_names = [f'{col}_{cat}' for cat in encoder.categories_[0][1:]]  # Skip first due to drop='first'
                    encoded_df = pd.DataFrame(encoded_array, columns=feature_names, index=X_df.index)
                    
                elif isinstance(encoder, FeatureHasher):
                    # Hash encoding
                    encoded_array = encoder.transform(X_df[col].astype(str)).toarray()
                    feature_names = [f'{col}_hash_{i}' for i in range(encoded_array.shape[1])]
                    encoded_df = pd.DataFrame(encoded_array, columns=feature_names, index=X_df.index)
                
                else:
                    # Fallback: use stored frequency encoding if available
                    if col in self.frequency_maps_:
                        freq_map = self.frequency_maps_[col]
                        encoded_col = X_df[col].map(freq_map).fillna(0)
                        encoded_df = pd.DataFrame({f'{col}_freq': encoded_col})
                    else:
                        # Emergency fallback: create identity mapping
                        encoded_col = X_df[col].astype('category').cat.codes
                        encoded_df = pd.DataFrame({f'{col}_encoded': encoded_col})
                
                encoded_dfs.append(encoded_df)
        
        if encoded_dfs:
            result = pd.concat(encoded_dfs, axis=1)
            # Return numpy array for sklearn compatibility
            return result.values
        else:
            return np.array([]).reshape(len(X_df), 0)
    
    def get_encoding_info(self):
        """Get information about encoding methods used"""
        return self.encoding_methods_
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names for sklearn compatibility"""
        output_names = []
        feature_names = input_features or self.feature_names_
        
        for col in feature_names:
            if col in self.encoders_:
                encoder = self.encoders_[col]
                
                if encoder == 'frequency':
                    output_names.append(f'{col}_freq')
                elif isinstance(encoder, OneHotEncoder):
                    if hasattr(encoder, 'categories_'):
                        for cat in encoder.categories_[0][1:]:  # Skip first due to drop='first'
                            output_names.append(f'{col}_{cat}')
                elif isinstance(encoder, FeatureHasher):
                    n_features = encoder.n_features
                    for i in range(n_features):
                        output_names.append(f'{col}_hash_{i}')
                        
        return output_names

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Automatic feature engineering for common patterns"""
    def __init__(self, create_datetime_features=True, create_interaction_terms=False, max_interactions=5):
        self.create_datetime_features = create_datetime_features
        self.create_interaction_terms = create_interaction_terms
        self.max_interactions = max_interactions
        self.datetime_columns_ = []
        self.numeric_columns_ = []
        self.engineered_features_ = []
        self.feature_names_ = []
        
    def _ensure_dataframe(self, X):
        """Convert numpy array to DataFrame if needed"""
        if isinstance(X, np.ndarray):
            if len(self.feature_names_) == X.shape[1]:
                return pd.DataFrame(X, columns=self.feature_names_)
            else:
                return pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        return X
        
    def fit(self, X, y=None):
        """Identify columns for feature engineering"""
        # Convert to DataFrame if needed and store feature names
        X_df = self._ensure_dataframe(X)
        self.feature_names_ = list(X_df.columns)
        
        self.datetime_columns_ = []
        self.numeric_columns_ = []
        
        for col in X_df.columns:
            # Check for datetime columns
            if pd.api.types.is_datetime64_any_dtype(X_df[col]):
                self.datetime_columns_.append(col)
            elif self._is_datetime_string(X_df[col]):
                self.datetime_columns_.append(col)
            elif pd.api.types.is_numeric_dtype(X_df[col]):
                self.numeric_columns_.append(col)
        
        return self
    
    def _is_datetime_string(self, series):
        """Check if string column contains datetime-like values"""
        sample = series.dropna().head(100)
        datetime_count = 0
        
        for value in sample:
            try:
                pd.to_datetime(str(value))
                datetime_count += 1
            except:
                continue
        
        return datetime_count / len(sample) > 0.8 if len(sample) > 0 else False
    
    def transform(self, X):
        """Create engineered features"""
        # Convert to DataFrame if needed
        X_df = self._ensure_dataframe(X)
        X_engineered = X_df.copy()
        self.engineered_features_ = []
        
        # DateTime feature engineering
        if self.create_datetime_features:
            for col in self.datetime_columns_:
                if col in X_engineered.columns:
                    try:
                        # Convert to datetime if not already
                        if not pd.api.types.is_datetime64_any_dtype(X_engineered[col]):
                            X_engineered[col] = pd.to_datetime(X_engineered[col], errors='coerce')
                        
                        # Extract datetime features
                        X_engineered[f'{col}_year'] = X_engineered[col].dt.year
                        X_engineered[f'{col}_month'] = X_engineered[col].dt.month
                        X_engineered[f'{col}_day'] = X_engineered[col].dt.day
                        X_engineered[f'{col}_dayofweek'] = X_engineered[col].dt.dayofweek
                        X_engineered[f'{col}_hour'] = X_engineered[col].dt.hour
                        X_engineered[f'{col}_is_weekend'] = (X_engineered[col].dt.dayofweek >= 5).astype(int)
                        X_engineered[f'{col}_quarter'] = X_engineered[col].dt.quarter
                        
                        # Add to engineered features list
                        datetime_features = [f'{col}_year', f'{col}_month', f'{col}_day', f'{col}_dayofweek', 
                                           f'{col}_hour', f'{col}_is_weekend', f'{col}_quarter']
                        self.engineered_features_.extend(datetime_features)
                        
                    except Exception as e:
                        print(f"Warning: Could not engineer datetime features for {col}: {e}")
        
        # Interaction terms (limited to avoid explosion)
        if self.create_interaction_terms and len(self.numeric_columns_) >= 2:
            interaction_count = 0
            for i, col1 in enumerate(self.numeric_columns_):
                for col2 in self.numeric_columns_[i+1:]:
                    if interaction_count >= self.max_interactions:
                        break
                    
                    try:
                        # Multiplicative interaction
                        X_engineered[f'{col1}_x_{col2}'] = X_engineered[col1] * X_engineered[col2]
                        self.engineered_features_.append(f'{col1}_x_{col2}')
                        interaction_count += 1
                        
                        # Ratio features (avoid division by zero)
                        if (X_engineered[col2] != 0).all():
                            X_engineered[f'{col1}_div_{col2}'] = X_engineered[col1] / X_engineered[col2]
                            self.engineered_features_.append(f'{col1}_div_{col2}')
                            interaction_count += 1
                            
                    except Exception as e:
                        continue
                
                if interaction_count >= self.max_interactions:
                    break
        
        # Return numpy array for sklearn compatibility
        return X_engineered.values
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names for sklearn compatibility"""
        feature_names = input_features or self.feature_names_
        return list(feature_names) + self.engineered_features_
    
    def get_engineered_features(self):
        """Get list of engineered features"""
        return self.engineered_features_

class DataPreprocessingService:
    """Service for data preprocessing and EDA"""
    
    def __init__(self):
        self.scaler = None
        self.encoder = None
        self.preprocessor = None
        self.quality_reporter = DataQualityReporter()
        
    def generate_eda(self, df: pd.DataFrame, upload_id: str, 
                    column_types: Dict[str, str]) -> EDAResponse:
        """Generate comprehensive EDA report"""
        
        # Basic statistics
        summary_stats = self._get_summary_statistics(df, column_types)
        
        # Correlation matrix for numerical columns
        correlation_matrix = self._get_correlation_matrix(df, column_types)
        
        # Missing data information
        missing_data_info = self._get_missing_data_info(df)
        
        # Generate visualizations
        visualizations = self._generate_visualizations(df, column_types)
        
        return EDAResponse(
            upload_id=upload_id,
            summary_stats=summary_stats,
            correlation_matrix=correlation_matrix,
            missing_data_info=missing_data_info,
            visualizations=visualizations
        )
    
    def _get_summary_statistics(self, df: pd.DataFrame, 
                               column_types: Dict[str, str]) -> Dict[str, Any]:
        """Get summary statistics for the dataset"""
        
        stats = {
            "dataset_shape": list(df.shape),  # Convert tuple to list
            "total_missing": int(df.isnull().sum().sum()),  # Convert to int
            "memory_usage_mb": float(df.memory_usage(deep=True).sum() / 1024 / 1024),  # Convert to float
            "numerical_columns": [],
            "categorical_columns": [],
            "datetime_columns": [],
            "text_columns": []
        }
        
        for col, col_type in column_types.items():
            col_info = {
                "name": col,
                "type": col_type,
                "missing_count": int(df[col].isnull().sum()),
                "missing_percentage": float(df[col].isnull().sum() / len(df) * 100),
                "unique_values": int(df[col].nunique())
            }
            
            if col_type == ColumnType.NUMERICAL.value:
                col_info.update({
                    "mean": float(df[col].mean()) if not df[col].isnull().all() else None,
                    "median": float(df[col].median()) if not df[col].isnull().all() else None,
                    "std": float(df[col].std()) if not df[col].isnull().all() else None,
                    "min": float(df[col].min()) if not df[col].isnull().all() else None,
                    "max": float(df[col].max()) if not df[col].isnull().all() else None,
                    "q25": float(df[col].quantile(0.25)) if not df[col].isnull().all() else None,
                    "q75": float(df[col].quantile(0.75)) if not df[col].isnull().all() else None
                })
                stats["numerical_columns"].append(col_info)
                
            elif col_type == ColumnType.CATEGORICAL.value:
                value_counts = df[col].value_counts().head(10)
                col_info.update({
                    "most_frequent": value_counts.index[0] if len(value_counts) > 0 else None,
                    "most_frequent_count": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                    "top_values": {str(k): int(v) for k, v in value_counts.items()}  # Convert keys and values
                })
                stats["categorical_columns"].append(col_info)
                
            elif col_type == ColumnType.DATETIME.value:
                if not df[col].isnull().all():
                    col_info.update({
                        "min_date": str(df[col].min()),
                        "max_date": str(df[col].max()),
                        "date_range_days": int((df[col].max() - df[col].min()).days)  # Convert to int
                    })
                stats["datetime_columns"].append(col_info)
                
            else:  # TEXT
                col_info.update({
                    "avg_length": float(df[col].astype(str).str.len().mean()),
                    "max_length": int(df[col].astype(str).str.len().max()),
                    "min_length": int(df[col].astype(str).str.len().min())
                })
                stats["text_columns"].append(col_info)
        
        return stats
    
    def _get_correlation_matrix(self, df: pd.DataFrame, 
                               column_types: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """Calculate correlation matrix for numerical columns"""
        
        numerical_cols = [col for col, col_type in column_types.items() 
                         if col_type == ColumnType.NUMERICAL.value]
        
        if len(numerical_cols) < 2:
            return None
        
        corr_matrix = df[numerical_cols].corr()
        
        # Convert to dictionary format for JSON serialization
        correlation_data = {
            "columns": numerical_cols,
            "matrix": corr_matrix.values.tolist(),
            "pairs": []
        }
        
        # Find highly correlated pairs
        for i in range(len(numerical_cols)):
            for j in range(i+1, len(numerical_cols)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:  # High correlation threshold
                    correlation_data["pairs"].append({
                        "col1": numerical_cols[i],
                        "col2": numerical_cols[j],
                        "correlation": float(corr_val)
                    })
        
        return correlation_data
    
    def _get_missing_data_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing data patterns"""
        
        missing_info = {
            "total_missing": int(df.isnull().sum().sum()),
            "missing_percentage": float(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100),
            "columns_with_missing": [],
            "missing_patterns": []
        }
        
        # Per-column missing data
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                missing_info["columns_with_missing"].append({
                    "column": col,
                    "missing_count": int(missing_count),
                    "missing_percentage": float(missing_count / len(df) * 100)
                })
        
        # Missing data patterns (combinations of missing columns)
        if missing_info["columns_with_missing"]:
            missing_pattern = df.isnull().sum(axis=1).value_counts().head(5)
            for pattern, count in missing_pattern.items():
                missing_info["missing_patterns"].append({
                    "missing_columns_count": int(pattern),
                    "rows_with_pattern": int(count),
                    "percentage": float(count / len(df) * 100)
                })
        
        return missing_info
    
    def _generate_visualizations(self, df: pd.DataFrame, 
                               column_types: Dict[str, str]) -> List[Dict[str, Any]]:
        """Generate visualization data for the frontend"""
        
        visualizations = []
        
        # 1. Data types distribution
        type_counts = list(column_types.values())
        type_distribution = pd.Series(type_counts).value_counts()
        
        fig = px.pie(
            values=type_distribution.values,
            names=type_distribution.index,
            title="Column Types Distribution"
        )
        fig.update_layout(
            height=400,
            margin=dict(t=50, b=50, l=50, r=50)
        )
        visualizations.append({
            "type": "pie",
            "title": "Column Types Distribution",
            "data": json.dumps(fig, cls=PlotlyJSONEncoder)
        })
        
        # 2. Missing data heatmap
        if df.isnull().sum().sum() > 0:
            missing_data = df.isnull().sum()
            missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
            
            fig = px.bar(
                x=missing_data.index,
                y=missing_data.values,
                title="Missing Data by Column",
                labels={"x": "Columns", "y": "Missing Count"}
            )
            fig.update_xaxis(tickangle=45)
            fig.update_layout(
                height=400,
                margin=dict(t=50, b=100, l=50, r=50)
            )
            
            visualizations.append({
                "type": "bar",
                "title": "Missing Data by Column",
                "data": json.dumps(fig, cls=PlotlyJSONEncoder)
            })
        
        # 3. Numerical columns distributions
        numerical_cols = [col for col, col_type in column_types.items() 
                         if col_type == ColumnType.NUMERICAL.value]
        
        for col in numerical_cols[:4]:  # Limit to first 4 to avoid too many plots
            try:
                fig = px.histogram(
                    df,
                    x=col,
                    title=f"Distribution of {col}",
                    nbins=30
                )
                fig.update_layout(
                    height=400,
                    margin=dict(t=50, b=50, l=50, r=50)
                )
                
                visualizations.append({
                    "type": "histogram",
                    "title": f"Distribution of {col}",
                    "data": json.dumps(fig, cls=PlotlyJSONEncoder)
                })
            except Exception:
                continue  # Skip problematic columns
        
        # 4. Categorical columns distributions
        categorical_cols = [col for col, col_type in column_types.items() 
                           if col_type == ColumnType.CATEGORICAL.value]
        
        for col in categorical_cols[:3]:  # Limit to first 3
            try:
                value_counts = df[col].value_counts().head(10)
                if len(value_counts) > 1:
                    fig = px.bar(
                        x=value_counts.index,
                        y=value_counts.values,
                        title=f"Top 10 Values in {col}",
                        labels={"x": col, "y": "Count"}
                    )
                    fig.update_xaxis(tickangle=45)
                    fig.update_layout(
                        height=400,
                        margin=dict(t=50, b=100, l=50, r=50)
                    )
                    
                    visualizations.append({
                        "type": "bar",
                        "title": f"Top 10 Values in {col}",
                        "data": json.dumps(fig, cls=PlotlyJSONEncoder)
                    })
            except Exception:
                continue
        
        return visualizations
    
    def preprocess_data(self, df: pd.DataFrame, target_column: str,
                       problem_type: str, column_types: Dict[str, str],
                       test_size: float = 0.2, dataset_info=None) -> Dict[str, Any]:
        """Enhanced preprocessing with quality checks and feature selection"""
        
        # Separate features and target
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Apply quality checks - drop low-quality columns
        quality_info = {}
        if dataset_info and hasattr(dataset_info, 'columns_to_drop'):
            columns_to_drop = dataset_info.columns_to_drop or []
            columns_to_hash_encode = dataset_info.columns_to_hash_encode or []
            
            if columns_to_drop:
                X = X.drop(columns=columns_to_drop, errors='ignore')
                quality_info['dropped_columns'] = columns_to_drop
                
        # Handle target variable
        if problem_type == ProblemType.CLASSIFICATION.value:
            # Encode target if it's categorical
            if not pd.api.types.is_numeric_dtype(y):
                target_encoder = LabelEncoder()
                y = target_encoder.fit_transform(y)
            else:
                target_encoder = None
        else:
            # For regression, ensure target is numeric
            y = pd.to_numeric(y, errors='coerce')
            if y.isnull().any():
                raise ValueError("Target column contains non-numeric values for regression")
            target_encoder = None
        
        # Identify column types for preprocessing (after dropping columns)
        numerical_cols = []
        categorical_cols = []
        hash_encode_cols = []
        
        for col in X.columns:
            if col in column_types:
                # Check if this column should be hash encoded
                if dataset_info and hasattr(dataset_info, 'columns_to_hash_encode') and col in (dataset_info.columns_to_hash_encode or []):
                    hash_encode_cols.append(col)
                elif column_types[col] == ColumnType.NUMERICAL.value:
                    numerical_cols.append(col)
                elif column_types[col] == ColumnType.CATEGORICAL.value:
                    categorical_cols.append(col)
                # Skip datetime and text columns for now
        
        # STEP 1: Feature Engineering (before other preprocessing)
        feature_engineer = FeatureEngineer(create_datetime_features=True, create_interaction_terms=False)
        X_engineered = feature_engineer.fit_transform(X)
        
        # Ensure X_engineered is a DataFrame
        if not isinstance(X_engineered, pd.DataFrame):
            # Reconstruct DataFrame from numpy array
            original_cols = list(X.columns)
            engineered_cols = feature_engineer.get_feature_names_out(original_cols)
            X_engineered = pd.DataFrame(X_engineered, columns=engineered_cols, index=X.index)
        
        # Update column lists with engineered features (only for numerical engineered features)
        engineered_feature_names = feature_engineer.get_engineered_features()
        for col in X_engineered.columns:
            if col in engineered_feature_names and pd.api.types.is_numeric_dtype(X_engineered[col]):
                if col not in numerical_cols:
                    numerical_cols.append(col)
        
        # STEP 2: Enhanced Preprocessing Pipeline
        preprocessors = []
        preprocessing_info = {
            'outlier_info': {},
            'scaling_info': {},
            'encoding_info': {},
            'engineered_features': feature_engineer.get_engineered_features()
        }
        
        if numerical_cols:
            # Advanced numerical pipeline with outlier detection and smart scaling
            numerical_pipeline = Pipeline([
                ('cleaner', NumericCleaner()),  # Convert text errors to NaN
                ('outlier_detector', OutlierDetector(method='iqr', clip_outliers=True)),  # Handle outliers
                ('imputer', SimpleImputer(strategy='median')),  # Impute NaN with median
                ('scaler', SmartScaler(auto_detect=True))  # Intelligent scaling
            ])
            preprocessors.append(('num', numerical_pipeline, numerical_cols))
        
        if categorical_cols:
            # Smart categorical encoding based on cardinality
            categorical_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('smart_encoder', SmartCategoricalEncoder())
            ])
            preprocessors.append(('cat', categorical_pipeline, categorical_cols))
            
        if hash_encode_cols:
            # Hash encoding for very high cardinality categorical features
            hash_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('hasher', FeatureHasher(n_features=32, input_type='string'))
            ])
            preprocessors.append(('hash', hash_pipeline, hash_encode_cols))
        
        
        if not preprocessors:
            raise ValueError("No suitable columns found for preprocessing")
        
        # Create column transformer
        self.preprocessor = ColumnTransformer(
            transformers=preprocessors,
            remainder='drop'  # Drop columns that are not processed
        )
        
        # Split data using engineered features
        X_train, X_test, y_train, y_test = train_test_split(
            X_engineered, y, test_size=test_size, random_state=42, 
            stratify=y if problem_type == ProblemType.CLASSIFICATION.value else None
        )
        
        # Fit and transform the data
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)
        
        # Extract advanced preprocessing information
        for name, pipeline, cols in preprocessors:
            if name == 'num' and numerical_cols:
                # Get outlier and scaling information
                num_pipeline = self.preprocessor.named_transformers_['num']
                if hasattr(num_pipeline.named_steps['outlier_detector'], 'get_outlier_summary'):
                    preprocessing_info['outlier_info'] = num_pipeline.named_steps['outlier_detector'].get_outlier_summary()
                if hasattr(num_pipeline.named_steps['scaler'], 'get_scaling_info'):
                    preprocessing_info['scaling_info'] = num_pipeline.named_steps['scaler'].get_scaling_info()
            
            elif name == 'cat' and categorical_cols:
                # Get encoding information
                cat_pipeline = self.preprocessor.named_transformers_['cat']
                if hasattr(cat_pipeline.named_steps['smart_encoder'], 'get_encoding_info'):
                    preprocessing_info['encoding_info'] = cat_pipeline.named_steps['smart_encoder'].get_encoding_info()
        
        # Convert to DataFrame for feature selection
        try:
            feature_names = self._get_feature_names()
            X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
            X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)
        except:
            # Fallback if feature names can't be generated
            n_features = X_train_processed.shape[1]
            feature_names = [f"feature_{i}" for i in range(n_features)]
            X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
            X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)
        
        # Apply feature selection if we have too many features
        feature_selection_info = {}
        if len(feature_names) > 100:  # Apply feature selection for high-dimensional data
            max_features = min(50, len(feature_names) // 2)  # Keep at most 50 features
            feature_selector = FeatureSelector(
                problem_type=problem_type, 
                max_features=max_features,
                correlation_threshold=0.95
            )
            
            # Fit feature selector
            feature_selector.fit(X_train_df, y_train)
            
            # Transform data
            X_train_selected = feature_selector.transform(X_train_df)
            X_test_selected = feature_selector.transform(X_test_df)
            
            # Update feature info
            feature_selection_info = feature_selector.feature_selection_info_
            selected_feature_names = feature_selector.get_feature_names_out()
            
            # Convert back to numpy arrays
            X_train_processed = X_train_selected.values
            X_test_processed = X_test_selected.values
            feature_names = selected_feature_names
        
        # Generate comprehensive data quality report
        quality_report = self.quality_reporter.generate_data_summary(
            df=X.copy(),  # Use original data for quality analysis
            column_types=column_types,
            preprocessing_info=preprocessing_info
        )
        
        return {
            "X_train": X_train_processed,
            "X_test": X_test_processed,
            "y_train": y_train,
            "y_test": y_test,
            "feature_names": feature_names,
            "target_encoder": target_encoder,
            "preprocessor": self.preprocessor,
            "original_columns": {
                "numerical": numerical_cols,
                "categorical": categorical_cols,
                "hash_encoded": hash_encode_cols
            },
            "quality_info": quality_info,
            "feature_selection_info": feature_selection_info,
            "preprocessing_info": preprocessing_info,  # Advanced preprocessing details
            "quality_report": quality_report  # Comprehensive data quality analysis
        }
    
    def _get_feature_names(self) -> List[str]:
        """Get feature names after preprocessing"""
        feature_names = []
        
        if self.preprocessor is None:
            return feature_names
        
        for name, transformer, columns in self.preprocessor.transformers_:
            if name == 'num':
                feature_names.extend(columns)
            elif name == 'cat':
                # For one-hot encoded features
                if hasattr(transformer.named_steps['onehot'], 'get_feature_names_out'):
                    encoded_names = transformer.named_steps['onehot'].get_feature_names_out(columns)
                    feature_names.extend(encoded_names)
                else:
                    # Fallback for older sklearn versions
                    feature_names.extend([f"{col}_{i}" for col in columns for i in range(10)])  # Approximate
        
        return feature_names