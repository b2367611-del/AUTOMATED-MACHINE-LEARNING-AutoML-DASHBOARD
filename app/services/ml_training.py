import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostRegressor
import joblib
import time
import uuid
from pathlib import Path
from typing import Dict, List, Any, Tuple
import json
import warnings
warnings.filterwarnings('ignore')

from app.models.schemas import ProblemType, ModelPerformance, TrainingResponse

class MLTrainingService:
    """Service for automated machine learning training"""
    
    def __init__(self, models_dir: str = "trained_models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Define model configurations
        self.classification_models = {
            "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
            "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
            "XGBoost": xgb.XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0),
            "LightGBM": lgb.LGBMClassifier(random_state=42, verbosity=-1),
            "CatBoost": CatBoostClassifier(random_state=42, verbose=False),
            "SVM": SVC(random_state=42, probability=True)
        }
        
        self.regression_models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(random_state=42, n_estimators=100),
            "XGBoost": xgb.XGBRegressor(random_state=42, verbosity=0),
            "LightGBM": lgb.LGBMRegressor(random_state=42, verbosity=-1),
            "CatBoost": CatBoostRegressor(random_state=42, verbose=False),
            "SVM": SVR()
        }
    
    def train_models(self, preprocessed_data: Dict[str, Any], problem_type: str,
                    cv_folds: int = 5, models_to_train: List[str] = None) -> Dict[str, Any]:
        """Train multiple models and return results"""
        
        training_id = str(uuid.uuid4())
        
        # Extract data
        X_train = preprocessed_data["X_train"]
        X_test = preprocessed_data["X_test"]
        y_train = preprocessed_data["y_train"]
        y_test = preprocessed_data["y_test"]
        
        # Select models based on problem type
        if problem_type == ProblemType.CLASSIFICATION.value:
            available_models = self.classification_models
            scoring_metric = 'accuracy'
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        else:
            available_models = self.regression_models
            scoring_metric = 'r2'
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Filter models if specified
        if models_to_train:
            available_models = {k: v for k, v in available_models.items() if k in models_to_train}
        
        # Check data characteristics for model suitability
        data_warnings = []
        models_to_exclude = []
        
        # Detect extreme value ranges that may cause SVM instability
        if problem_type == ProblemType.REGRESSION.value:
            target_range = float(np.max(y_train)) - float(np.min(y_train))
            target_scale = float(np.max(np.abs(y_train)))
            
            # If target values are extremely large (> 1e8) or have huge range, exclude SVM
            if target_scale > 1e8 or target_range > 1e9:
                if "SVM" in available_models:
                    models_to_exclude.append("SVM")
                    data_warnings.append(f"SVM excluded due to extreme target scale (max: {target_scale:.2e})")
                    print(f"⚠️  Warning: Excluding SVM due to extreme target values (scale: {target_scale:.2e})")
        
        # Remove excluded models
        for model_name in models_to_exclude:
            available_models.pop(model_name, None)
        
        # Train models
        model_results = []
        trained_models = {}
        training_errors = []
        
        for model_name, model in available_models.items():
            try:
                print(f"Training {model_name}...")
                start_time = time.time()
                
                # Special validation for XGBoost classifier to prevent class mismatch
                if problem_type == ProblemType.CLASSIFICATION.value and "XGBoost" in model_name:
                    unique_classes = np.unique(y_train)
                    expected_classes = np.arange(len(unique_classes))
                    
                    # If classes don't start from 0, remap them
                    if not np.array_equal(unique_classes, expected_classes):
                        print(f"Warning: Remapping classes for {model_name}")
                        print(f"Original classes: {unique_classes}")
                        print(f"Expected classes: {expected_classes}")
                        
                        # Create a mapping from original to expected classes
                        class_mapping = dict(zip(unique_classes, expected_classes))
                        y_train_mapped = np.array([class_mapping[cls] for cls in y_train])
                        y_test_mapped = np.array([class_mapping[cls] for cls in y_test])
                        
                        # Use mapped targets for this model
                        y_train_for_model = y_train_mapped
                        y_test_for_model = y_test_mapped
                    else:
                        y_train_for_model = y_train
                        y_test_for_model = y_test
                else:
                    y_train_for_model = y_train
                    y_test_for_model = y_test
                
                # Validate data before training
                if len(X_train) == 0 or len(y_train_for_model) == 0:
                    raise ValueError(f"Empty training data for {model_name}")
                
                # Cross-validation on preprocessed data
                print(f"  Running cross-validation for {model_name}...")
                cv_scores = cross_val_score(
                    model, X_train, y_train_for_model, 
                    cv=cv, scoring=scoring_metric
                )
                
                if len(cv_scores) == 0 or np.all(np.isnan(cv_scores)):
                    raise ValueError(f"Cross-validation failed for {model_name}")
                
                # Fit on full training data
                print(f"  Fitting {model_name} on full training data...")
                model.fit(X_train, y_train_for_model)
                
                # Predict on test set
                y_pred = model.predict(X_test)
                
                # Validate predictions
                if len(y_pred) == 0 or np.all(np.isnan(y_pred)):
                    raise ValueError(f"Invalid predictions from {model_name}")
                
                if problem_type == ProblemType.CLASSIFICATION.value and hasattr(model, "predict_proba"):
                    y_pred_proba = model.predict_proba(X_test)
                else:
                    y_pred_proba = None
                
                # Calculate metrics (use original targets for metrics)
                metrics = self._calculate_metrics(
                    y_test_for_model, y_pred, y_pred_proba, problem_type
                )
                
                # Validate metrics
                if not metrics or any(np.isnan(list(metrics.values()))):
                    raise ValueError(f"Invalid metrics computed for {model_name}")
                
                # Add cross-validation score
                metrics[f'cv_{scoring_metric}_mean'] = float(np.mean(cv_scores))
                metrics[f'cv_{scoring_metric}_std'] = float(np.std(cv_scores))
                
                training_time = time.time() - start_time
                
                # Create a pipeline for deployment (includes preprocessing)
                if preprocessed_data.get("preprocessor"):
                    deployment_pipeline = Pipeline([
                        ('preprocessor', preprocessed_data["preprocessor"]),
                        ('model', model)
                    ])
                else:
                    deployment_pipeline = model
                
                # Store results
                model_performance = ModelPerformance(
                    model_name=model_name,
                    metrics=metrics,
                    training_time=training_time
                )
                model_results.append(model_performance)
                trained_models[model_name] = deployment_pipeline
                
                print(f"✓ Completed {model_name} in {training_time:.2f}s")
                
            except Exception as e:
                error_msg = f"Error training {model_name}: {str(e)}"
                print(f"❌ {error_msg}")
                training_errors.append({
                    "model_name": model_name,
                    "error": str(e),
                    "error_type": type(e).__name__
                })
                continue
        
        if not model_results:
            raise ValueError("No models could be trained successfully")
        
        # Determine best model
        best_model_name, best_model = self._select_best_model(
            model_results, trained_models, problem_type
        )
        
        # Mark best model
        for result in model_results:
            result.is_best = (result.model_name == best_model_name)
        
        # Save best model and metadata
        self._save_model_artifacts(
            training_id, best_model, preprocessed_data, 
            problem_type, model_results, best_model_name
        )
        
        # Create training summary
        training_summary = {
            "training_id": training_id,
            "problem_type": problem_type,
            "models_trained": len(model_results),
            "models_attempted": len(available_models) + len(models_to_exclude),
            "models_failed": len(training_errors),
            "models_excluded": len(models_to_exclude),
            "best_model": best_model_name,
            "feature_names": preprocessed_data.get("feature_names", []),
            "cv_folds": cv_folds,
            "test_size": len(y_test) / (len(y_train) + len(y_test)),
            "warnings": data_warnings,
            "training_errors": training_errors
        }
        
        return {
            "training_id": training_id,
            "problem_type": problem_type,
            "best_model": best_model_name,
            "model_performances": model_results,
            "training_summary": training_summary,
            "warnings": data_warnings,
            "training_errors": training_errors
        }
    
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba, problem_type: str) -> Dict[str, float]:
        """Calculate metrics based on problem type"""
        
        metrics = {}
        
        if problem_type == ProblemType.CLASSIFICATION.value:
            # Classification metrics
            metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
            
            # Handle binary vs multiclass
            if len(np.unique(y_true)) == 2:
                metrics['precision'] = float(precision_score(y_true, y_pred))
                metrics['recall'] = float(recall_score(y_true, y_pred))
                metrics['f1_score'] = float(f1_score(y_true, y_pred))
                
                if y_pred_proba is not None:
                    metrics['roc_auc'] = float(roc_auc_score(y_true, y_pred_proba[:, 1]))
            else:
                metrics['precision'] = float(precision_score(y_true, y_pred, average='weighted'))
                metrics['recall'] = float(recall_score(y_true, y_pred, average='weighted'))
                metrics['f1_score'] = float(f1_score(y_true, y_pred, average='weighted'))
                
                if y_pred_proba is not None:
                    try:
                        metrics['roc_auc'] = float(roc_auc_score(y_true, y_pred_proba, multi_class='ovr'))
                    except ValueError:
                        pass  # Skip ROC AUC for some multiclass cases
        
        else:
            # Regression metrics
            metrics['r2_score'] = float(r2_score(y_true, y_pred))
            metrics['mean_squared_error'] = float(mean_squared_error(y_true, y_pred))
            metrics['mean_absolute_error'] = float(mean_absolute_error(y_true, y_pred))
            metrics['root_mean_squared_error'] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        
        return metrics
    
    def _select_best_model(self, model_results: List[ModelPerformance], 
                          trained_models: Dict[str, Any], problem_type: str) -> Tuple[str, Any]:
        """Select the best model based on the primary metric"""
        
        if problem_type == ProblemType.CLASSIFICATION.value:
            primary_metric = 'accuracy'
            best_result = max(model_results, key=lambda x: x.metrics.get(primary_metric, 0))
        else:
            primary_metric = 'r2_score'
            best_result = max(model_results, key=lambda x: x.metrics.get(primary_metric, -float('inf')))
        
        return best_result.model_name, trained_models[best_result.model_name]
    
    def _save_model_artifacts(self, training_id: str, best_model: Any, 
                             preprocessed_data: Dict[str, Any], problem_type: str,
                             model_results: List[ModelPerformance], best_model_name: str):
        """Save model and metadata"""
        
        model_dir = self.models_dir / training_id
        model_dir.mkdir(exist_ok=True)
        
        # Save the best model
        model_path = model_dir / "best_model.joblib"
        joblib.dump(best_model, model_path)
        
        # Save metadata
        metadata = {
            "training_id": training_id,
            "problem_type": problem_type,
            "best_model_name": best_model_name,
            "feature_names": preprocessed_data.get("feature_names", []),
            "target_encoder": None,  # Will be handled separately if needed
            "preprocessor": None,  # Will be handled separately if needed
            "model_performances": [result.dict() for result in model_results],
            "original_columns": preprocessed_data.get("original_columns", {}),
            "timestamp": time.time()
        }
        
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Save target encoder if exists
        if preprocessed_data.get("target_encoder"):
            encoder_path = model_dir / "target_encoder.joblib"
            joblib.dump(preprocessed_data["target_encoder"], encoder_path)
        
        # Save quality report if available
        if preprocessed_data.get("quality_report"):
            quality_report_path = model_dir / "quality_report.json"
            with open(quality_report_path, "w") as f:
                json.dump(preprocessed_data["quality_report"], f, indent=2)
        
        # Save preprocessor separately if exists
        if preprocessed_data.get("preprocessor"):
            preprocessor_path = model_dir / "preprocessor.joblib"
            joblib.dump(preprocessed_data["preprocessor"], preprocessor_path)
        
        print(f"Model artifacts saved to {model_dir}")
    
    def load_model(self, training_id: str) -> Dict[str, Any]:
        """Load a trained model and its metadata"""
        
        model_dir = self.models_dir / training_id
        if not model_dir.exists():
            raise ValueError(f"Training ID {training_id} not found")
        
        # Load metadata
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        # Load model
        model_path = model_dir / "best_model.joblib"
        model = joblib.load(model_path)
        
        # Load additional artifacts
        artifacts = {"model": model, "metadata": metadata}
        
        # Load target encoder if exists
        encoder_path = model_dir / "target_encoder.joblib"
        if encoder_path.exists():
            artifacts["target_encoder"] = joblib.load(encoder_path)
        
        # Load preprocessor if exists
        preprocessor_path = model_dir / "preprocessor.joblib"
        if preprocessor_path.exists():
            artifacts["preprocessor"] = joblib.load(preprocessor_path)
        
        return artifacts
    
    def predict(self, training_id: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Make predictions using a trained model"""
        
        # Load model artifacts
        artifacts = self.load_model(training_id)
        model = artifacts["model"]
        metadata = artifacts["metadata"]
        
        try:
            # Make predictions
            predictions = model.predict(data)
            
            # Get prediction probabilities for classification
            if metadata["problem_type"] == ProblemType.CLASSIFICATION.value:
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(data)
                else:
                    probabilities = None
            else:
                probabilities = None
            
            # Decode predictions if target encoder was used
            if artifacts.get("target_encoder"):
                predictions = artifacts["target_encoder"].inverse_transform(predictions)
            
            result = {
                "predictions": predictions.tolist(),
                "model_info": {
                    "model_name": metadata["best_model_name"],
                    "problem_type": metadata["problem_type"],
                    "training_id": training_id
                }
            }
            
            if probabilities is not None:
                result["probabilities"] = probabilities.tolist()
            
            return result
            
        except Exception as e:
            raise ValueError(f"Error making predictions: {str(e)}")
    
    def get_feature_importance(self, training_id: str) -> Dict[str, Any]:
        """Get feature importance from the trained model"""
        
        artifacts = self.load_model(training_id)
        model = artifacts["model"]
        metadata = artifacts["metadata"]
        feature_names = metadata.get("feature_names", [])
        
        importance = None
        
        # Extract the actual model if it's in a pipeline
        actual_model = model
        if hasattr(model, 'named_steps'):
            # It's a pipeline, get the last step (the model)
            actual_model = list(model.named_steps.values())[-1]
        
        # Get feature importance based on model type
        if hasattr(actual_model, 'feature_importances_'):
            importance = actual_model.feature_importances_
        elif hasattr(actual_model, 'coef_'):
            # For linear models, use absolute coefficients
            if len(actual_model.coef_.shape) == 1:
                importance = np.abs(actual_model.coef_)
            else:
                # Multi-class case, take mean of absolute coefficients
                importance = np.mean(np.abs(actual_model.coef_), axis=0)
        
        if importance is not None:
            # Create feature importance dictionary
            if len(feature_names) == len(importance):
                feature_importance = dict(zip(feature_names, importance))
            else:
                # Fallback to generic names
                feature_importance = {f"feature_{i}": imp for i, imp in enumerate(importance)}
            
            # Sort by importance
            sorted_importance = dict(
                sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            )
            
            return {
                "feature_importance": sorted_importance,
                "model_name": metadata["best_model_name"]
            }
        
        return {"feature_importance": {}, "model_name": metadata["best_model_name"]}