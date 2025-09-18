"""
===============================================================================
TRAIN ML MODELS FOR SOLAR CELL OPTIMIZATION - MPP and IntSRHn_mean ONLY
===============================================================================

PURPOSE:
This script trains machine learning models for solar cell optimization focused on the
two target variables defined in the workflow: MPP and IntSRHn_mean. Matches the 
workflow from scripts 1-4.

WHAT THIS SCRIPT DOES:
1. Loads ML-ready data from script 4 (prepare_ml_data.py)
2. Trains models to predict MPP (Maximum Power Point) from device parameters
3. Trains models to predict IntSRHn_mean (mean electron interfacial recombination rate)
4. Uses robust scaling and cross-validation for reliable model performance
5. Saves trained models and scalers for optimization use

TARGET VARIABLES (matching scripts 1-4):
- MPP: Maximum Power Point (W/cm²) - efficiency target
- IntSRHn_mean: Mean electron interfacial recombination rate - recombination target

ALGORITHMS:
- Random Forest Regressor (ensemble, robust to outliers)
- XGBoost Regressor (gradient boosting, if available)
- Gradient Boosting Regressor (fallback if XGBoost unavailable)

IMPROVEMENTS:
- RobustScaler for both features and targets (handles outliers)
- 5-fold cross-validation for model selection
- Comprehensive evaluation metrics (R², MAE, RMSE)
- Proper model and scaler persistence

INPUT FILES:
- results/prepare_ml_data/X_full.csv (features from script 4)
- results/prepare_ml_data/y_efficiency_full.csv (MPP targets)
- results/prepare_ml_data/y_recombination_full.csv (IntSRHn_mean targets)

OUTPUT FILES:
- results/train_optimization_models/models/efficiency_MPP_*.joblib (MPP prediction models)
- results/train_optimization_models/models/recombination_IntSRHn_mean_*.joblib (recombination models)
- results/train_optimization_models/scalers/ (feature and target scalers)
- results/train_optimization_models/training_metadata.json (training statistics)
- results/train_optimization_models/training.log (detailed log)

USAGE:
python scripts/5_train_models.py

PREREQUISITES:
1. Run scripts/1_create_feature_names.py
2. Run scripts/2_generate_simulations.py  
3. Run scripts/3_extract_simulation_data.py
4. Run scripts/4_prepare_ml_data.py

AUTHOR: Anthony Barker
DATE: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
import joblib
import os
import logging
from datetime import datetime
import sys
import json

# Import XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    logging.info("XGBoost library available")
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost library not available. Install with: pip install xgboost")

# SHAP for comprehensive feature importance analysis
try:
    import shap
    SHAP_AVAILABLE = True
    logging.info("SHAP library available for feature importance analysis")
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP library not available. Install with: pip install shap")

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Define ML models and names directly
ML_MODELS = {}
ML_MODEL_NAMES = {}

# Add Random Forest
ML_MODELS['RandomForest'] = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
ML_MODEL_NAMES['RandomForest'] = 'Random Forest'

# Add XGBoost if available
if XGBOOST_AVAILABLE:
    ML_MODELS['XGBoost'] = xgb.XGBRegressor(
        n_estimators=100, 
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )
    ML_MODEL_NAMES['XGBoost'] = 'XGBoost'
else:
    # Fallback to sklearn Gradient Boosting if XGBoost not available
    from sklearn.ensemble import GradientBoostingRegressor
    ML_MODELS['GradientBoosting'] = GradientBoostingRegressor(n_estimators=100, random_state=42)
    ML_MODEL_NAMES['GradientBoosting'] = 'Gradient Boosting'

# Linear Regression removed - focusing on ensemble methods only

# Set up logging
log_dir = 'results/train_optimization_models'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'training.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

def load_enhanced_data():
    """Load the ML-ready data prepared by the prepare_ml_data script."""
    # Load the full dataset prepared by script 4
    X_path = 'results/prepare_ml_data/X_full.csv'
    y_efficiency_path = 'results/prepare_ml_data/y_efficiency_full.csv'
    y_recombination_path = 'results/prepare_ml_data/y_recombination_full.csv'
    
    if not os.path.exists(X_path):
        logging.error(f"ML data not found at {X_path}")
        logging.error("Please run scripts/4_prepare_ml_data.py first")
        raise FileNotFoundError(f"ML data not found: {X_path}")
    
    # Load features and targets
    X = pd.read_csv(X_path)
    y_efficiency = pd.read_csv(y_efficiency_path) if os.path.exists(y_efficiency_path) else pd.DataFrame()
    y_recombination = pd.read_csv(y_recombination_path) if os.path.exists(y_recombination_path) else pd.DataFrame()
    
    logging.info(f"Loaded ML data: X={X.shape}, y_efficiency={y_efficiency.shape}, y_recombination={y_recombination.shape}")
    logging.info(f"Features: {list(X.columns)}")
    
    # Combine into a single dataframe for compatibility
    df = X.copy()
    for col in y_efficiency.columns:
        df[col] = y_efficiency[col]
    for col in y_recombination.columns:
        df[col] = y_recombination[col]
    
    return df

def prepare_optimization_data(df):
    """Prepare data for optimization models - MPP and IntSRHn_mean only."""
    logging.info("Preparing optimization data for MPP and IntSRHn_mean targets...")
    
    # Define target columns (only the two we care about)
    target_columns = ['MPP', 'IntSRHn_mean']
    
    # All features are columns that are not target variables
    all_features = [col for col in df.columns if col not in target_columns]
    logging.info(f"All features: {all_features}")
    logging.info(f"Total features: {len(all_features)}")
    
    # Check which targets are available
    available_efficiency = ['MPP'] if 'MPP' in df.columns else []
    available_recombination = ['IntSRHn_mean'] if 'IntSRHn_mean' in df.columns else []
    
    logging.info(f"Available efficiency metrics: {available_efficiency}")
    logging.info(f"Available recombination metrics: {available_recombination}")
    
    # Verify we have the required targets
    if not available_efficiency:
        raise ValueError("MPP target not found in data. Check script 3 and 4 outputs.")
    if not available_recombination:
        raise ValueError("IntSRHn_mean target not found in data. Check script 3 and 4 outputs.")
    
    # Remove rows with missing data for required columns only
    required_cols = all_features + available_efficiency + available_recombination
    df_clean = df[required_cols].dropna()
    logging.info(f"Clean data shape: {df_clean.shape}")
    
    if len(df_clean) == 0:
        raise ValueError("No clean data available after removing missing values.")
    
    # Handle extreme outliers in targets (cap at 99th percentile)
    for target in available_efficiency + available_recombination:
        if target in df_clean.columns:
            q99 = df_clean[target].quantile(0.99)
            q01 = df_clean[target].quantile(0.01)
            df_clean[target] = df_clean[target].clip(lower=q01, upper=q99)
            logging.info(f"Clipped {target} to range [{q01:.2e}, {q99:.2e}]")
    
    # Split features and targets
    X = df_clean[all_features]
    y_efficiency = df_clean[available_efficiency]
    y_recombination = df_clean[available_recombination]
    
    logging.info(f"Features (X): {X.shape}")
    logging.info(f"Efficiency targets: {y_efficiency.shape}")
    logging.info(f"Recombination targets: {y_recombination.shape}")
    
    return X, y_efficiency, y_recombination, all_features

def evaluate_model_comprehensive(model, X_test, y_test, target_name):
    """Evaluate model with multiple metrics."""
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    # Calculate relative errors
    y_mean = np.mean(np.abs(y_test))
    if y_mean > 0:
        rmse_relative = rmse / y_mean * 100
        mae_relative = mae / y_mean * 100
    else:
        rmse_relative = 0
        mae_relative = 0
    
    # Calculate MAPE
    valid_mask = np.abs(y_test) > 1e-10
    if np.sum(valid_mask) > 0:
        mape = np.mean(np.abs((y_test[valid_mask] - y_pred[valid_mask]) / np.abs(y_test[valid_mask]))) * 100
    else:
        mape = 0
    
    return {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'rmse_relative': rmse_relative,
        'mae_relative': mae_relative,
        'mape': mape
    }

def train_efficiency_predictor_improved(X, y_efficiency, all_features):
    """Train model to predict efficiency with improved scaling and evaluation."""
    logging.info("\n=== Training Efficiency Predictor (Improved) ===")
    
    # Handle small datasets
    if len(X) < 10:
        logging.warning(f"Very small dataset ({len(X)} samples) - using all data for training")
        X_train, X_test = X, X
        y_train, y_test = y_efficiency, y_efficiency
    else:
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_efficiency, test_size=0.2, random_state=42
        )
    
    # Scale features using RobustScaler (better for outliers)
    feature_scaler = RobustScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_test_scaled = feature_scaler.transform(X_test)
    
    # Scale targets using RobustScaler
    target_scalers = {}
    y_train_scaled = y_train.copy()
    y_test_scaled = y_test.copy()
    
    for target in y_efficiency.columns:
        target_scaler = RobustScaler()
        y_train_scaled[target] = target_scaler.fit_transform(y_train[[target]]).flatten()
        y_test_scaled[target] = target_scaler.transform(y_test[[target]]).flatten()
        target_scalers[target] = target_scaler
        logging.info(f"Scaled {target}: mean={np.mean(y_train_scaled[target]):.4f}, std={np.std(y_train_scaled[target]):.4f}")
    
    # Train models for each efficiency metric
    efficiency_models = {}
    efficiency_scalers = {}
    model_metadata = {}
    
    for target in y_efficiency.columns:
        logging.info(f"\nTraining model for {target}...")
        
        # Train multiple models
        models = {}
        model_scores = {}
        
        for model_name in ML_MODEL_NAMES:
            model = ML_MODELS[model_name]
            
            # Use cross-validation for better evaluation
            cv_scores = cross_val_score(model, X_train_scaled, y_train_scaled[target], 
                                      cv=5, scoring='r2')
            
            # Train on full training set
            model.fit(X_train_scaled, y_train_scaled[target])
            
            # Evaluate on test set
            metrics = evaluate_model_comprehensive(model, X_test_scaled, y_test_scaled[target], target)
            
            models[model_name] = model
            model_scores[model_name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_metrics': metrics
            }
            
            logging.info(f"{model_name} - {target}:")
            logging.info(f"  CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            logging.info(f"  Test R²: {metrics['r2']:.4f}")
            logging.info(f"  Test RMSE: {metrics['rmse']:.4f} ({metrics['rmse_relative']:.1f}%)")
            logging.info(f"  Test MAE: {metrics['mae']:.4f} ({metrics['mae_relative']:.1f}%)")
            logging.info(f"  Test MAPE: {metrics['mape']:.1f}%")
        
        # Save best model (highest CV score)
        best_model_name = max(model_scores.keys(), key=lambda m: model_scores[m]['cv_mean'])
        efficiency_models[target] = models[best_model_name]
        efficiency_scalers[target] = {
            'feature_scaler': feature_scaler,
            'target_scaler': target_scalers[target]
        }
        
        # Save model metadata
        model_metadata[target] = {
            'best_model': best_model_name,
            'all_scores': model_scores,
            'feature_names': all_features
        }
        
        # Save model (create directories if they don't exist)
        models_dir = 'results/train_optimization_models/models'
        os.makedirs(models_dir, exist_ok=True)
        
        model_path = f'{models_dir}/efficiency_{target}.joblib'
        scaler_path = f'{models_dir}/efficiency_{target}_scalers.joblib'
        metadata_path = f'{models_dir}/efficiency_{target}_metadata.json'
        
        joblib.dump(models[best_model_name], model_path)
        joblib.dump(efficiency_scalers[target], scaler_path)
        
        with open(metadata_path, 'w') as f:
            json.dump(model_metadata[target], f, indent=2, default=str)
        
        logging.info(f"Saved best model for {target}: {best_model_name}")
    
    return efficiency_models, efficiency_scalers, model_metadata

def train_recombination_predictor_improved(X, y_recombination, all_features):
    """Train model to predict recombination with improved scaling and evaluation."""
    logging.info("\n=== Training Recombination Predictor (Improved) ===")
    
    # Handle small datasets
    if len(X) < 10:
        logging.warning(f"Very small dataset ({len(X)} samples) - using all data for training")
        X_train, X_test = X, X
        y_train, y_test = y_recombination, y_recombination
    else:
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_recombination, test_size=0.2, random_state=42
        )
    
    # Scale features using RobustScaler (better for outliers)
    feature_scaler = RobustScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_test_scaled = feature_scaler.transform(X_test)
    
    # Scale targets using RobustScaler
    target_scalers = {}
    y_train_scaled = y_train.copy()
    y_test_scaled = y_test.copy()
    
    for target in y_recombination.columns:
        target_scaler = RobustScaler()
        y_train_scaled[target] = target_scaler.fit_transform(y_train[[target]]).flatten()
        y_test_scaled[target] = target_scaler.transform(y_test[[target]]).flatten()
        target_scalers[target] = target_scaler
        logging.info(f"Scaled {target}: mean={np.mean(y_train_scaled[target]):.4f}, std={np.std(y_train_scaled[target]):.4f}")
    
    # Train models for each recombination metric
    recombination_models = {}
    recombination_scalers = {}
    model_metadata = {}
    
    for target in y_recombination.columns:
        logging.info(f"\nTraining model for {target}...")
        
        # Train multiple models
        models = {}
        model_scores = {}
        
        for model_name in ML_MODEL_NAMES:
            model = ML_MODELS[model_name]
            
            # Use cross-validation for better evaluation
            cv_scores = cross_val_score(model, X_train_scaled, y_train_scaled[target], 
                                      cv=5, scoring='r2')
            
            # Train on full training set
            model.fit(X_train_scaled, y_train_scaled[target])
            
            # Evaluate on test set
            metrics = evaluate_model_comprehensive(model, X_test_scaled, y_test_scaled[target], target)
            
            models[model_name] = model
            model_scores[model_name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'test_metrics': metrics
            }
            
            logging.info(f"{model_name} - {target}:")
            logging.info(f"  CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
            logging.info(f"  Test R²: {metrics['r2']:.4f}")
            logging.info(f"  Test RMSE: {metrics['rmse']:.4f} ({metrics['rmse_relative']:.1f}%)")
            logging.info(f"  Test MAE: {metrics['mae']:.4f} ({metrics['mae_relative']:.1f}%)")
            logging.info(f"  Test MAPE: {metrics['mape']:.1f}%")
        
        # Save best model (highest CV score)
        best_model_name = max(model_scores.keys(), key=lambda m: model_scores[m]['cv_mean'])
        recombination_models[target] = models[best_model_name]
        recombination_scalers[target] = {
            'feature_scaler': feature_scaler,
            'target_scaler': target_scalers[target]
        }
        
        # Save model metadata
        model_metadata[target] = {
            'best_model': best_model_name,
            'all_scores': model_scores,
            'feature_names': all_features
        }
        
        # Save model (create directories if they don't exist)
        models_dir = 'results/train_optimization_models/models'
        os.makedirs(models_dir, exist_ok=True)
        
        model_path = f'{models_dir}/recombination_{target}.joblib'
        scaler_path = f'{models_dir}/recombination_{target}_scalers.joblib'
        metadata_path = f'{models_dir}/recombination_{target}_metadata.json'
        
        joblib.dump(models[best_model_name], model_path)
        joblib.dump(recombination_scalers[target], scaler_path)
        
        with open(metadata_path, 'w') as f:
            json.dump(model_metadata[target], f, indent=2, default=str)
        
        logging.info(f"Saved best model for {target}: {best_model_name}")
    
    return recombination_models, recombination_scalers, model_metadata

def save_training_metadata(efficiency_metadata, recombination_metadata):
    """Save comprehensive training metadata."""
    metadata = {
        'training_date': datetime.now().isoformat(),
        'efficiency_models': efficiency_metadata,
        'recombination_models': recombination_metadata,
        'improvements': {
            'target_scaling': 'RobustScaler for both features and targets',
            'cross_validation': '5-fold CV for model selection',
            'outlier_handling': '99th percentile clipping',
            'comprehensive_evaluation': 'Multiple metrics beyond R²'
        }
    }
    
    # Create directory if it doesn't exist
    results_dir = 'results/train_optimization_models'
    os.makedirs(results_dir, exist_ok=True)
    
    metadata_path = f'{results_dir}/training_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    logging.info(f"Training metadata saved to {metadata_path}")

def main():
    """Main function for training MPP and IntSRHn_mean prediction models."""
    logging.info("Starting ML model training for solar cell optimization...")
    
    try:
        # Load data
        df = load_enhanced_data()
        logging.info(f"Loaded data with shape: {df.shape}")
        
        # Prepare optimization data
        X, y_efficiency, y_recombination, all_features = prepare_optimization_data(df)
        logging.info(f"Prepared data - X: {X.shape}, y_efficiency: {y_efficiency.shape}, y_recombination: {y_recombination.shape}")
        
        # Train efficiency predictors (MPP only)
        logging.info("\n=== Training MPP Prediction Models ===")
        efficiency_models, efficiency_scalers, efficiency_metadata = train_efficiency_predictor_improved(X, y_efficiency, all_features)
        logging.info(f"Trained {len(efficiency_models)} MPP prediction models")
        
        # Train recombination predictors (IntSRHn_mean only)
        logging.info("\n=== Training IntSRHn_mean Prediction Models ===")
        recombination_models, recombination_scalers, recombination_metadata = train_recombination_predictor_improved(X, y_recombination, all_features)
        logging.info(f"Trained {len(recombination_models)} IntSRHn_mean prediction models")
        
        # Save comprehensive metadata
        save_training_metadata(efficiency_metadata, recombination_metadata)
        
        # Print summary
        print("\n=== ML MODEL TRAINING SUMMARY ===")
        print(f"Total models trained: {len(efficiency_models) + len(recombination_models)}")
        print(f"MPP prediction models: {len(efficiency_models)}")
        print(f"IntSRHn_mean prediction models: {len(recombination_models)}")
        print("\nTarget Variables (matching scripts 1-4):")
        print("- MPP: Maximum Power Point (W/cm²)")
        print("- IntSRHn_mean: Mean electron interfacial recombination rate")
        print("\nKey Features:")
        print("- RobustScaler for features and targets (handles outliers)")
        print("- 5-fold cross-validation for model selection")
        print("- Multiple algorithms: Random Forest, XGBoost/Gradient Boosting")
        print("- Comprehensive evaluation metrics (R², MAE, RMSE)")
        print(f"\nModels and scalers saved to: {log_dir}/models/")
        
        logging.info("ML model training completed successfully!")
        
    except Exception as e:
        logging.error(f"Error in model training: {str(e)}")
        raise

if __name__ == "__main__":
    main() 