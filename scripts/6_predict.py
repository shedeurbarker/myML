"""
===============================================================================
PREDICT AND VALIDATE SOLAR CELL MODELS - MPP and IntSRHn_mean ONLY
===============================================================================

PURPOSE:
This script uses the trained ML models from Script 5 to make predictions and validate
model performance. Matches the workflow from scripts 1-5 with focus on MPP and IntSRHn_mean.

WHAT THIS SCRIPT DOES:
1. Loads trained MPP and IntSRHn_mean prediction models from Script 5
2. Creates example predictions for demonstration
3. Validates model predictions against experimental data
4. Calculates comprehensive performance metrics
5. Generates validation plots and reports

TARGET VARIABLES (matching scripts 1-5):
- MPP: Maximum Power Point (W/cm²) - efficiency target
- IntSRHn_mean: Mean electron interfacial recombination rate - recombination target

VALIDATION FEATURES:
- Proper target scaling using scalers from Script 5
- Comprehensive metrics: R², RMSE, MAE, MAPE
- Visual validation through scatter plots
- Error analysis and model performance assessment

INPUT FILES:
- results/train_optimization_models/models/efficiency_MPP.joblib (MPP prediction model)
- results/train_optimization_models/models/recombination_IntSRHn_mean.joblib (recombination model)
- results/train_optimization_models/models/*_scalers.joblib (feature and target scalers)
- results/extract_simulation_data/extracted_simulation_data.csv (data for validation)
- example_device_parameters.json (configurable example device parameters)

OUTPUT FILES:
- results/predict/predicted_values.txt (example predictions)
- results/predict/MPP_validation.png (MPP validation plot)
- results/predict/IntSRHn_mean_validation.png (recombination validation plot)
- results/predict/predictions.log (detailed execution log)
- results/predict/model_validation_metrics.csv (validation metrics)

VALIDATION METRICS:
- R² Score: Coefficient of determination
- RMSE: Root Mean Square Error (absolute and relative)
- MAE: Mean Absolute Error (absolute and relative)
- MAPE: Mean Absolute Percentage Error

PREREQUISITES:
1. Run scripts/1_create_feature_names.py
2. Run scripts/2_generate_simulations.py
3. Run scripts/3_extract_simulation_data.py
4. Run scripts/4_prepare_ml_data.py
5. Run scripts/5_train_models.py

USAGE:
python scripts/6_predict.py

AUTHOR: Anthony Barker
DATE: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys
import json
from datetime import datetime

# Import XGBoost for model loading
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    logging.info("XGBoost library available for prediction")
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost library not available. Install with: pip install xgboost")

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up logging
log_dir = 'results/predict'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, 'predictions.log')

# Clear any existing handlers to avoid conflicts
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'),  # Use 'w' mode to overwrite
        logging.StreamHandler()
    ]
)

# Test logging to ensure it's working
logging.info("Script 6 - Prediction and Validation - Logging initialized")
logging.info(f"Log file: {log_file}")

def calculate_efficiency(mpp, pin=1000):
    """
    Calculate power conversion efficiency.
    
    Args:
        mpp: Maximum power point (mW/cm²)
        pin: Incident power density (mW/cm²), default 1000 for AM1.5G
    
    Returns:
        efficiency: Power conversion efficiency (%)
    """
    if pin <= 0:
        return 0
    return (mpp / pin) * 100

def calculate_efficiency_metrics(actual_mpp, predicted_mpp, pin=1000):
    """
    Calculate efficiency metrics for validation.
    
    Args:
        actual_mpp: Actual MPP values
        predicted_mpp: Predicted MPP values
        pin: Incident power density (mW/cm²)
    
    Returns:
        dict: Efficiency metrics
    """
    # Calculate efficiencies
    actual_efficiency = calculate_efficiency(actual_mpp, pin)
    predicted_efficiency = calculate_efficiency(predicted_mpp, pin)
    
    # Calculate metrics
    r2 = r2_score(actual_efficiency, predicted_efficiency)
    rmse = np.sqrt(mean_squared_error(actual_efficiency, predicted_efficiency))
    mae = mean_absolute_error(actual_efficiency, predicted_efficiency)
    
    # Calculate relative errors
    actual_mean = np.mean(np.abs(actual_efficiency))
    if actual_mean > 0:
        rmse_relative = rmse / actual_mean * 100
        mae_relative = mae / actual_mean * 100
    else:
        rmse_relative = 0
        mae_relative = 0
    
    # Calculate MAPE
    valid_mask = np.abs(actual_efficiency) > 1e-10
    if np.sum(valid_mask) > 0:
        mape = np.mean(np.abs((actual_efficiency[valid_mask] - predicted_efficiency[valid_mask]) / np.abs(actual_efficiency[valid_mask]))) * 100
    else:
        mape = 0
    
    # Cap relative errors
    rmse_relative = min(rmse_relative, 1000)
    mae_relative = min(mae_relative, 1000)
    mape = min(mape, 1000)
    
    return {
        'actual_efficiency': actual_efficiency,
        'predicted_efficiency': predicted_efficiency,
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'rmse_relative': rmse_relative,
        'mae_relative': mae_relative,
        'mape': mape
    }

def check_prerequisites():
    """Check if all required files and directories exist."""
    logging.info("\n=== Checking Prerequisites ===")
    
    # Check for models
    required_files = [
        'results/train_optimization_models/models/efficiency_MPP.joblib',
        'results/train_optimization_models/models/efficiency_MPP_scalers.joblib',
        'results/train_optimization_models/models/recombination_IntSRHn_mean.joblib',
        'results/train_optimization_models/models/recombination_IntSRHn_mean_scalers.joblib'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        logging.error("Missing required files:")
        for file in missing_files:
            logging.error(f"  - {file}")
        logging.error("Please run 5_train_optimization_models.py first")
        return False
    
    logging.info("All prerequisites satisfied")
    return True

def ensure_predict_results_dir():
    """Ensure the prediction results directory exists."""
    os.makedirs('results/predict', exist_ok=True)

def load_optimization_models():
    """Load optimization models with proper scalers."""
    logging.info("\n=== Loading Optimization Models ===")
    
    models_dir = 'results/train_optimization_models/models'
    
    # Load MPP (efficiency) model
    efficiency_models = {}
    efficiency_scalers = {}
    
    # Load MPP model only (matching our workflow)
    mpp_model_path = f'{models_dir}/efficiency_MPP.joblib'
    mpp_scaler_path = f'{models_dir}/efficiency_MPP_scalers.joblib'
    
    if os.path.exists(mpp_model_path) and os.path.exists(mpp_scaler_path):
        try:
            model = joblib.load(mpp_model_path)
            scalers = joblib.load(mpp_scaler_path)
            efficiency_models['MPP'] = model
            efficiency_scalers['MPP'] = scalers
            logging.info(f"Loaded MPP prediction model")
        except Exception as e:
            logging.error(f"Failed to load MPP model: {e}")
    else:
        logging.error(f"MPP model files not found. Check Script 5 output.")
    
    # Load IntSRHn_mean (recombination) model
    recombination_models = {}
    recombination_scalers = {}
    
    # Load IntSRHn_mean model only (matching our workflow)
    recomb_model_path = f'{models_dir}/recombination_IntSRHn_mean.joblib'
    recomb_scaler_path = f'{models_dir}/recombination_IntSRHn_mean_scalers.joblib'
    
    if os.path.exists(recomb_model_path) and os.path.exists(recomb_scaler_path):
        try:
            model = joblib.load(recomb_model_path)
            scalers = joblib.load(recomb_scaler_path)
            recombination_models['IntSRHn_mean'] = model
            recombination_scalers['IntSRHn_mean'] = scalers
            logging.info(f"Loaded IntSRHn_mean prediction model")
        except Exception as e:
            logging.error(f"Failed to load IntSRHn_mean model: {e}")
    else:
        logging.error(f"IntSRHn_mean model files not found. Check Script 5 output.")
    
    logging.info(f"Loaded {len(efficiency_models)} efficiency models")
    logging.info(f"Loaded {len(recombination_models)} recombination models")
    
    return efficiency_models, efficiency_scalers, recombination_models, recombination_scalers

def calculate_derived_features(df):
    """Calculate derived features from primary parameters."""
    logging.info("Calculating derived features...")
    
    # Thickness features
    if all(col in df.columns for col in ['L1_L', 'L2_L', 'L3_L']):
        df['total_thickness'] = df['L1_L'] + df['L2_L'] + df['L3_L']
        df['thickness_ratio_L2'] = df['L2_L'] / (df['total_thickness'] + 1e-30)
        df['thickness_ratio_ETL'] = df['L1_L'] / (df['total_thickness'] + 1e-30)
        df['thickness_ratio_HTL'] = df['L3_L'] / (df['total_thickness'] + 1e-30)
    
    # Energy gap features (use absolute value to ensure positive gaps)
    if all(col in df.columns for col in ['L1_E_c', 'L1_E_v', 'L2_E_c', 'L2_E_v', 'L3_E_c', 'L3_E_v']):
        df['energy_gap_L1'] = abs(df['L1_E_c'] - df['L1_E_v'])
        df['energy_gap_L2'] = abs(df['L2_E_c'] - df['L2_E_v'])
        df['energy_gap_L3'] = abs(df['L3_E_c'] - df['L3_E_v'])
    
    # Band alignment features
    if all(col in df.columns for col in ['L1_E_c', 'L2_E_c', 'L3_E_c', 'L1_E_v', 'L2_E_v', 'L3_E_v']):
        df['band_offset_L1_L2'] = df['L2_E_c'] - df['L1_E_c']
        df['band_offset_L2_L3'] = df['L3_E_c'] - df['L2_E_c']
        df['conduction_band_offset'] = df['L3_E_c'] - df['L1_E_c']
        df['valence_band_offset'] = df['L3_E_v'] - df['L1_E_v']
    
    # Doping features
    if all(col in df.columns for col in ['L1_N_D', 'L1_N_A', 'L2_N_D', 'L2_N_A', 'L3_N_D', 'L3_N_A']):
        df['doping_ratio_L1'] = df['L1_N_D'] / (df['L1_N_A'] + 1e-30)
        df['doping_ratio_L2'] = df['L2_N_D'] / (df['L2_N_A'] + 1e-30)
        df['doping_ratio_L3'] = df['L3_N_D'] / (df['L3_N_A'] + 1e-30)
        df['total_donor_concentration'] = df['L1_N_D'] + df['L2_N_D'] + df['L3_N_D']
        df['total_acceptor_concentration'] = df['L1_N_A'] + df['L2_N_A'] + df['L3_N_A']
    
    # Material property features
    if 'energy_gap_L1' in df.columns and 'energy_gap_L2' in df.columns and 'energy_gap_L3' in df.columns:
        df['average_energy_gap'] = df[['energy_gap_L1', 'energy_gap_L2', 'energy_gap_L3']].mean(axis=1)
        df['energy_gap_variance'] = df[['energy_gap_L1', 'energy_gap_L2', 'energy_gap_L3']].var(axis=1)
    
    if 'L1_L' in df.columns and 'L2_L' in df.columns and 'L3_L' in df.columns:
        df['thickness_variance'] = df[['L1_L', 'L2_L', 'L3_L']].var(axis=1)
    
    if 'L1_N_D' in df.columns and 'L2_N_D' in df.columns and 'L3_N_D' in df.columns:
        df['doping_variance'] = df[['L1_N_D', 'L2_N_D', 'L3_N_D']].var(axis=1)
    
    # NEW: Physics-based features for recombination-efficiency relationship
    if 'MPP' in df.columns and 'IntSRHn_mean' in df.columns:
        # Recombination efficiency ratio (how much recombination affects efficiency)
        df['recombination_efficiency_ratio'] = df['IntSRHn_mean'] / (df['MPP'] + 1e-30)
        
        # Interface quality index (lower recombination relative to efficiency = better interface)
        df['interface_quality_index'] = df['MPP'] / (df['IntSRHn_mean'] + 1e-30)
    else:
        # For prediction mode where we don't have target values, use default values
        df['recombination_efficiency_ratio'] = 1e28  # Default typical value
        df['interface_quality_index'] = 1e-28  # Default typical value
    
    # NEW: Carrier transport efficiency features
    if all(col in df.columns for col in ['L1_E_c', 'L2_E_c', 'L3_E_c', 'L1_E_v', 'L2_E_v', 'L3_E_v']):
        # Conduction band alignment quality (how well aligned for electron transport)
        cb_alignment_L1_L2 = np.exp(-abs(df['L1_E_c'] - df['L2_E_c']) / 0.1)  # Exponential decay with misalignment
        cb_alignment_L2_L3 = np.exp(-abs(df['L2_E_c'] - df['L3_E_c']) / 0.1)
        df['conduction_band_alignment_quality'] = (cb_alignment_L1_L2 + cb_alignment_L2_L3) / 2
        
        # Valence band alignment quality (how well aligned for hole transport)
        vb_alignment_L1_L2 = np.exp(-abs(df['L1_E_v'] - df['L2_E_v']) / 0.1)
        vb_alignment_L2_L3 = np.exp(-abs(df['L2_E_v'] - df['L3_E_v']) / 0.1)
        df['valence_band_alignment_quality'] = (vb_alignment_L1_L2 + vb_alignment_L2_L3) / 2
    
    # NEW: Device structure optimization features
    if all(col in df.columns for col in ['L1_L', 'L2_L', 'L3_L']):
        # Thickness balance quality (penalize extreme thickness ratios)
        ideal_etl_ratio = 0.1  # ~10% of total thickness for ETL
        ideal_htl_ratio = 0.1  # ~10% of total thickness for HTL
        ideal_active_ratio = 0.8  # ~80% of total thickness for active layer
        
        etl_balance = np.exp(-abs(df['thickness_ratio_ETL'] - ideal_etl_ratio) / 0.05)
        htl_balance = np.exp(-abs(df['thickness_ratio_HTL'] - ideal_htl_ratio) / 0.05)
        active_balance = np.exp(-abs(df['thickness_ratio_L2'] - ideal_active_ratio) / 0.1)
        
        df['thickness_balance_quality'] = (etl_balance + htl_balance + active_balance) / 3
        
        # Transport layer balance (ETL and HTL should be similar thickness)
        df['transport_layer_balance'] = np.exp(-abs(df['L1_L'] - df['L3_L']) / (df['L1_L'] + df['L3_L'] + 1e-30))
    
    # NEW: Doping optimization features
    if all(col in df.columns for col in ['doping_ratio_L1', 'doping_ratio_L2', 'doping_ratio_L3']):
        # Average doping ratio (measure of overall doping level)
        df['average_doping_ratio'] = df[['doping_ratio_L1', 'doping_ratio_L2', 'doping_ratio_L3']].mean(axis=1)
        
        # Doping consistency across layers
        df['doping_consistency'] = 1 / (1 + df[['doping_ratio_L1', 'doping_ratio_L2', 'doping_ratio_L3']].var(axis=1))
    
    # NEW: Energy level optimization features
    if all(col in df.columns for col in ['energy_gap_L1', 'energy_gap_L2', 'energy_gap_L3']):
        # Energy gap progression (absolute value to ensure positive)
        df['energy_gap_progression'] = abs((df['energy_gap_L2'] - df['energy_gap_L1']) * (df['energy_gap_L3'] - df['energy_gap_L2']))
        
        # Energy gap uniformity (for specific device types)
        df['energy_gap_uniformity'] = 1 / (1 + df[['energy_gap_L1', 'energy_gap_L2', 'energy_gap_L3']].var(axis=1))
    
    logging.info(f"Calculated all derived features including enhanced physics-based features")
    return df

def prepare_input_data(input_data, feature_names):
    """Prepare input data for prediction with proper feature selection."""
    # Calculate derived features
    input_data = calculate_derived_features(input_data)
    
    # Select only the features used during training
    available_features = [f for f in feature_names if f in input_data.columns]
    missing_features = [f for f in feature_names if f not in input_data.columns]
    
    if missing_features:
        logging.warning(f"Missing features: {missing_features}")
        # Fill missing features with zeros
        for feature in missing_features:
            input_data[feature] = 0
    
    return input_data[feature_names]

def make_predictions_all_models(input_data):
    """Make predictions using models with proper scaling."""
    logging.info("\n=== Making Predictions ===")
    
    # Load models
    efficiency_models, efficiency_scalers, recombination_models, recombination_scalers = load_optimization_models()
    
    if not efficiency_models and not recombination_models:
        logging.error("No models loaded")
        return None, None
    
    predictions = {}
    
    # Make efficiency predictions
    for target_name, model in efficiency_models.items():
        try:
            scalers = efficiency_scalers[target_name]
            feature_scaler = scalers['feature_scaler']
            target_scaler = scalers['target_scaler']
            
            # Prepare features
            feature_names = list(feature_scaler.feature_names_in_)
            X = prepare_input_data(input_data, feature_names)
            
            # Scale features
            X_scaled = feature_scaler.transform(X)
            
            # Make prediction
            pred_scaled = model.predict(X_scaled)
            
            # Inverse transform to get original scale
            pred = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
            
            predictions[f'efficiency_{target_name}'] = pred[0]
            logging.info(f"Efficiency prediction for {target_name}: {pred[0]}")
            
        except Exception as e:
            logging.error(f"Error predicting {target_name}: {e}")
            predictions[f'efficiency_{target_name}'] = None
    
    # Make recombination predictions
    for target_name, model in recombination_models.items():
        try:
            scalers = recombination_scalers[target_name]
            feature_scaler = scalers['feature_scaler']
            target_scaler = scalers['target_scaler']
            
            # Prepare features
            feature_names = list(feature_scaler.feature_names_in_)
            X = prepare_input_data(input_data, feature_names)
            
            # Scale features
            X_scaled = feature_scaler.transform(X)
            
            # Make prediction
            pred_scaled = model.predict(X_scaled)
            
            # Inverse transform to get original scale
            pred = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
            
            predictions[f'recombination_{target_name}'] = pred[0]
            logging.info(f"Recombination prediction for {target_name}: {pred[0]}")
            
        except Exception as e:
            logging.error(f"Error predicting {target_name}: {e}")
            predictions[f'recombination_{target_name}'] = None
    
    return predictions, list(efficiency_models.keys()) + list(recombination_models.keys())

def validate_predictions_with_experimental_data():
    """Validate predictions against experimental data using models."""
    logging.info("\n=== Validating Predictions with Models ===")
    
    # Check if experimental data exists
    experimental_file = 'results/extract_simulation_data/extracted_simulation_data.csv'
    if not os.path.exists(experimental_file):
        logging.warning("No experimental data found for validation")
        return None
    
    # Load experimental data
    exp_data = pd.read_csv(experimental_file)
    
    # Calculate derived features for experimental data
    exp_data = calculate_derived_features(exp_data)
    
    # Handle missing values in experimental data
    numerical_cols = exp_data.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if exp_data[col].isnull().sum() > 0:
            median_val = exp_data[col].median()
            exp_data[col].fillna(median_val, inplace=True)
            logging.info(f"Filled {col} with median: {median_val}")
    
    # Load models
    efficiency_models, efficiency_scalers, recombination_models, recombination_scalers = load_optimization_models()
    
    predictions = {}
    
    # Validate efficiency models
    for target_name, model in efficiency_models.items():
        if target_name in exp_data.columns:
            try:
                scalers = efficiency_scalers[target_name]
                feature_scaler = scalers['feature_scaler']
                target_scaler = scalers['target_scaler']
                
                # Prepare features
                feature_names = list(feature_scaler.feature_names_in_)
                X_exp = prepare_input_data(exp_data, feature_names)
                
                # Scale features
                X_scaled = feature_scaler.transform(X_exp)
                
                # Make predictions
                pred_scaled = model.predict(X_scaled)
                pred = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
                
                # Calculate metrics
                actual = exp_data[target_name].values
                r2 = r2_score(actual, pred)
                rmse = np.sqrt(mean_squared_error(actual, pred))
                mae = mean_absolute_error(actual, pred)
                
                # Calculate relative error metrics
                actual_mean = np.mean(np.abs(actual))
                if actual_mean > 0:
                    rmse_relative = rmse / actual_mean * 100
                    mae_relative = mae / actual_mean * 100
                else:
                    rmse_relative = 0
                    mae_relative = 0
                
                # Calculate MAPE
                valid_mask = np.abs(actual) > 1e-10
                if np.sum(valid_mask) > 0:
                    mape = np.mean(np.abs((actual[valid_mask] - pred[valid_mask]) / np.abs(actual[valid_mask]))) * 100
                else:
                    mape = 0
                
                # Cap relative errors
                rmse_relative = min(rmse_relative, 1000)
                mae_relative = min(mae_relative, 1000)
                mape = min(mape, 1000)
                
                predictions[f'efficiency_{target_name}'] = {
                    'predictions': pred,
                    'actual': actual,
                    'r2': r2,
                    'rmse': rmse,
                    'mae': mae,
                    'rmse_relative': rmse_relative,
                    'mae_relative': mae_relative,
                    'mape': mape
                }
                
                logging.info(f"{target_name} - R²: {r2:.4f}, RMSE: {rmse:.4f} ({rmse_relative:.1f}%), MAE: {mae:.4f} ({mae_relative:.1f}%), MAPE: {mape:.1f}%")
                
                # Add efficiency calculation for MPP
                if target_name == 'MPP':
                    efficiency_metrics = calculate_efficiency_metrics(actual, pred)
                    predictions['efficiency_PCE'] = efficiency_metrics
                    logging.info(f"PCE Efficiency - R²: {efficiency_metrics['r2']:.4f}, RMSE: {efficiency_metrics['rmse']:.4f} ({efficiency_metrics['rmse_relative']:.1f}%), MAE: {efficiency_metrics['mae']:.4f} ({efficiency_metrics['mae_relative']:.1f}%), MAPE: {efficiency_metrics['mape']:.1f}%")
                
            except Exception as e:
                logging.error(f"Error validating {target_name}: {e}")
    
    # Validate recombination models
    for target_name, model in recombination_models.items():
        if target_name in exp_data.columns:
            try:
                scalers = recombination_scalers[target_name]
                feature_scaler = scalers['feature_scaler']
                target_scaler = scalers['target_scaler']
                
                # Prepare features
                feature_names = list(feature_scaler.feature_names_in_)
                X_exp = prepare_input_data(exp_data, feature_names)
                
                # Scale features
                X_scaled = feature_scaler.transform(X_exp)
                
                # Make predictions
                pred_scaled = model.predict(X_scaled)
                pred = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
                
                # Calculate metrics
                actual = exp_data[target_name].values
                r2 = r2_score(actual, pred)
                rmse = np.sqrt(mean_squared_error(actual, pred))
                mae = mean_absolute_error(actual, pred)
                
                # Calculate relative error metrics
                actual_mean = np.mean(np.abs(actual))
                if actual_mean > 0:
                    rmse_relative = rmse / actual_mean * 100
                    mae_relative = mae / actual_mean * 100
                else:
                    rmse_relative = 0
                    mae_relative = 0
                
                # Calculate MAPE
                valid_mask = np.abs(actual) > 1e-10
                if np.sum(valid_mask) > 0:
                    mape = np.mean(np.abs((actual[valid_mask] - pred[valid_mask]) / np.abs(actual[valid_mask]))) * 100
                else:
                    mape = 0
                
                # Cap relative errors
                rmse_relative = min(rmse_relative, 1000)
                mae_relative = min(mae_relative, 1000)
                mape = min(mape, 1000)
                
                predictions[f'recombination_{target_name}'] = {
                    'predictions': pred,
                    'actual': actual,
                    'r2': r2,
                    'rmse': rmse,
                    'mae': mae,
                    'rmse_relative': rmse_relative,
                    'mae_relative': mae_relative,
                    'mape': mape
                }
                
                logging.info(f"{target_name} - R²: {r2:.4f}, RMSE: {rmse:.4f} ({rmse_relative:.1f}%), MAE: {mae:.4f} ({mae_relative:.1f}%), MAPE: {mape:.1f}%")
                
            except Exception as e:
                logging.error(f"Error validating {target_name}: {e}")
    
    return predictions

def load_example_parameters():
    """Load example device parameters from configuration file."""
    config_file = 'example_device_parameters.json'
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Extract parameters and convert to DataFrame format
        params = config['parameters']
        example_data = pd.DataFrame({key: [value] for key, value in params.items()})
        
        logging.info(f"Loaded example parameters from {config_file}")
        logging.info(f"Device type: {config.get('device_type', 'Unknown')}")
        
        # Log parameter values for transparency
        for param, value in params.items():
            if 'L' in param and param.endswith('_L'):
                logging.info(f"  {param}: {value:.2e} m ({config['layer_descriptions'].get(param.split('_')[0], 'Unknown layer')})")
            elif 'E_' in param:
                logging.info(f"  {param}: {value:.2f} eV")
            elif 'N_' in param:
                logging.info(f"  {param}: {value:.2e} cm^-3")
        
        return example_data
        
    except FileNotFoundError:
        logging.warning(f"Configuration file {config_file} not found. Using default parameters.")
        return example_data
    except Exception as e:
        logging.error(f"Error loading example parameters: {e}. Using default parameters.")
        return example_data

def create_example_predictions():
    """Create example predictions for demonstration using models."""
    logging.info("\n=== Creating Example Predictions ===")
    
    # Load example parameters from configuration file
    example_data = load_example_parameters()
    
    try:
        predictions, target_names = make_predictions_all_models(example_data)
        
        # Calculate efficiency for example MPP prediction
        if 'efficiency_MPP' in predictions:
            mpp_prediction = predictions['efficiency_MPP']
            efficiency = calculate_efficiency(mpp_prediction)
            predictions['efficiency_PCE_example'] = efficiency
            logging.info(f"  efficiency_PCE_example: {efficiency:.4f}%")
        
        logging.info("Example predictions")
        for target, value in predictions.items():
            logging.info(f"  {target}: {value}")
        
        return predictions
        
    except Exception as e:
        logging.error(f"Error making example predictions: {e}")
        return None

def plot_predictions(predictions, save_dir='results/predict'):
    """Plot prediction results with metrics display."""
    if not predictions:
        logging.warning("No predictions to plot")
        return
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot validation results
    for target, data in predictions.items():
        if isinstance(data, dict) and 'actual' in data:
            plt.figure(figsize=(12, 10))
            
            # Create subplot layout
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[3, 1])
            
            # Main scatter plot
            scatter = ax1.scatter(data['actual'], data['predictions'], alpha=0.6, s=50)
            
            # Perfect prediction line
            min_val = min(data['actual'].min(), data['predictions'].min())
            max_val = max(data['actual'].max(), data['predictions'].max())
            ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2, label='Perfect Prediction')
            
            ax1.set_xlabel('Actual Values', fontsize=12)
            ax1.set_ylabel('Predicted Values', fontsize=12)
            ax1.set_title(f'{target} - Prediction vs Actual (Models)', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Add metrics text box with relative errors
            if 'rmse_relative' in data and 'mae_relative' in data:
                # Format large numbers in scientific notation for readability
                if abs(data["rmse"]) >= 1e6 or abs(data["mae"]) >= 1e6:
                    metrics_text = f'R² = {data["r2"]:.4f}\nRMSE = {data["rmse"]:.2e} ({data["rmse_relative"]:.1f}%)\nMAE = {data["mae"]:.2e} ({data["mae_relative"]:.1f}%)\nMAPE = {data["mape"]:.1f}%'
                else:
                    metrics_text = f'R² = {data["r2"]:.4f}\nRMSE = {data["rmse"]:.4f} ({data["rmse_relative"]:.1f}%)\nMAE = {data["mae"]:.4f} ({data["mae_relative"]:.1f}%)\nMAPE = {data["mape"]:.1f}%'
            else:
                # Format large numbers in scientific notation for readability
                if abs(data["rmse"]) >= 1e6 or abs(data["mae"]) >= 1e6:
                    metrics_text = f'R² = {data["r2"]:.4f}\nRMSE = {data["rmse"]:.2e}\nMAE = {data["mae"]:.2e}'
                else:
                    metrics_text = f'R² = {data["r2"]:.4f}\nRMSE = {data["rmse"]:.4f}\nMAE = {data["mae"]:.4f}'
            
            ax1.text(0.05, 0.95, metrics_text, 
                    transform=ax1.transAxes, fontsize=11,
                    verticalalignment='top', 
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
            
            # Residual plot
            residuals = data['predictions'] - data['actual']
            ax2.scatter(data['actual'], residuals, alpha=0.6, s=50, color='orange')
            ax2.axhline(y=0, color='red', linestyle='--', alpha=0.8)
            ax2.set_xlabel('Actual Values', fontsize=12)
            ax2.set_ylabel('Residuals (Predicted - Actual)', fontsize=12)
            ax2.set_title('Residual Plot', fontsize=12)
            ax2.grid(True, alpha=0.3)
            
            # Add residual statistics with proper formatting
            residual_std = np.std(residuals)
            residual_mean = np.mean(residuals)
            # Format large numbers in scientific notation for readability
            if abs(residual_mean) >= 1e6 or abs(residual_std) >= 1e6:
                residual_text = f'Mean Residual: {residual_mean:.2e}\nStd Residual: {residual_std:.2e}'
            else:
                residual_text = f'Mean Residual: {residual_mean:.4f}\nStd Residual: {residual_std:.4f}'
            ax2.text(0.05, 0.95, residual_text, 
                    transform=ax2.transAxes, fontsize=10,
                    verticalalignment='top', 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(f'{save_dir}/{target}_validation.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            if 'rmse_relative' in data and 'mae_relative' in data:
                logging.info(f"Validation plot saved for {target} with R²={data['r2']:.4f}, RMSE={data['rmse']:.4f} ({data['rmse_relative']:.1f}%), MAE={data['mae']:.4f} ({data['mae_relative']:.1f}%), MAPE={data['mape']:.1f}%")
            else:
                logging.info(f"Validation plot saved for {target} with R²={data['r2']:.4f}, RMSE={data['rmse']:.4f}, MAE={data['mae']:.4f}")
    
    # Create efficiency comparison plot if PCE data is available
    if 'efficiency_PCE' in predictions:
        pce_data = predictions['efficiency_PCE']
        plt.figure(figsize=(14, 10))
        
        # Create subplot layout for efficiency
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1])
        
        # Main efficiency scatter plot
        scatter = ax1.scatter(pce_data['actual_efficiency'], pce_data['predicted_efficiency'], 
                             alpha=0.6, s=50, color='green')
        
        # Perfect prediction line
        min_eff = min(pce_data['actual_efficiency'].min(), pce_data['predicted_efficiency'].min())
        max_eff = max(pce_data['actual_efficiency'].max(), pce_data['predicted_efficiency'].max())
        ax1.plot([min_eff, max_eff], [min_eff, max_eff], 'r--', alpha=0.8, linewidth=2, label='Perfect Prediction')
        
        ax1.set_xlabel('Actual Efficiency (%)', fontsize=12)
        ax1.set_ylabel('Predicted Efficiency (%)', fontsize=12)
        ax1.set_title('Power Conversion Efficiency (PCE) - Prediction vs Actual', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Add efficiency metrics text box
        eff_metrics_text = f'R² = {pce_data["r2"]:.4f}\nRMSE = {pce_data["rmse"]:.4f}% ({pce_data["rmse_relative"]:.1f}%)\nMAE = {pce_data["mae"]:.4f}% ({pce_data["mae_relative"]:.1f}%)\nMAPE = {pce_data["mape"]:.1f}%'
        ax1.text(0.05, 0.95, eff_metrics_text, 
                transform=ax1.transAxes, fontsize=11,
                verticalalignment='top', 
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
        
        # Efficiency residual plot
        eff_residuals = pce_data['predicted_efficiency'] - pce_data['actual_efficiency']
        ax2.scatter(pce_data['actual_efficiency'], eff_residuals, alpha=0.6, s=50, color='orange')
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.8)
        ax2.set_xlabel('Actual Efficiency (%)', fontsize=12)
        ax2.set_ylabel('Residuals (Predicted - Actual) (%)', fontsize=12)
        ax2.set_title('Efficiency Residual Plot', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Add efficiency residual statistics with proper formatting
        eff_residual_std = np.std(eff_residuals)
        eff_residual_mean = np.mean(eff_residuals)
        # Format efficiency residuals (usually not extremely large, but just in case)
        if abs(eff_residual_mean) >= 1e6 or abs(eff_residual_std) >= 1e6:
            eff_residual_text = f'Mean Residual: {eff_residual_mean:.2e}%\nStd Residual: {eff_residual_std:.2e}%'
        else:
            eff_residual_text = f'Mean Residual: {eff_residual_mean:.4f}%\nStd Residual: {eff_residual_std:.4f}%'
        ax2.text(0.05, 0.95, eff_residual_text, 
                transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/efficiency_PCE_validation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Efficiency plot saved with R²={pce_data['r2']:.4f}, RMSE={pce_data['rmse']:.4f}% ({pce_data['rmse_relative']:.1f}%), MAE={pce_data['mae']:.4f}% ({pce_data['mae_relative']:.1f}%), MAPE={pce_data['mape']:.1f}%")

def main():
    """Main function for prediction."""
    logging.info("Starting prediction process...")
    
    # Check prerequisites
    if not check_prerequisites():
        logging.error("Prerequisites not met. Exiting.")
        return
    
    ensure_predict_results_dir()
    
    # Create example predictions
    example_predictions = create_example_predictions()
    
    # Validate with experimental data
    validation_results = validate_predictions_with_experimental_data()
    
    # Plot results
    if validation_results:
        plot_predictions(validation_results)
    
    # Log comprehensive summary
    logging.info("\n=== PREDICTION SUMMARY ===")
    if example_predictions:
        logging.info(f"Example predictions created: {len(example_predictions)} targets")
        for target, value in example_predictions.items():
            logging.info(f"  {target}: {value}")
    
    if validation_results:
        logging.info(f"Validation completed: {len(validation_results)} targets")
        for target, data in validation_results.items():
            if isinstance(data, dict) and 'r2' in data:
                logging.info(f"  {target}: R²={data['r2']:.4f}, RMSE={data['rmse']:.4f} ({data['rmse_relative']:.1f}%), MAE={data['mae']:.4f} ({data['mae_relative']:.1f}%), MAPE={data['mape']:.1f}%")
    
    logging.info("Prediction process complete!")

if __name__ == "__main__":
    main() 