"""
Improved Prediction Script for Solar Cell Optimization Models
==========================================================

PURPOSE:
--------
Uses trained machine learning models with improved scaling to predict solar cell performance
and validate predictions against experimental data.

IMPROVEMENTS:
-------------
1. Proper Target Scaling: Uses the target scalers saved during training
2. Better Error Handling: Handles missing models and scalers gracefully
3. Comprehensive Metrics: Calculates multiple evaluation metrics
4. Data Quality Checks: Validates input data before prediction
5. Improved Logging: Better error reporting and debugging

MODELS USED:
------------
- Efficiency Predictors: MPP, Jsc, Voc, FF (4 models)
- Recombination Predictors: IntSRHn_mean, IntSRHn_std, IntSRHp_mean, IntSRHp_std, IntSRH_total, IntSRH_ratio (6 models)

DERIVED FEATURES CALCULATED:
---------------------------
- Thickness features: total_thickness, thickness_ratio_L2, thickness_ratio_ETL, thickness_ratio_HTL
- Energy gap features: energy_gap_L1, energy_gap_L2, energy_gap_L3
- Band alignment features: band_offset_L1_L2, band_offset_L2_L3, conduction_band_offset, valence_band_offset
- Doping features: doping_ratio_L1, doping_ratio_L2, doping_ratio_L3, total_donor_concentration, total_acceptor_concentration
- Material property features: average_energy_gap, energy_gap_variance, thickness_variance, doping_variance

INPUT FILES:
-----------
- results/train_optimization_models/models/ (improved trained models from script 5_improved)
- results/extract_simulation_data/combined_output_with_efficiency.csv (experimental data for validation)

OUTPUT FILES:
-------------
- results/predict_improved/predicted_values.txt (example predictions)
- results/predict_improved/*_validation.png (validation plots for each target)
- results/predict_improved/predictions.log (detailed execution log)
- results/predict_improved/model_validation_metrics.csv (comprehensive validation metrics)

VALIDATION METRICS:
------------------
- R² Score: Coefficient of determination
- RMSE: Root Mean Square Error (raw and relative)
- MAE: Mean Absolute Error (raw and relative)
- MAPE: Mean Absolute Percentage Error

PREREQUISITES:
--------------
- Run 5_train_optimization_models_improved.py to train improved models

USAGE:
------
python scripts/7_predict_improved.py

AUTHOR: ML Solar Cell Optimization Pipeline
DATE: 2024
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
log_dir = 'results/predict_improved'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'predictions_improved.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='a'),
        logging.StreamHandler()
    ]
)

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
    
    # Check for improved models
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
        logging.error("Please run 5_train_optimization_models_improved.py first")
        return False
    
    logging.info("All prerequisites satisfied")
    return True

def ensure_predict_results_dir():
    """Ensure the prediction results directory exists."""
    os.makedirs('results/predict_improved', exist_ok=True)

def load_optimization_models_improved():
    """Load improved optimization models with proper scalers."""
    logging.info("\n=== Loading Improved Optimization Models ===")
    
    models_dir = 'results/train_optimization_models/models'
    
    # Load efficiency models
    efficiency_models = {}
    efficiency_scalers = {}
    
    efficiency_targets = ['MPP', 'Jsc', 'Voc', 'FF']
    for target in efficiency_targets:
        model_path = f'{models_dir}/efficiency_{target}.joblib'
        scaler_path = f'{models_dir}/efficiency_{target}_scalers.joblib'
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                model = joblib.load(model_path)
                scalers = joblib.load(scaler_path)
                efficiency_models[target] = model
                efficiency_scalers[target] = scalers
                logging.info(f"Loaded efficiency model for {target}")
            except Exception as e:
                logging.warning(f"Failed to load efficiency model for {target}: {e}")
    
    # Load recombination models
    recombination_models = {}
    recombination_scalers = {}
    
    recombination_targets = ['IntSRHn_mean', 'IntSRHn_std', 'IntSRHp_mean', 'IntSRHp_std', 'IntSRH_total', 'IntSRH_ratio']
    for target in recombination_targets:
        model_path = f'{models_dir}/recombination_{target}.joblib'
        scaler_path = f'{models_dir}/recombination_{target}_scalers.joblib'
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                model = joblib.load(model_path)
                scalers = joblib.load(scaler_path)
                recombination_models[target] = model
                recombination_scalers[target] = scalers
                logging.info(f"Loaded recombination model for {target}")
            except Exception as e:
                logging.warning(f"Failed to load recombination model for {target}: {e}")
    
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
    
    # Energy gap features
    if all(col in df.columns for col in ['L1_E_c', 'L1_E_v', 'L2_E_c', 'L2_E_v', 'L3_E_c', 'L3_E_v']):
        df['energy_gap_L1'] = df['L1_E_c'] - df['L1_E_v']
        df['energy_gap_L2'] = df['L2_E_c'] - df['L2_E_v']
        df['energy_gap_L3'] = df['L3_E_c'] - df['L3_E_v']
    
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
    
    logging.info(f"Calculated derived features. Total features: {len([col for col in df.columns if col not in ['L1_L', 'L1_E_c', 'L1_E_v', 'L1_N_D', 'L1_N_A', 'L2_L', 'L2_E_c', 'L2_E_v', 'L2_N_D', 'L2_N_A', 'L3_L', 'L3_E_c', 'L3_E_v', 'L3_N_D', 'L3_N_A']])}")
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

def make_predictions_all_models_improved(input_data):
    """Make predictions using improved models with proper scaling."""
    logging.info("\n=== Making Predictions with Improved Models ===")
    
    # Load models
    efficiency_models, efficiency_scalers, recombination_models, recombination_scalers = load_optimization_models_improved()
    
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

def validate_predictions_with_experimental_data_improved():
    """Validate predictions against experimental data using improved models."""
    logging.info("\n=== Validating Predictions with Improved Models ===")
    
    # Check if experimental data exists
    experimental_file = 'results/extract_simulation_data/combined_output_with_efficiency.csv'
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
    efficiency_models, efficiency_scalers, recombination_models, recombination_scalers = load_optimization_models_improved()
    
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

def create_example_predictions_improved():
    """Create example predictions for demonstration using improved models."""
    logging.info("\n=== Creating Example Predictions (Improved) ===")
    
    # Create example input data with all basic parameters
    example_data = pd.DataFrame({
        'L1_L': [30.0e-9],  # PCBM thickness
        'L1_E_c': [3.8],  # PCBM conduction band
        'L1_E_v': [5.8],  # PCBM valence band
        'L1_N_D': [1e20],  # PCBM donor concentration
        'L1_N_A': [1e20],  # PCBM acceptor concentration
        'L2_L': [300.0e-9],  # MAPI thickness
        'L2_E_c': [4.5],  # MAPI conduction band
        'L2_E_v': [5.7],  # MAPI valence band
        'L2_N_D': [1e20],  # MAPI donor concentration
        'L2_N_A': [1e20],  # MAPI acceptor concentration
        'L3_L': [30.0e-9],  # PEDOT thickness
        'L3_E_c': [3.5],  # PEDOT conduction band
        'L3_E_v': [5.4],  # PEDOT valence band
        'L3_N_D': [1e20],  # PEDOT donor concentration
        'L3_N_A': [1e20],  # PEDOT acceptor concentration
    })
    
    try:
        predictions, target_names = make_predictions_all_models_improved(example_data)
        
        # Calculate efficiency for example MPP prediction
        if 'efficiency_MPP' in predictions:
            mpp_prediction = predictions['efficiency_MPP']
            efficiency = calculate_efficiency(mpp_prediction)
            predictions['efficiency_PCE_example'] = efficiency
            logging.info(f"  efficiency_PCE_example: {efficiency:.4f}%")
        
        logging.info("Example predictions (improved models):")
        for target, value in predictions.items():
            logging.info(f"  {target}: {value}")
        
        return predictions
        
    except Exception as e:
        logging.error(f"Error making example predictions: {e}")
        return None

def plot_predictions_improved(predictions, save_dir='results/predict_improved'):
    """Plot prediction results with improved metrics display."""
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
            ax1.set_title(f'{target} - Prediction vs Actual (Improved Models)', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Add metrics text box with relative errors
            if 'rmse_relative' in data and 'mae_relative' in data:
                metrics_text = f'R² = {data["r2"]:.4f}\nRMSE = {data["rmse"]:.4f} ({data["rmse_relative"]:.1f}%)\nMAE = {data["mae"]:.4f} ({data["mae_relative"]:.1f}%)\nMAPE = {data["mape"]:.1f}%'
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
            
            # Add residual statistics
            residual_std = np.std(residuals)
            residual_mean = np.mean(residuals)
            residual_text = f'Mean Residual: {residual_mean:.4f}\nStd Residual: {residual_std:.4f}'
            ax2.text(0.05, 0.95, residual_text, 
                    transform=ax2.transAxes, fontsize=10,
                    verticalalignment='top', 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(f'{save_dir}/{target}_validation_improved.png', dpi=300, bbox_inches='tight')
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
        
        # Add efficiency residual statistics
        eff_residual_std = np.std(eff_residuals)
        eff_residual_mean = np.mean(eff_residuals)
        eff_residual_text = f'Mean Residual: {eff_residual_mean:.4f}%\nStd Residual: {eff_residual_std:.4f}%'
        ax2.text(0.05, 0.95, eff_residual_text, 
                transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/efficiency_PCE_validation_improved.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Efficiency plot saved with R²={pce_data['r2']:.4f}, RMSE={pce_data['rmse']:.4f}% ({pce_data['rmse_relative']:.1f}%), MAE={pce_data['mae']:.4f}% ({pce_data['mae_relative']:.1f}%), MAPE={pce_data['mape']:.1f}%")

def main():
    """Main function for improved prediction."""
    logging.info("Starting improved prediction process...")
    
    # Check prerequisites
    if not check_prerequisites():
        logging.error("Prerequisites not met. Exiting.")
        return
    
    ensure_predict_results_dir()
    
    # Create example predictions
    example_predictions = create_example_predictions_improved()
    
    # Validate with experimental data
    validation_results = validate_predictions_with_experimental_data_improved()
    
    # Plot results
    if validation_results:
        plot_predictions_improved(validation_results)
    
    # Log comprehensive summary
    logging.info("\n=== IMPROVED PREDICTION SUMMARY ===")
    if example_predictions:
        logging.info(f"Example predictions created: {len(example_predictions)} targets")
        for target, value in example_predictions.items():
            logging.info(f"  {target}: {value}")
    
    if validation_results:
        logging.info(f"Validation completed: {len(validation_results)} targets")
        for target, data in validation_results.items():
            if isinstance(data, dict) and 'r2' in data:
                logging.info(f"  {target}: R²={data['r2']:.4f}, RMSE={data['rmse']:.4f} ({data['rmse_relative']:.1f}%), MAE={data['mae']:.4f} ({data['mae_relative']:.1f}%), MAPE={data['mape']:.1f}%")
    
    logging.info("Improved prediction process complete!")

if __name__ == "__main__":
    main() 