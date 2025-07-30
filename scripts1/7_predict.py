"""
===============================================================================
PREDICT SOLAR CELL PERFORMANCE USING TRAINED OPTIMIZATION MODELS
===============================================================================

PURPOSE:
This script uses trained machine learning models to predict solar cell performance
and validate predictions against experimental data. It demonstrates the practical
application of the trained models for real-world predictions.

WHAT THIS SCRIPT DOES:
1. Loads trained optimization models from script 5 (efficiency and recombination predictors)
2. Calculates derived features from basic device parameters (same as script 4)
3. Makes predictions on example data and experimental data
4. Validates predictions against experimental data with comprehensive metrics
5. Creates validation plots showing prediction vs actual performance
6. Saves prediction results and validation metrics

MODELS USED:
- Efficiency Predictors: MPP, Jsc, Voc, FF (4 models)
- Recombination Predictors: IntSRHn_mean, IntSRHn_std, IntSRHp_mean, IntSRHp_std, IntSRH_total, IntSRH_ratio (6 models)

DERIVED FEATURES CALCULATED:
- Thickness features: total_thickness, thickness_ratio_L2, thickness_ratio_ETL, thickness_ratio_HTL
- Energy gap features: energy_gap_L1, energy_gap_L2, energy_gap_L3
- Band alignment features: band_offset_L1_L2, band_offset_L2_L3, conduction_band_offset, valence_band_offset
- Doping features: doping_ratio_L1, doping_ratio_L2, doping_ratio_L3, total_donor_concentration, total_acceptor_concentration
- Material property features: average_energy_gap, energy_gap_variance, thickness_variance, doping_variance

INPUT FILES:
- results/train_optimization_models/models/ (trained models from script 5)
- results/extract_simulation_data/combined_output_with_efficiency.csv (experimental data for validation)

OUTPUT FILES:
- results/predict/predicted_values.txt (example predictions)
- results/predict/*_validation.png (validation plots for each target)
- results/predict/predictions.log (detailed execution log)
- results/predict/model_validation_metrics.csv (comprehensive validation metrics)

VALIDATION METRICS:
- R² Score: Coefficient of determination
- RMSE: Root Mean Square Error
- MAE: Mean Absolute Error

PREREQUISITES:
- Run 1_create_feature_names.py to define feature structure
- Run 2_generate_simulations_enhanced.py to generate simulation data
- Run 3_extract_simulation_data.py to extract simulation results
- Run 4_prepare_ml_data.py to prepare ML datasets
- Run 5_train_optimization_models.py to train prediction models

USAGE:
python scripts/7_predict.py

AUTHOR: ANTHONY BARKER
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

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up logging
log_dir = 'results/predict'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'predictions.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='a'),
        logging.StreamHandler()
    ]
)

def check_prerequisites():
    """Check if all required files and directories exist."""
    logging.info("\n=== Checking Prerequisites ===")
    
    required_files = [
        'results/train_optimization_models/models/metadata.json',
        'results/train_optimization_models/models/efficiency_MPP.joblib',
        'results/train_optimization_models/models/recombination_IntSRHn_mean.joblib'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        logging.error("Missing required files:")
        for file_path in missing_files:
            logging.error(f"  - {file_path}")
        logging.error("Run script 5_train_optimization_models.py first to train the models.")
        return False
    
    logging.info("All prerequisites satisfied")
    return True

def ensure_predict_results_dir():
    os.makedirs('results/predict', exist_ok=True)

def load_optimization_models():
    """Load the trained optimization models from script 5."""
    models_dir = 'results/train_optimization_models/models'
    
    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Models directory not found: {models_dir}")
    
    # Load metadata
    metadata_path = os.path.join(models_dir, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Load efficiency models
    efficiency_models = {}
    efficiency_scalers = {}
    for target in metadata['efficiency_targets']:
        model_path = os.path.join(models_dir, f'efficiency_{target}.joblib')
        scaler_path = os.path.join(models_dir, f'efficiency_{target}_scaler.joblib')
        
        if os.path.exists(model_path):
            efficiency_models[target] = joblib.load(model_path)
            efficiency_scalers[target] = joblib.load(scaler_path)
    
    # Load recombination models
    recombination_models = {}
    recombination_scalers = {}
    for target in metadata['recombination_targets']:
        model_path = os.path.join(models_dir, f'recombination_{target}.joblib')
        scaler_path = os.path.join(models_dir, f'recombination_{target}_scaler.joblib')
        
        if os.path.exists(model_path):
            recombination_models[target] = joblib.load(model_path)
            recombination_scalers[target] = joblib.load(scaler_path)
    
    logging.info(f"Loaded {len(efficiency_models)} efficiency models")
    logging.info(f"Loaded {len(recombination_models)} recombination models")
    
    return efficiency_models, efficiency_scalers, recombination_models, recombination_scalers, metadata

def calculate_derived_features(df):
    """Calculate derived features from basic device parameters."""
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
    
    logging.info(f"Calculated derived features. Total features: {len(df.columns)}")
    return df

def prepare_input_data(input_data, feature_names):
    """Prepare input data for prediction."""
    # Calculate derived features first
    input_data = calculate_derived_features(input_data)
    
    # Ensure all required features are present
    missing_features = set(feature_names) - set(input_data.columns)
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Select features in the correct order
    X = input_data[feature_names].values
    
    # Create a DataFrame with the correct feature names
    X_df = pd.DataFrame(X, columns=feature_names)
    
    logging.debug(f"Input data shape: {X_df.shape}")
    
    return X_df

def make_predictions_all_models(input_data):
    """Make predictions using all trained optimization models."""
    ensure_predict_results_dir()
    
    # Load models and metadata
    efficiency_models, efficiency_scalers, recombination_models, recombination_scalers, metadata = load_optimization_models()
    
    # Get feature names from metadata
    feature_names = metadata['device_params']
    
    # Prepare input data
    X_df = prepare_input_data(input_data, feature_names)
    
    # Dictionary to store predictions
    predictions = {}
    
    # Make efficiency predictions
    for target_name in efficiency_models.keys():
        try:
            model = efficiency_models[target_name]
            scaler = efficiency_scalers[target_name]
            
            # Scale input data
            X_scaled = scaler.transform(X_df)
            
            # Make prediction
            prediction = model.predict(X_scaled)
            
            predictions[f'efficiency_{target_name}'] = prediction
            logging.info(f"Efficiency prediction for {target_name}: {prediction}")
            
        except Exception as e:
            logging.error(f"Error predicting {target_name}: {e}")
    
    # Make recombination predictions
    for target_name in recombination_models.keys():
        try:
            model = recombination_models[target_name]
            scaler = recombination_scalers[target_name]
            
            # Scale input data
            X_scaled = scaler.transform(X_df)
            
            # Make prediction
            prediction = model.predict(X_scaled)
            
            predictions[f'recombination_{target_name}'] = prediction
            logging.info(f"Recombination prediction for {target_name}: {prediction}")
            
        except Exception as e:
            logging.error(f"Error predicting {target_name}: {e}")
    
    return predictions, feature_names

def validate_predictions_with_experimental_data():
    """Validate predictions against experimental data if available."""
    logging.info("\n=== Validating Predictions ===")
    
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
    efficiency_models, efficiency_scalers, recombination_models, recombination_scalers, metadata = load_optimization_models()
    
    # Prepare features
    feature_names = metadata['device_params']
    X_exp = exp_data[feature_names]
    
    # Make predictions on experimental data
    predictions = {}
    for target_name in efficiency_models.keys():
        if target_name in exp_data.columns:
            model = efficiency_models[target_name]
            scaler = efficiency_scalers[target_name]
            
            X_scaled = scaler.transform(X_exp)
            pred = model.predict(X_scaled)
            
            # Calculate metrics
            actual = exp_data[target_name].values
            r2 = r2_score(actual, pred)
            rmse = np.sqrt(mean_squared_error(actual, pred))
            mae = mean_absolute_error(actual, pred)
            
            # Calculate relative error metrics (more meaningful for different scales)
            actual_mean = np.mean(np.abs(actual))
            if actual_mean > 0:
                rmse_relative = rmse / actual_mean * 100  # Percentage
                mae_relative = mae / actual_mean * 100   # Percentage
            else:
                rmse_relative = 0
                mae_relative = 0
            
            # Calculate MAPE (Mean Absolute Percentage Error) with better handling of small values
            # Filter out very small actual values to avoid division by near-zero
            valid_mask = np.abs(actual) > 1e-10
            if np.sum(valid_mask) > 0:
                mape = np.mean(np.abs((actual[valid_mask] - pred[valid_mask]) / np.abs(actual[valid_mask]))) * 100
            else:
                mape = 0
            
            # Cap relative errors at 1000% to avoid extreme values
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
    
    for target_name in recombination_models.keys():
        if target_name in exp_data.columns:
            model = recombination_models[target_name]
            scaler = recombination_scalers[target_name]
            
            X_scaled = scaler.transform(X_exp)
            pred = model.predict(X_scaled)
            
            # Calculate metrics
            actual = exp_data[target_name].values
            r2 = r2_score(actual, pred)
            rmse = np.sqrt(mean_squared_error(actual, pred))
            mae = mean_absolute_error(actual, pred)
            
            # Calculate relative error metrics (more meaningful for different scales)
            actual_mean = np.mean(np.abs(actual))
            if actual_mean > 0:
                rmse_relative = rmse / actual_mean * 100  # Percentage
                mae_relative = mae / actual_mean * 100   # Percentage
            else:
                rmse_relative = 0
                mae_relative = 0
            
            # Calculate MAPE (Mean Absolute Percentage Error) with better handling of small values
            # Filter out very small actual values to avoid division by near-zero
            valid_mask = np.abs(actual) > 1e-10
            if np.sum(valid_mask) > 0:
                mape = np.mean(np.abs((actual[valid_mask] - pred[valid_mask]) / np.abs(actual[valid_mask]))) * 100
            else:
                mape = 0
            
            # Cap relative errors at 1000% to avoid extreme values
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
    
    return predictions

def write_predictions_to_file(predictions, filename='results/predict/predicted_values.txt'):
    """Write predictions to a file."""
    with open(filename, 'w') as f:
        f.write("=== PREDICTION RESULTS ===\n\n")
        for target, values in predictions.items():
            if isinstance(values, dict):
                f.write(f"{target}:\n")
                f.write(f"  R² Score: {values['r2']:.4f}\n")
                f.write(f"  RMSE: {values['rmse']:.4f}")
                if 'rmse_relative' in values:
                    f.write(f" ({values['rmse_relative']:.1f}% relative)\n")
                else:
                    f.write("\n")
                f.write(f"  MAE: {values['mae']:.4f}")
                if 'mae_relative' in values:
                    f.write(f" ({values['mae_relative']:.1f}% relative)\n")
                else:
                    f.write("\n")
                if 'mape' in values:
                    f.write(f"  MAPE: {values['mape']:.1f}%\n")
                if 'predictions' in values and 'actual' in values:
                    f.write(f"  Predictions Range: [{values['predictions'].min():.4f}, {values['predictions'].max():.4f}]\n")
                    f.write(f"  Actual Range: [{values['actual'].min():.4f}, {values['actual'].max():.4f}]\n")
                f.write("\n")
            else:
                f.write(f"{target}: {values}\n")
    
    logging.info(f"Predictions written to {filename}")

def plot_predictions(predictions, save_dir='results/predict'):
    """Plot prediction results."""
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
            ax1.set_title(f'{target} - Prediction vs Actual', fontsize=14, fontweight='bold')
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
            plt.savefig(f'{save_dir}/{target}_validation.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            if 'rmse_relative' in data and 'mae_relative' in data:
                logging.info(f"Validation plot saved for {target} with R²={data['r2']:.4f}, RMSE={data['rmse']:.4f} ({data['rmse_relative']:.1f}%), MAE={data['mae']:.4f} ({data['mae_relative']:.1f}%), MAPE={data['mape']:.1f}%")
            else:
                logging.info(f"Validation plot saved for {target} with R²={data['r2']:.4f}, RMSE={data['rmse']:.4f}, MAE={data['mae']:.4f}")

def create_example_predictions():
    """Create example predictions for demonstration."""
    logging.info("\n=== Creating Example Predictions ===")
    
    # Create example input data with all basic parameters
    example_data = pd.DataFrame({
        'L1_L': [30.0],  # PCBM thickness
        'L1_E_c': [3.8],  # PCBM conduction band
        'L1_E_v': [5.8],  # PCBM valence band
        'L1_N_D': [1e20],  # PCBM donor concentration
        'L1_N_A': [1e20],  # PCBM acceptor concentration
        'L2_L': [300.0],  # MAPI thickness
        'L2_E_c': [4.5],  # MAPI conduction band
        'L2_E_v': [5.7],  # MAPI valence band
        'L2_N_D': [1e20],  # MAPI donor concentration
        'L2_N_A': [1e20],  # MAPI acceptor concentration
        'L3_L': [30.0],  # PEDOT thickness
        'L3_E_c': [3.5],  # PEDOT conduction band
        'L3_E_v': [5.4],  # PEDOT valence band
        'L3_N_D': [1e20],  # PEDOT donor concentration
        'L3_N_A': [1e20],  # PEDOT acceptor concentration
    })
    
    try:
        predictions, feature_names = make_predictions_all_models(example_data)
        
        logging.info("Example predictions:")
        for target, value in predictions.items():
            logging.info(f"  {target}: {value}")
        
        return predictions
        
    except Exception as e:
        logging.error(f"Error making example predictions: {e}")
        return None

def save_prediction_metadata(example_predictions, validation_results):
    """Save comprehensive metadata about prediction results."""
    logging.info("\n=== Saving Prediction Metadata ===")
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    metadata = {
        'prediction_date': datetime.now().isoformat(),
        'example_predictions': convert_numpy(example_predictions) if example_predictions else {},
        'validation_summary': {},
        'model_performance': {},
        'data_statistics': {}
    }
    
    # Save validation metrics to CSV for easy analysis
    if validation_results:
        metrics_data = []
        for target, data in validation_results.items():
            if isinstance(data, dict) and 'r2' in data:
                metrics_data.append({
                    'Target': target,
                    'R²': data['r2'],
                    'RMSE': data['rmse'],
                    'RMSE_Relative_Percent': data.get('rmse_relative', 0),
                    'MAE': data['mae'],
                    'MAE_Relative_Percent': data.get('mae_relative', 0),
                    'MAPE_Percent': data.get('mape', 0),
                    'Predictions_Mean': np.mean(data['predictions']),
                    'Predictions_Std': np.std(data['predictions']),
                    'Actual_Mean': np.mean(data['actual']),
                    'Actual_Std': np.std(data['actual']),
                    'Predictions_Min': np.min(data['predictions']),
                    'Predictions_Max': np.max(data['predictions']),
                    'Actual_Min': np.min(data['actual']),
                    'Actual_Max': np.max(data['actual'])
                })
        
        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            metrics_csv_path = 'results/predict/model_validation_metrics.csv'
            metrics_df.to_csv(metrics_csv_path, index=False)
            logging.info(f"Validation metrics saved to {metrics_csv_path}")
            
            # Add summary statistics to metadata
            metadata['model_performance'] = {
                'mean_r2': metrics_df['R²'].mean(),
                'max_r2': metrics_df['R²'].max(),
                'min_r2': metrics_df['R²'].min(),
                'mean_rmse': metrics_df['RMSE'].mean(),
                'mean_rmse_relative': metrics_df['RMSE_Relative_Percent'].mean(),
                'mean_mae': metrics_df['MAE'].mean(),
                'mean_mae_relative': metrics_df['MAE_Relative_Percent'].mean(),
                'mean_mape': metrics_df['MAPE_Percent'].mean(),
                'total_targets': len(metrics_df)
            }
    
    # Add validation summary if available
    if validation_results:
        for target, data in validation_results.items():
            if isinstance(data, dict) and 'r2' in data:
                metadata['validation_summary'][target] = {
                    'r2_score': float(data['r2']),
                    'rmse': float(data['rmse']),
                    'mae': float(data['mae']),
                    'sample_count': int(len(data['actual']))
                }
    
    # Calculate overall performance statistics
    if metadata['validation_summary']:
        r2_scores = [data['r2_score'] for data in metadata['validation_summary'].values()]
        metadata['model_performance'] = {
            'mean_r2': float(np.mean(r2_scores)),
            'max_r2': float(np.max(r2_scores)),
            'min_r2': float(np.min(r2_scores)),
            'std_r2': float(np.std(r2_scores)),
            'total_targets': int(len(metadata['validation_summary']))
        }
    
    # Save metadata
    metadata_path = 'results/predict/prediction_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logging.info(f"Prediction metadata saved to: {metadata_path}")
    return metadata

def main():
    """Main function for prediction."""
    logging.info("Starting prediction process...")
    
    # Check prerequisites
    if not check_prerequisites():
        logging.error("Prerequisites not met. Exiting.")
        return
    
    # Create example predictions
    example_predictions = create_example_predictions()
    
    # Validate with experimental data
    validation_results = validate_predictions_with_experimental_data()
    
    # Write results
    if example_predictions:
        write_predictions_to_file(example_predictions)
    
    # Plot results
    if validation_results:
        plot_predictions(validation_results)
    
    # Save metadata
    metadata = save_prediction_metadata(example_predictions, validation_results)
    
    # Log comprehensive summary
    logging.info("\n=== PREDICTION SUMMARY ===")
    if example_predictions:
        logging.info(f"Example predictions created: {len(example_predictions)} targets")
        for target, value in example_predictions.items():
            logging.info(f"  {target}: {value}")
    
    if validation_results:
        logging.info(f"Validation completed: {len(validation_results)} targets")
        if metadata.get('model_performance'):
            perf = metadata['model_performance']
            logging.info(f"Model Performance Summary:")
            logging.info(f"  Mean R²: {perf['mean_r2']:.4f}")
            logging.info(f"  Max R²: {perf['max_r2']:.4f}")
            logging.info(f"  Min R²: {perf['min_r2']:.4f}")
            logging.info(f"  Total targets: {perf['total_targets']}")
    
    logging.info("Prediction process complete!")

if __name__ == "__main__":
    main() 