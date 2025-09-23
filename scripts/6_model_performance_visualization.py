"""
Script 6: Model Performance Visualization

This script creates comprehensive visualizations of trained ML model performance,
validation metrics, prediction accuracy, and training analysis.

WORKFLOW:
1. Load trained models and metadata from Script 5
2. Validate model predictions against simulation dataset
3. Create comprehensive performance visualizations
4. Generate model comparison charts and accuracy analysis
5. Build interactive dashboard of model performance
6. Save visualization reports and performance summaries

USAGE: python scripts/6_model_performance_visualization.py

REQUIREMENTS:
- Trained models from Script 5 (results/5_train_optimization_models/)
- Simulation data from Script 3 (results/3_extract_simulation_data/)

OUTPUT:
- results/6_model_performance/
  ├── model_performance_log.txt          # Detailed execution log
  ├── comprehensive_dashboard.png        # Complete performance dashboard
  ├── model_comparison.png               # Algorithm comparison charts
  ├── prediction_accuracy.png            # Prediction vs actual scatter plots
  ├── error_analysis.png                 # Error distribution and residual analysis
  ├── training_performance.png           # Training metrics from Script 5
  ├── cross_validation_analysis.png      # CV stability and variance analysis
  └── performance_summary.json           # Complete performance metrics
"""

import os
import sys
import json
import logging
import warnings
from datetime import datetime
import numpy as np
import pandas as pd

def setup_logging():
    """Set up logging configuration."""
    # Create results directory
    results_dir = 'results/6_model_performance'
    os.makedirs(results_dir, exist_ok=True)
    
    # Clear any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Set up logging
    log_filename = f'{results_dir}/model_performance_log.txt'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Suppress sklearn warnings
    warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
    
    return log_filename

def load_trained_models():
    """Load trained models and metadata from Script 5."""
    models_dir = 'results/5_train_optimization_models'
    
    # Load models
    import joblib
    
    models_data = {}
    
    # Load efficiency models
    mpp_model_path = f'{models_dir}/models/efficiency_MPP.joblib'
    mpp_scaler_path = f'{models_dir}/models/efficiency_MPP_scalers.joblib'
    
    if os.path.exists(mpp_model_path) and os.path.exists(mpp_scaler_path):
        models_data['mpp_model'] = joblib.load(mpp_model_path)
        models_data['mpp_scalers'] = joblib.load(mpp_scaler_path)
        logging.info(f"Loaded MPP model and scalers")
    
    # Load recombination models
    recomb_model_path = f'{models_dir}/models/recombination_IntSRHn_mean.joblib'
    recomb_scaler_path = f'{models_dir}/models/recombination_IntSRHn_mean_scalers.joblib'
    
    if os.path.exists(recomb_model_path) and os.path.exists(recomb_scaler_path):
        models_data['recomb_model'] = joblib.load(recomb_model_path)
        models_data['recomb_scalers'] = joblib.load(recomb_scaler_path)
        logging.info(f"Loaded recombination model and scalers")
    
    # Load training metadata
    metadata_path = f'{models_dir}/training_metadata.json'
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            models_data['metadata'] = json.load(f)
        logging.info("Loaded training metadata")
    
    return models_data

def load_simulation_data():
    """Load simulation data for validation."""
    data_path = 'results/3_extract_simulation_data/extracted_simulation_data.csv'
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Simulation data not found: {data_path}")
    
    df = pd.read_csv(data_path)
    logging.info(f"Loaded simulation data: {len(df)} samples, {len(df.columns)} features")
    
    return df

def calculate_derived_features(df):
    """Calculate derived features to match training data."""
    logging.info("Calculating derived features for validation...")
    
    # Basic thickness features
    if all(col in df.columns for col in ['L1_L', 'L2_L', 'L3_L']):
        total_thickness = df['L1_L'] + df['L2_L'] + df['L3_L']
        df['thickness_ratio_L2'] = df['L2_L'] / total_thickness
        df['thickness_ratio_ETL'] = df['L1_L'] / total_thickness
        df['thickness_ratio_HTL'] = df['L3_L'] / total_thickness
    
    # Energy gap features
    if all(col in df.columns for col in ['L1_E_c', 'L1_E_v', 'L2_E_c', 'L2_E_v', 'L3_E_c', 'L3_E_v']):
        df['energy_gap_L1'] = abs(df['L1_E_c'] - df['L1_E_v'])
        df['energy_gap_L2'] = abs(df['L2_E_c'] - df['L2_E_v'])
        df['energy_gap_L3'] = abs(df['L3_E_c'] - df['L3_E_v'])
    
    # Band alignment features
    if all(col in df.columns for col in ['L1_E_c', 'L2_E_c', 'L3_E_c', 'L1_E_v', 'L2_E_v', 'L3_E_v']):
        df['band_offset_L1_L2'] = df['L2_E_c'] - df['L1_E_c']
        df['band_offset_L2_L3'] = df['L3_E_c'] - df['L2_E_c']
    
    # Doping features
    if all(col in df.columns for col in ['L1_N_D', 'L1_N_A', 'L2_N_D', 'L2_N_A', 'L3_N_D', 'L3_N_A']):
        df['doping_ratio_L1'] = df['L1_N_D'] / (df['L1_N_A'] + 1e-30)
        df['doping_ratio_L2'] = df['L2_N_D'] / (df['L2_N_A'] + 1e-30)
        df['doping_ratio_L3'] = df['L3_N_D'] / (df['L3_N_A'] + 1e-30)
        df['total_donor_concentration'] = df['L1_N_D'] + df['L2_N_D'] + df['L3_N_D']
        df['total_acceptor_concentration'] = df['L1_N_A'] + df['L2_N_A'] + df['L3_N_A']
    
    # Enhanced physics-based features
    if 'MPP' in df.columns and 'IntSRHn_mean' in df.columns:
        df['recombination_efficiency_ratio'] = df['IntSRHn_mean'] / (df['MPP'] + 1e-30)
        df['interface_quality_index'] = df['MPP'] / (df['IntSRHn_mean'] + 1e-30)
    else:
        df['recombination_efficiency_ratio'] = 1e28
        df['interface_quality_index'] = 1e-28
    
    # Carrier transport efficiency features
    if all(col in df.columns for col in ['band_offset_L1_L2', 'band_offset_L2_L3']):
        df['conduction_band_alignment_quality'] = 1 / (1 + abs(df['band_offset_L1_L2']) + abs(df['band_offset_L2_L3']))
    
    # Ensure valence_band_alignment_quality exists
    if 'valence_band_alignment_quality' not in df.columns:
        df['valence_band_alignment_quality'] = 0.5
    
    # Thickness optimization features
    if all(col in df.columns for col in ['thickness_ratio_L2', 'thickness_ratio_ETL', 'thickness_ratio_HTL']):
        df['thickness_balance_quality'] = df['thickness_ratio_L2'] / (df['thickness_ratio_ETL'] + df['thickness_ratio_HTL'] + 1e-30)
        df['transport_layer_balance'] = 1 / (1 + abs(df['thickness_ratio_ETL'] - df['thickness_ratio_HTL']))
    
    # Doping optimization features
    if all(col in df.columns for col in ['doping_ratio_L1', 'doping_ratio_L2', 'doping_ratio_L3']):
        df['average_doping_ratio'] = df[['doping_ratio_L1', 'doping_ratio_L2', 'doping_ratio_L3']].mean(axis=1)
        df['doping_consistency'] = 1 / (1 + df[['doping_ratio_L1', 'doping_ratio_L2', 'doping_ratio_L3']].var(axis=1))
    
    # Energy level optimization features
    if all(col in df.columns for col in ['energy_gap_L1', 'energy_gap_L2', 'energy_gap_L3']):
        df['energy_gap_progression'] = abs((df['energy_gap_L2'] - df['energy_gap_L1']) * (df['energy_gap_L3'] - df['energy_gap_L2']))
        df['energy_gap_uniformity'] = 1 / (1 + df[['energy_gap_L1', 'energy_gap_L2', 'energy_gap_L3']].var(axis=1))
    
    logging.info(f"Calculated derived features. Total features: {len(df.columns)}")
    return df

def validate_model_performance(models_data, validation_data):
    """Replicate the EXACT same validation approach as Script 5 training."""
    logging.info("Replicating Script 5 training validation approach...")
    
    validation_results = {}
    
    # Load the exact same data that Script 5 uses (X_full.csv, y_efficiency_full.csv, y_recombination_full.csv)
    try:
        X_full = pd.read_csv('results/4_prepare_ml_data/X_full.csv')
        y_efficiency_full = pd.read_csv('results/4_prepare_ml_data/y_efficiency_full.csv')
        y_recombination_full = pd.read_csv('results/4_prepare_ml_data/y_recombination_full.csv')
        
        logging.info(f"Loaded Script 5 data: X={X_full.shape}, y_eff={y_efficiency_full.shape}, y_rec={y_recombination_full.shape}")
        
        # Replicate the exact same train/test split as Script 5
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        
        # MPP model validation (replicate Script 5 efficiency training exactly)
        if 'mpp_model' in models_data and 'MPP' in y_efficiency_full.columns:
            # Use the exact same split as Script 5 (random_state=42, test_size=0.2)
            X_train, X_test, y_train, y_test = train_test_split(
                X_full, y_efficiency_full, test_size=0.2, random_state=42
            )
            
            model = models_data['mpp_model']
            scalers = models_data['mpp_scalers']
            feature_scaler = scalers['feature_scaler']
            target_scaler = scalers['target_scaler']
            
            # Apply the exact same scaling as Script 5
            X_test_scaled = feature_scaler.transform(X_test)
            y_pred_scaled = model.predict(X_test_scaled)
            
            # Scale the test targets exactly like Script 5 does for evaluation
            y_test_scaled = target_scaler.transform(y_test[['MPP']]).flatten()
            
            # Calculate metrics on ORIGINAL targets (fixed like Script 5)
            y_pred_original = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            y_test_original = y_test['MPP'].values
            
            r2 = r2_score(y_test_original, y_pred_original)
            rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
            mae = mean_absolute_error(y_test_original, y_pred_original)
            
            # Calculate MAPE with improved handling
            valid_mask = np.abs(y_test_original) > 1e-10
            if np.sum(valid_mask) > 0:
                relative_errors = np.abs((y_test_original[valid_mask] - y_pred_original[valid_mask]) / np.abs(y_test_original[valid_mask]))
                relative_errors = np.minimum(relative_errors, 10.0)  # Cap at 1000% error
                mape = np.mean(relative_errors) * 100
            else:
                mape = 0
            
            # For plotting, convert back to unscaled for interpretability
            y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            y_true = y_test['MPP'].values
            
            # Get training metadata for comparison
            metadata = models_data.get('metadata', {})
            training_metrics = {}
            if 'efficiency_models' in metadata and 'MPP' in metadata['efficiency_models']:
                best_model_name = metadata['efficiency_models']['MPP'].get('best_model', 'Unknown')
                if 'all_scores' in metadata['efficiency_models']['MPP'] and best_model_name in metadata['efficiency_models']['MPP']['all_scores']:
                    training_metrics = metadata['efficiency_models']['MPP']['all_scores'][best_model_name]
            
            validation_results['MPP'] = {
                'r2': r2,
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'cv_mean': training_metrics.get('cv_mean', 0),
                'cv_std': training_metrics.get('cv_std', 0),
                'best_algorithm': best_model_name,
                'y_true': y_true,
                'y_pred': y_pred,
                'n_samples': len(y_true)
            }
            
            logging.info(f"MPP Model ({best_model_name}) - R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f} ({len(y_true)} samples)")
            
            # Compare with training metadata
            training_r2 = training_metrics.get('test_metrics', {}).get('r2', 0)
            logging.info(f"  Training metadata R²: {training_r2:.4f} (should match: {abs(r2 - training_r2) < 0.001})")
        
        # Recombination model validation (replicate Script 5 recombination training exactly)
        if 'recomb_model' in models_data and 'IntSRHn_mean' in y_recombination_full.columns:
            # Use the exact same split as Script 5 (random_state=42, test_size=0.2)
            X_train, X_test, y_train, y_test = train_test_split(
                X_full, y_recombination_full, test_size=0.2, random_state=42
            )
            
            model = models_data['recomb_model']
            scalers = models_data['recomb_scalers']
            feature_scaler = scalers['feature_scaler']
            target_scaler = scalers['target_scaler']
            
            # Apply the exact same scaling as Script 5 (log-transformed approach)
            X_test_scaled = feature_scaler.transform(X_test)
            y_pred_scaled = model.predict(X_test_scaled)
            
            # CRITICAL FIX: Use the EXACT same evaluation as Script 5
            # Copy the exact evaluation logic from Script 5
            y_pred_log = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            y_pred_original = 10 ** y_pred_log
            y_true = y_test['IntSRHn_mean'].values
            
            # Calculate metrics on original scale (exactly like Script 5)
            r2 = r2_score(y_true, y_pred_original)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred_original))
            mae = mean_absolute_error(y_true, y_pred_original)
            
            # Calculate relative errors
            y_mean = np.mean(np.abs(y_true))
            rmse_relative = rmse / y_mean * 100
            mae_relative = mae / y_mean * 100
            
            # Calculate MAPE with improved handling (exactly like Script 5)
            valid_mask = np.abs(y_true) > 1e-10
            if np.sum(valid_mask) > 0:
                relative_errors = np.abs((y_true[valid_mask] - y_pred_original[valid_mask]) / np.abs(y_true[valid_mask]))
                relative_errors = np.minimum(relative_errors, 10.0)  # Cap at 1000% error
                mape = np.mean(relative_errors) * 100
            else:
                mape = 0
            
            # y_pred_original and y_true are already calculated above
            
            # Get training metadata for comparison
            training_metrics = {}
            if 'recombination_models' in metadata and 'IntSRHn_mean' in metadata['recombination_models']:
                best_model_name = metadata['recombination_models']['IntSRHn_mean'].get('best_model', 'Unknown')
                if 'all_scores' in metadata['recombination_models']['IntSRHn_mean'] and best_model_name in metadata['recombination_models']['IntSRHn_mean']['all_scores']:
                    training_metrics = metadata['recombination_models']['IntSRHn_mean']['all_scores'][best_model_name]
            
            validation_results['IntSRHn_mean'] = {
                'r2': r2,
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'cv_mean': training_metrics.get('cv_mean', 0),
                'cv_std': training_metrics.get('cv_std', 0),
                'best_algorithm': best_model_name,
                'y_true': y_true,
                'y_pred': y_pred_original,
                'n_samples': len(y_true)
            }
            
            logging.info(f"Recombination Model ({best_model_name}) - R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f} ({len(y_true)} samples)")
            
            # Compare with training metadata
            training_r2 = training_metrics.get('test_metrics', {}).get('r2', 0)
            logging.info(f"  Training metadata R²: {training_r2:.4f} (should match: {abs(r2 - training_r2) < 0.001})")
        
    except Exception as e:
        logging.error(f"Error replicating Script 5 validation: {e}")
        return {}
    
    return validation_results

def create_separate_visualizations(models_data, validation_results):
    """Create 4 separate model performance visualization plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.style.use('default')
        sns.set_palette("husl")
    except ImportError:
        logging.error("Matplotlib/Seaborn not available. Skipping visualizations.")
        return
    
    results_dir = 'results/6_model_performance'
    
    # 1. Model Performance Metrics
    create_performance_metrics_plot(models_data, validation_results, results_dir)
    
    # 2. Prediction vs Actual Scatter Plots
    create_prediction_scatter_plots(validation_results, results_dir)
    
    # 3. Error Distribution Analysis
    create_error_distribution_plots(validation_results, results_dir)
    
    # 4. Training vs Validation Comparison
    create_training_validation_comparison(models_data, validation_results, results_dir)
    
    logging.info("All 4 visualization plots created successfully!")

def create_performance_metrics_plot(models_data, validation_results, results_dir):
    """Create Plot 1: Model Performance Metrics Comparison."""
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Model Performance Metrics', fontsize=16, fontweight='bold')
    
    # R² Scores
    targets = list(validation_results.keys())
    r2_scores = [validation_results[target]['r2'] for target in targets]
    
    bars1 = ax1.bar(targets, r2_scores, alpha=0.7, color=['skyblue', 'lightcoral'][:len(targets)])
    ax1.set_ylabel('R² Score')
    ax1.set_title('Model Accuracy (R² Scores)')
    ax1.set_ylim(0, 1)
    
    # Add value labels
    for bar, r2 in zip(bars1, r2_scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{r2:.4f}', ha='center', va='bottom', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Error Metrics (RMSE and MAE)
    rmse_values = [validation_results[target]['rmse'] for target in targets]
    mae_values = [validation_results[target]['mae'] for target in targets]
    
    x_pos = np.arange(len(targets))
    width = 0.35
    
    bars2 = ax2.bar(x_pos - width/2, rmse_values, width, label='RMSE', alpha=0.7, color='orange')
    bars3 = ax2.bar(x_pos + width/2, mae_values, width, label='MAE', alpha=0.7, color='green')
    
    ax2.set_ylabel('Error Value')
    ax2.set_title('Model Error Metrics')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(targets)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add value labels with smart formatting
    for bar, value in zip(bars2, rmse_values):
        if value >= 1000:
            label = f'{value:.1e}'
        else:
            label = f'{value:.3f}'
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(rmse_values) * 0.02,
                label, ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    for bar, value in zip(bars3, mae_values):
        if value >= 1000:
            label = f'{value:.1e}'
        else:
            label = f'{value:.3f}'
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mae_values) * 0.02,
                label, ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/1_performance_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Created performance metrics plot: {results_dir}/1_performance_metrics.png")

def create_prediction_scatter_plots(validation_results, results_dir):
    """Create Plot 2: Prediction vs Actual Scatter Plots."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    if not validation_results:
        logging.warning("No validation results for scatter plots")
        return
    
    n_targets = len(validation_results)
    fig, axes = plt.subplots(1, n_targets, figsize=(7*n_targets, 6))
    if n_targets == 1:
        axes = [axes]
    
    fig.suptitle('Prediction vs Actual Performance', fontsize=16, fontweight='bold')
    
    for i, (target, results) in enumerate(validation_results.items()):
        ax = axes[i]
        y_true = results['y_true']
        y_pred = results['y_pred']
        r2 = results['r2']
        
        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.6, s=20, color='blue')
        
        # Perfect prediction line
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2, label='Perfect Prediction')
        
        # Labels and formatting
        if 'MPP' in target:
            unit = 'W/cm²'
        else:
            unit = ''
        
        ax.set_xlabel(f'Actual {target} {unit}')
        ax.set_ylabel(f'Predicted {target} {unit}')
        ax.set_title(f'{target} Predictions\n(R² = {r2:.4f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/2_prediction_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Created prediction scatter plots: {results_dir}/2_prediction_scatter.png")

def create_error_distribution_plots(validation_results, results_dir):
    """Create Plot 3: Error Distribution Analysis."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    if not validation_results:
        logging.warning("No validation results for error distribution")
        return
    
    n_targets = len(validation_results)
    fig, axes = plt.subplots(2, n_targets, figsize=(7*n_targets, 10))
    if n_targets == 1:
        axes = axes.reshape(-1, 1)
    
    fig.suptitle('Error Distribution Analysis', fontsize=16, fontweight='bold')
    
    for i, (target, results) in enumerate(validation_results.items()):
        y_true = results['y_true']
        y_pred = results['y_pred']
        errors = y_pred - y_true
        
        # Error histogram (top row)
        ax_hist = axes[0, i]
        ax_hist.hist(errors, bins=30, alpha=0.7, edgecolor='black', color='lightblue')
        ax_hist.axvline(0, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Zero Error')
        ax_hist.set_xlabel('Prediction Error')
        ax_hist.set_ylabel('Frequency')
        ax_hist.set_title(f'{target} Error Distribution')
        ax_hist.legend()
        ax_hist.grid(True, alpha=0.3)
        
        # Residual plot (bottom row)
        ax_resid = axes[1, i]
        ax_resid.scatter(y_true, errors, alpha=0.6, s=20, color='orange')
        ax_resid.axhline(0, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Zero Error')
        ax_resid.set_xlabel(f'Actual {target}')
        ax_resid.set_ylabel('Residuals (Predicted - Actual)')
        ax_resid.set_title(f'{target} Residual Plot')
        ax_resid.legend()
        ax_resid.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/3_error_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Created error distribution plots: {results_dir}/3_error_distribution.png")

def create_training_validation_comparison(models_data, validation_results, results_dir):
    """Create Plot 4: Training vs Validation Performance Comparison."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Training vs Validation Performance', fontsize=16, fontweight='bold')
    
    # Extract training and validation metrics
    targets = []
    training_r2 = []
    validation_r2 = []
    
    if 'metadata' in models_data:
        metadata = models_data['metadata']
        
        # Get training metrics
        if 'efficiency_models' in metadata:
            for target, model_data in metadata['efficiency_models'].items():
                if target in validation_results:
                    targets.append(target)
                    best_model = model_data.get('best_model', 'Unknown')
                    if 'all_scores' in model_data and best_model in model_data['all_scores']:
                        train_r2 = model_data['all_scores'][best_model].get('cv_mean', 0)
                        training_r2.append(train_r2)
                    else:
                        training_r2.append(0)
                    
                    validation_r2.append(validation_results[target]['r2'])
        
        if 'recombination_models' in metadata:
            for target, model_data in metadata['recombination_models'].items():
                if target in validation_results:
                    targets.append(target)
                    best_model = model_data.get('best_model', 'Unknown')
                    if 'all_scores' in model_data and best_model in model_data['all_scores']:
                        train_r2 = model_data['all_scores'][best_model].get('cv_mean', 0)
                        training_r2.append(train_r2)
                    else:
                        training_r2.append(0)
                    
                    validation_r2.append(validation_results[target]['r2'])
    
    if targets:
        # R² Comparison
        x_pos = np.arange(len(targets))
        width = 0.35
        
        bars1 = ax1.bar(x_pos - width/2, training_r2, width, label='Training (CV)', alpha=0.7, color='lightgreen')
        bars2 = ax1.bar(x_pos + width/2, validation_r2, width, label='Validation', alpha=0.7, color='lightblue')
        
        ax1.set_ylabel('R² Score')
        ax1.set_title('Training vs Validation R² Scores')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(targets)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars1, training_r2):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        for bar, value in zip(bars2, validation_r2):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # Model Algorithm Comparison
        algorithms = []
        algorithm_scores = []
        
        if 'metadata' in models_data:
            for target in targets:
                if target in metadata.get('efficiency_models', {}):
                    model_data = metadata['efficiency_models'][target]
                    best_model = model_data.get('best_model', 'Unknown')
                    algorithms.append(f'{target}\n({best_model})')
                    if target in validation_results:
                        algorithm_scores.append(validation_results[target]['r2'])
                    else:
                        algorithm_scores.append(0)
                elif target in metadata.get('recombination_models', {}):
                    model_data = metadata['recombination_models'][target]
                    best_model = model_data.get('best_model', 'Unknown')
                    algorithms.append(f'{target}\n({best_model})')
                    if target in validation_results:
                        algorithm_scores.append(validation_results[target]['r2'])
                    else:
                        algorithm_scores.append(0)
        
        if algorithms:
            bars = ax2.bar(algorithms, algorithm_scores, alpha=0.7, color=['gold', 'lightcoral'][:len(algorithms)])
            ax2.set_ylabel('R² Score')
            ax2.set_title('Best Model Algorithms')
            ax2.set_ylim(0, 1)
            
            # Add value labels
            for bar, score in zip(bars, algorithm_scores):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.4f}', ha='center', va='bottom', fontweight='bold')
            ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/4_training_validation_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Created training vs validation comparison: {results_dir}/4_training_validation_comparison.png")


def save_performance_summary(models_data, validation_results):
    """Save performance summary to JSON."""
    results_dir = 'results/6_model_performance'
    
    summary = {
        "analysis_date": datetime.now().isoformat(),
        "validation_metrics": {},
        "training_summary": {},
        "model_info": {}
    }
    
    # Add validation metrics
    for target, results in validation_results.items():
        summary["validation_metrics"][target] = {
            "r2_score": results['r2'],
            "rmse": results['rmse'],
            "mae": results['mae'],
            "mape": results['mape']
        }
    
    # Add training info from metadata
    if 'metadata' in models_data:
        metadata = models_data['metadata']
        summary["training_summary"] = {
            "training_date": metadata.get('training_date', 'Unknown'),
            "efficiency_models": {},
            "recombination_models": {}
        }
        
        if 'efficiency_models' in metadata:
            for target, model_data in metadata['efficiency_models'].items():
                summary["training_summary"]["efficiency_models"][target] = {
                    "best_model": model_data.get('best_model', 'Unknown'),
                    "cv_mean": model_data.get('all_scores', {}).get(model_data.get('best_model', ''), {}).get('cv_mean', 0)
                }
        
        if 'recombination_models' in metadata:
            for target, model_data in metadata['recombination_models'].items():
                summary["training_summary"]["recombination_models"][target] = {
                    "best_model": model_data.get('best_model', 'Unknown'),
                    "cv_mean": model_data.get('all_scores', {}).get(model_data.get('best_model', ''), {}).get('cv_mean', 0)
                }
    
    # Save summary
    summary_path = f'{results_dir}/performance_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logging.info(f"Performance summary saved: {summary_path}")

def main():
    """Main model performance visualization workflow."""
    # Setup
    log_file = setup_logging()
    logging.info("=== ML MODEL PERFORMANCE VISUALIZATION ===")
    logging.info(f"Log file: {log_file}")
    
    try:
        # 1. Load trained models
        logging.info("\n=== Step 1: Loading Trained Models ===")
        models_data = load_trained_models()
        
        # 2. Load simulation data for validation
        logging.info("\n=== Step 2: Loading Simulation Data ===")
        validation_data = load_simulation_data()
        
        # 3. Validate model performance
        logging.info("\n=== Step 3: Validating Model Performance ===")
        validation_results = validate_model_performance(models_data, validation_data)
        
        # 4. Create separate visualizations
        logging.info("\n=== Step 4: Creating Visualizations ===")
        create_separate_visualizations(models_data, validation_results)
        
        # 5. Save performance summary
        logging.info("\n=== Step 5: Saving Performance Summary ===")
        save_performance_summary(models_data, validation_results)
        
        # Final summary
        logging.info("\n=== MODEL PERFORMANCE ANALYSIS COMPLETE ===")
        logging.info(f"Results saved to: results/6_model_performance/")
        logging.info(f"Plot 1: 1_performance_metrics.png")
        logging.info(f"Plot 2: 2_prediction_scatter.png")
        logging.info(f"Plot 3: 3_error_distribution.png")
        logging.info(f"Plot 4: 4_training_validation_comparison.png")
        logging.info(f"Summary: performance_summary.json")
        logging.info(f"Log: model_performance_log.txt")
        
        # Print key metrics
        if validation_results:
            logging.info("\n=== KEY PERFORMANCE METRICS ===")
            for target, results in validation_results.items():
                logging.info(f"{target}: R² = {results['r2']:.4f}, RMSE = {results['rmse']:.4f}")
        
    except Exception as e:
        logging.error(f"Model performance analysis failed: {e}")
        raise

if __name__ == "__main__":
    main()
