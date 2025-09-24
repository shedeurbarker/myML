"""
Script 7: Predict Experimental Device Performance

This script takes experimental device parameters from example_device_parameters.json
and generates comprehensive prediction results using trained ML models.

WORKFLOW:
1. Load experimental device parameters from example_device_parameters.json
2. Calculate all derived features to match training data
3. Use trained models to predict MPP, PCE, and recombination rates
4. Generate comprehensive prediction visualizations and analysis
5. Create detailed prediction reports with confidence intervals
6. Save results and recommendations

USAGE: python scripts/7_predict_experimental_data.py

REQUIREMENTS:
- Trained models from Script 5 (results/train_optimization_models/)
- Experimental device parameters (example_device_parameters.json)

OUTPUT:
- results/7_experimental_predictions/
  ├── prediction_log.txt                    # Detailed execution log
  ├── 1_performance_metrics_summary.png     # Overall performance predictions summary
  ├── 2_thickness_parameters.png            # Layer thickness analysis
  ├── 3_energy_parameters.png               # Energy level distribution
  ├── 4_doping_parameters.png               # Doping concentration analysis
  ├── 5_physics_validation.png              # Physics constraint validation
  ├── 6_efficiency_predictions.png            # Detailed MPP and PCE charts
  ├── 7_recombination_predictions.png         # Detailed recombination rate predictions
  ├── 8_parameter_analysis.png                # Combined parameter analysis
  └── prediction_report.json                # Detailed prediction results and recommendations
"""

import os
import sys
import json
import logging
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# Constants
RESULTS_DIR = 'results/7_experimental_predictions'

def setup_logging():
    """Set up logging configuration."""
    # Create results directory
    results_dir = RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)
    
    # Clear any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Set up logging
    log_filename = f'{results_dir}/prediction_log.txt'
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

def load_experimental_parameters():
    """Load experimental device parameters from JSON file."""
    param_file = 'example_device_parameters.json'
    
    if not os.path.exists(param_file):
        raise FileNotFoundError(f"Experimental parameters file not found: {param_file}")
    
    with open(param_file, 'r') as f:
        device_config = json.load(f)
    
    logging.info(f"Loaded experimental parameters from: {param_file}")
    logging.info(f"Device Type: {device_config.get('device_type', 'Unknown')}")
    
    # Log parameter summary
    params = device_config['parameters']
    logging.info("\nExperimental Device Parameters:")
    
    # Thickness parameters
    for layer in ['L1', 'L2', 'L3']:
        thickness = params[f'{layer}_L']
        logging.info(f"  {layer} thickness: {thickness*1e9:.1f} nm")
    
    # Energy parameters
    for layer in ['L1', 'L2', 'L3']:
        e_c = params[f'{layer}_E_c']
        e_v = params[f'{layer}_E_v']
        gap = abs(e_v - e_c)
        logging.info(f"  {layer} energy: E_c={e_c:.2f}eV, E_v={e_v:.2f}eV, Gap={gap:.2f}eV")
    
    # Doping parameters
    for layer in ['L1', 'L2', 'L3']:
        n_d = params[f'{layer}_N_D']
        n_a = params[f'{layer}_N_A']
        logging.info(f"  {layer} doping: N_D={n_d:.1e}, N_A={n_a:.1e} cm^-3")
    
    return params, device_config

def load_trained_models():
    """Load trained models and scalers."""
    models_dir = 'results/5_train_optimization_models'
    
    # Load models
    import joblib
    
    models_data = {}
    
    # Load MPP model
    mpp_model_path = f'{models_dir}/models/efficiency_MPP.joblib'
    mpp_scaler_path = f'{models_dir}/models/efficiency_MPP_scalers.joblib'
    
    if os.path.exists(mpp_model_path) and os.path.exists(mpp_scaler_path):
        models_data['mpp_model'] = joblib.load(mpp_model_path)
        models_data['mpp_scalers'] = joblib.load(mpp_scaler_path)
        logging.info(f"Loaded MPP model and scalers")
    
    # Load recombination model
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

def calculate_derived_features(df):
    """Calculate derived features to match training data."""
    logging.info("Calculating derived features...")
    
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
    
    # Band alignment features (EXACT names from training)
    if all(col in df.columns for col in ['L1_E_c', 'L2_E_c', 'L3_E_c', 'L1_E_v', 'L2_E_v', 'L3_E_v']):
        df['band_offset_L1_L2'] = df['L2_E_c'] - df['L1_E_c']
        df['band_offset_L2_L3'] = df['L3_E_c'] - df['L2_E_c']
        df['valence_band_offset'] = df['L3_E_v'] - df['L1_E_v']
    
    # Doping features
    if all(col in df.columns for col in ['L1_N_D', 'L1_N_A', 'L2_N_D', 'L2_N_A', 'L3_N_D', 'L3_N_A']):
        df['doping_ratio_L1'] = df['L1_N_D'] / (df['L1_N_A'] + 1e-30)
        df['doping_ratio_L2'] = df['L2_N_D'] / (df['L2_N_A'] + 1e-30)
        df['doping_ratio_L3'] = df['L3_N_D'] / (df['L3_N_A'] + 1e-30)
        df['total_donor_concentration'] = df['L1_N_D'] + df['L2_N_D'] + df['L3_N_D']
        df['total_acceptor_concentration'] = df['L1_N_A'] + df['L2_N_A'] + df['L3_N_A']
    
    # Enhanced physics-based features (default values for prediction mode)
    df['recombination_efficiency_ratio'] = 1e28  # Default typical value
    df['interface_quality_index'] = 1e-28  # Default typical value
    
    # Carrier transport efficiency features
    if all(col in df.columns for col in ['band_offset_L1_L2', 'band_offset_L2_L3']):
        df['conduction_band_alignment_quality'] = 1 / (1 + abs(df['band_offset_L1_L2']) + abs(df['band_offset_L2_L3']))
        
        # Valence band alignment quality
        if 'valence_band_offset' in df.columns:
            df['valence_band_alignment_quality'] = 1 / (1 + abs(df['valence_band_offset']))
    
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

def validate_shockley_queisser_limit(pce_pred, parameters):
    """Validate PCE prediction against Shockley-Queisser limit."""
    try:
        # Shockley-Queisser theoretical maximum for single-junction solar cells
        SQ_LIMIT = 33.7  # % (theoretical maximum efficiency)
        
        # Calculate active layer bandgap for reference
        active_bandgap = abs(parameters['L2_E_v'] - parameters['L2_E_c'])
        
        # Simple comparison against S-Q limit
        if pce_pred > SQ_LIMIT:
            logging.warning(f"PCE prediction ({pce_pred:.2f}%) exceeds Shockley-Queisser limit ({SQ_LIMIT}%)")
            logging.warning(f"Active layer bandgap: {active_bandgap:.2f} eV")
            logging.warning("This suggests unrealistic simulation data or model predictions")
            
            # Option 1: Cap at S-Q limit (conservative)
            # pce_pred = SQ_LIMIT
            
            # Option 2: Add warning but keep original prediction (current approach)
            logging.info(f"Keeping original prediction but flagging as potentially unrealistic")
        else:
            logging.info(f"PCE prediction ({pce_pred:.2f}%) is within S-Q limit ({SQ_LIMIT}%) [VALID]")
        
        return pce_pred
        
    except Exception as e:
        logging.warning(f"Error validating S-Q limit: {e}")
        return pce_pred

def predict_device_performance(parameters, models_data):
    """Predict comprehensive device performance."""
    # Create DataFrame with parameters
    df = pd.DataFrame([parameters])
    
    # Calculate derived features
    df = calculate_derived_features(df)
    
    # Get exact training features
    if 'metadata' in models_data and 'efficiency_models' in models_data['metadata']:
        training_features = models_data['metadata']['efficiency_models']['MPP']['feature_names']
        available_features = [f for f in training_features if f in df.columns]
        df_features = df[available_features]
        
        # Log any missing features
        missing_features = [f for f in training_features if f not in df.columns]
        if missing_features:
            logging.warning(f"Missing features: {missing_features}")
    else:
        df_features = df
        logging.warning("Could not find training feature names in metadata")
    
    predictions = {}
    
    # Predict MPP (efficiency)
    if 'mpp_model' in models_data:
        model = models_data['mpp_model']
        scalers = models_data['mpp_scalers']
        feature_scaler = scalers['feature_scaler']
        target_scaler = scalers['target_scaler']
        
        X_scaled = feature_scaler.transform(df_features)
        mpp_pred_scaled = model.predict(X_scaled)[0]
        
        # Inverse transform the scaled prediction back to original scale
        mpp_pred = target_scaler.inverse_transform([[mpp_pred_scaled]])[0][0]
        
        # Calculate PCE (Power Conversion Efficiency) from MPP using physics equation
        # PCE = (MPP / Incident_Power) × 100%
        # From simulation: MPP is in W/m² (V × J where J is in A/m²)
        # Standard solar conditions: 1000 W/m² incident power
        incident_power_W_per_m2 = 1000.0  # W/m² (AM1.5G standard)
        pce_pred = (mpp_pred / incident_power_W_per_m2) * 100  # Physics-based PCE calculation
        
        # Validate against Shockley-Queisser limit
        pce_pred = validate_shockley_queisser_limit(pce_pred, parameters)
        
        predictions['MPP'] = mpp_pred
        predictions['PCE'] = pce_pred
        
        logging.info(f"Efficiency Predictions:")
        logging.info(f"  MPP: {mpp_pred:.4f} W/m² (simulation units)")
        logging.info(f"  PCE: {pce_pred:.2f}%")
    
    # Predict recombination
    if 'recomb_model' in models_data:
        model = models_data['recomb_model']
        scalers = models_data['recomb_scalers']
        feature_scaler = scalers['feature_scaler']
        target_scaler = scalers['target_scaler']
        
        X_scaled = feature_scaler.transform(df_features)
        recomb_pred_scaled = model.predict(X_scaled)[0]
        
        # Inverse transform the scaled prediction back to original scale
        recomb_pred = target_scaler.inverse_transform([[recomb_pred_scaled]])[0][0]
        
        predictions['IntSRHn_mean'] = recomb_pred
        
        logging.info(f"Recombination Predictions:")
        logging.info(f"  IntSRHn_mean: {recomb_pred:.2e}")
    
    return predictions

def validate_experimental_parameters(parameters):
    """Validate experimental parameters against physics constraints."""
    logging.info("Validating experimental parameters...")
    
    # Extract parameters
    L1_E_c, L1_E_v = parameters['L1_E_c'], parameters['L1_E_v']
    L2_E_c, L2_E_v = parameters['L2_E_c'], parameters['L2_E_v'] 
    L3_E_c, L3_E_v = parameters['L3_E_c'], parameters['L3_E_v']
    
    validation_results = {
        'valid': True,
        'warnings': [],
        'constraints': {}
    }
    
    # Energy alignment constraints
    etl_active_alignment = L1_E_c >= L2_E_c
    active_htl_alignment = L2_E_v >= L3_E_v
    
    validation_results['constraints']['energy_alignment'] = {
        'ETL_Ec_ge_Active_Ec': etl_active_alignment,
        'Active_Ev_ge_HTL_Ev': active_htl_alignment
    }
    
    if not etl_active_alignment:
        validation_results['valid'] = False
        validation_results['warnings'].append(f"Energy alignment violation: ETL E_c ({L1_E_c:.2f}) < Active E_c ({L2_E_c:.2f})")
    
    if not active_htl_alignment:
        validation_results['valid'] = False
        validation_results['warnings'].append(f"Energy alignment violation: Active E_v ({L2_E_v:.2f}) < HTL E_v ({L3_E_v:.2f})")
    
    # Energy gap constraints
    gaps = {
        'L1': abs(L1_E_v - L1_E_c),
        'L2': abs(L2_E_v - L2_E_c),
        'L3': abs(L3_E_v - L3_E_c)
    }
    
    validation_results['constraints']['energy_gaps'] = gaps
    
    for layer, gap in gaps.items():
        if gap <= 0:
            validation_results['valid'] = False
            validation_results['warnings'].append(f"Invalid energy gap for {layer}: {gap:.3f} eV")
    
    # Electrode work function compatibility
    W_L, W_R = 4.05, 5.2  # From simulation setup
    electrode_L_compat = W_L >= L1_E_c
    electrode_R_compat = W_R <= L3_E_v
    
    validation_results['constraints']['electrode_compatibility'] = {
        'W_L_ge_ETL_Ec': electrode_L_compat,
        'W_R_le_HTL_Ev': electrode_R_compat
    }
    
    if not electrode_L_compat:
        validation_results['warnings'].append(f"Left electrode incompatible: W_L ({W_L}) < ETL E_c ({L1_E_c:.2f})")
    
    if not electrode_R_compat:
        validation_results['warnings'].append(f"Right electrode incompatible: W_R ({W_R}) > HTL E_v ({L3_E_v:.2f})")
    
    # Log validation results
    if validation_results['valid']:
        logging.info("Physics validation: PASSED")
    else:
        logging.warning("Physics validation: FAILED")
        for warning in validation_results['warnings']:
            logging.warning(f"  - {warning}")
    
    return validation_results

def create_prediction_visualizations(parameters, predictions, device_config, validation_results):
    """Create comprehensive prediction visualizations."""
    try:
        matplotlib.use('Agg')  # Use non-interactive backend
        import seaborn as sns
        plt.style.use('default')
        sns.set_palette("husl")
    except ImportError:
        logging.error("Matplotlib not available. Skipping visualizations.")
        return
    
    results_dir = RESULTS_DIR
    
    # 1. Individual Performance Summary Files
    create_performance_summary_charts(parameters, predictions, device_config, validation_results, results_dir)
    
    # 2. Efficiency Predictions Chart
    create_efficiency_chart(predictions, results_dir)
    
    # 3. Recombination Predictions Chart
    create_recombination_chart(predictions, results_dir)
    
    # 4. Parameter Analysis Chart
    create_parameter_analysis(parameters, results_dir)
    
    logging.info("All prediction visualizations created successfully!")

def create_performance_summary_charts(parameters, predictions, device_config, validation_results, results_dir):
    """Create separate performance summary charts."""
    import seaborn as sns
    
    # 1. Performance Metrics Summary
    create_performance_metrics_summary(predictions, validation_results, device_config, results_dir)
    
    # 2. Parameter Overview Charts
    create_individual_parameter_charts(parameters, results_dir)
    
    # 3. Physics Validation Chart
    create_physics_validation_summary(validation_results, results_dir)
    
    logging.info("Created separate performance summary charts")

def create_performance_metrics_summary(predictions, validation_results, device_config, results_dir):
    """Create performance metrics summary chart."""
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    fig.suptitle(f'Device Performance Predictions\n{device_config.get("device_type", "Unknown Device")}', 
                 fontsize=14, fontweight='bold')
    
    # Performance metrics
    metrics = []
    values = []
    colors = []
    
    if 'MPP' in predictions:
        metrics.append('MPP\n(W/cm²)')
        values.append(predictions['MPP'])
        colors.append('lightblue')
    
    if 'PCE' in predictions:
        metrics.append('PCE\n(%)')
        values.append(predictions['PCE'])
        colors.append('lightgreen')
    
    if 'IntSRHn_mean' in predictions:
        metrics.append('Recombination\n(log scale)')
        values.append(np.log10(predictions['IntSRHn_mean']) if predictions['IntSRHn_mean'] > 0 else 0)
        colors.append('lightcoral')
    
    if metrics:
        bars = ax.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Predicted Value')
        ax.set_title('Performance Predictions Summary')
        
        # Add value labels
        for bar, value, metric in zip(bars, values, metrics):
            if 'Recombination' in metric:
                label = f'1e{value:.1f}'
            elif 'PCE' in metric:
                label = f'{value:.2f}%'
            else:
                label = f'{value:.3f}'
            
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values) * 0.05,
                    label, ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        # Add extra space at the top for labels
        ax.set_ylim(0, max(values) * 1.2)
        
        # Add validation status
        status_color = 'green' if validation_results['valid'] else 'red'
        status_text = 'Physics: VALID' if validation_results['valid'] else 'Physics: INVALID'
        ax.text(0.02, 0.98, status_text, transform=ax.transAxes, fontsize=12, fontweight='bold',
                color=status_color, bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No predictions available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Device Performance Predictions')
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/1_performance_metrics_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Created performance metrics summary: {results_dir}/1_performance_metrics_summary.png")

def create_individual_parameter_charts(parameters, results_dir):
    """Create separate charts for each parameter type."""
    
    # Thickness Parameters Chart
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    thickness_params = ['L1_L', 'L2_L', 'L3_L']
    thickness_values = [parameters[p] * 1e9 for p in thickness_params]
    thickness_labels = ['ETL', 'Active', 'HTL']
    
    bars = ax.bar(thickness_labels, thickness_values, color=['skyblue', 'lightgreen', 'lightsalmon'], 
                  alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Thickness (nm)')
    ax.set_title('Layer Thickness Distribution')
    
    for bar, value in zip(bars, thickness_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(thickness_values) * 0.05,
                f'{value:.1f}nm', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    # Add extra space at the top for labels
    ax.set_ylim(0, max(thickness_values) * 1.2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/2_thickness_parameters.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Energy Parameters Chart
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    energy_params = ['L1_E_c', 'L1_E_v', 'L2_E_c', 'L2_E_v', 'L3_E_c', 'L3_E_v']
    energy_values = [parameters[p] for p in energy_params]
    energy_labels = ['ETL E_c', 'ETL E_v', 'Act E_c', 'Act E_v', 'HTL E_c', 'HTL E_v']
    
    bars = ax.bar(energy_labels, energy_values, color='gold', alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Energy (eV)')
    ax.set_title('Energy Level Distribution')
    
    for bar, value in zip(bars, energy_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(energy_values) * 0.02,
                f'{value:.2f}eV', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    # Add extra space at the top for labels
    ax.set_ylim(0, max(energy_values) * 1.15)
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/3_energy_parameters.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Doping Parameters Chart
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    doping_params = ['L1_N_D', 'L1_N_A', 'L2_N_D', 'L2_N_A', 'L3_N_D', 'L3_N_A']
    doping_values = [np.log10(parameters[p]) if parameters[p] > 0 else 0 for p in doping_params]
    doping_labels = ['ETL N_D', 'ETL N_A', 'Act N_D', 'Act N_A', 'HTL N_D', 'HTL N_A']
    
    bars = ax.bar(doping_labels, doping_values, color='purple', alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('log₁₀(Concentration) [cm⁻³]')
    ax.set_title('Doping Concentration Distribution')
    
    for bar, value in zip(bars, doping_values):
        if value > 0:
            label = f'1e{value:.1f}'
        else:
            label = '0'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(doping_values) * 0.02,
                label, ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)
    
    # Add extra space at the top for labels
    ax.set_ylim(0, max(doping_values) * 1.15)
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/4_doping_parameters.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Created parameter charts: thickness, energy, and doping")

def create_physics_validation_summary(validation_results, results_dir):
    """Create physics validation summary chart."""
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    if not validation_results['constraints']:
        ax.text(0.5, 0.5, 'No validation data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Physics Validation Summary')
        plt.savefig(f'{results_dir}/5_physics_validation.png', dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    # Collect all constraints
    all_constraints = []
    all_status = []
    
    for category, constraints in validation_results['constraints'].items():
        if isinstance(constraints, dict):
            for name, status in constraints.items():
                all_constraints.append(name.replace('_', ' '))
                all_status.append(status)
    
    if all_constraints:
        colors = ['green' if status else 'red' for status in all_status]
        y_pos = range(len(all_constraints))
        
        bars = ax.barh(y_pos, [1] * len(all_constraints), color=colors, alpha=0.7, edgecolor='black')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(all_constraints)
        ax.set_xlabel('Constraint Status')
        ax.set_title('Physics Constraint Validation Summary')
        ax.set_xlim(0, 1.2)
        
        # Add status labels
        for bar, status in zip(bars, all_status):
            label = 'PASS' if status else 'FAIL'
            ax.text(0.5, bar.get_y() + bar.get_height()/2, label, 
                    ha='center', va='center', fontweight='bold', color='white', fontsize=10)
        
        # Add overall status
        overall_status = 'VALID' if validation_results['valid'] else 'INVALID'
        status_color = 'green' if validation_results['valid'] else 'red'
        ax.text(0.02, 0.98, f'Overall: {overall_status}', transform=ax.transAxes, 
                fontsize=12, fontweight='bold', color=status_color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/5_physics_validation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Created physics validation summary: {results_dir}/5_physics_validation.png")

def create_performance_summary(predictions, validation_results, ax):
    """Create performance summary chart."""
    # Performance metrics
    metrics = []
    values = []
    colors = []
    
    if 'MPP' in predictions:
        metrics.append('MPP\n(W/cm²)')
        values.append(predictions['MPP'])
        colors.append('lightblue')
    
    if 'PCE' in predictions:
        metrics.append('PCE\n(%)')
        values.append(predictions['PCE'])
        colors.append('lightgreen')
    
    if 'IntSRHn_mean' in predictions:
        metrics.append('Recombination\n(log scale)')
        values.append(np.log10(predictions['IntSRHn_mean']) if predictions['IntSRHn_mean'] > 0 else 0)
        colors.append('lightcoral')
    
    if metrics:
        bars = ax.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Predicted Value')
        ax.set_title('Device Performance Predictions')
        
        # Add value labels
        for bar, value, metric in zip(bars, values, metrics):
            if 'Recombination' in metric:
                label = f'1e{value:.1f}'
            elif 'PCE' in metric:
                label = f'{value:.2f}%'
            else:
                label = f'{value:.2f}'
            
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values) * 0.02,
                    label, ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        # Add validation status
        status_color = 'green' if validation_results['valid'] else 'red'
        status_text = 'Physics: VALID' if validation_results['valid'] else 'Physics: INVALID'
        ax.text(0.02, 0.98, status_text, transform=ax.transAxes, fontsize=12, fontweight='bold',
                color=status_color, bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No predictions available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Device Performance Predictions')

def create_parameter_overview(parameters, ax, param_type):
    """Create parameter overview chart."""
    if param_type == 'Thickness':
        params = ['L1_L', 'L2_L', 'L3_L']
        values = [parameters[p] * 1e9 for p in params]  # Convert to nm
        labels = ['ETL', 'Active', 'HTL']
        ylabel = 'Thickness (nm)'
        colors = ['skyblue', 'lightgreen', 'lightsalmon']
    elif param_type == 'Energy':
        params = ['L1_E_c', 'L1_E_v', 'L2_E_c', 'L2_E_v', 'L3_E_c', 'L3_E_v']
        values = [parameters[p] for p in params]
        labels = ['ETL E_c', 'ETL E_v', 'Act E_c', 'Act E_v', 'HTL E_c', 'HTL E_v']
        ylabel = 'Energy (eV)'
        colors = ['lightblue'] * 6
    elif param_type == 'Doping':
        params = ['L1_N_D', 'L1_N_A', 'L2_N_D', 'L2_N_A', 'L3_N_D', 'L3_N_A']
        values = [np.log10(parameters[p]) if parameters[p] > 0 else 0 for p in params]
        labels = ['ETL N_D', 'ETL N_A', 'Act N_D', 'Act N_A', 'HTL N_D', 'HTL N_A']
        ylabel = 'log₁₀(Concentration)'
        colors = ['purple'] * 6
    
    bars = ax.bar(labels, values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel(ylabel)
    ax.set_title(f'{param_type} Parameters')
    
    # Add value labels
    for bar, value in zip(bars, values):
        if param_type == 'Thickness':
            label = f'{value:.1f}nm'
        elif param_type == 'Energy':
            label = f'{value:.2f}eV'
        else:  # Doping
            label = f'1e{value:.1f}'
        
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values) * 0.02,
                label, ha='center', va='bottom', fontweight='bold', fontsize=8, rotation=45)
    
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)


def create_efficiency_chart(predictions, results_dir):
    """Create detailed efficiency predictions chart."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Efficiency Predictions', fontsize=14, fontweight='bold')
    
    # MPP Chart
    if 'MPP' in predictions:
        mpp = predictions['MPP']
        ax1.bar(['MPP'], [mpp], color='lightblue', alpha=0.7, edgecolor='black')
        ax1.set_ylabel('Power Density (W/cm²)')
        ax1.set_title('Maximum Power Point (MPP)')
        ax1.text(0, mpp + abs(mpp) * 0.02, f'{mpp:.3f} W/cm²', 
                ha='center', va='bottom', fontweight='bold', fontsize=10)
        ax1.grid(True, alpha=0.3)
        # Add extra space at the top for labels
        ax1.set_ylim(0, mpp * 1.15)
    
    # PCE Chart
    if 'PCE' in predictions:
        pce = predictions['PCE']
        ax2.bar(['PCE'], [pce], color='lightgreen', alpha=0.7, edgecolor='black')
        ax2.set_ylabel('Efficiency (%)')
        ax2.set_title('Power Conversion Efficiency (PCE)')
        ax2.text(0, pce + abs(pce) * 0.02, f'{pce:.2f}%', 
                ha='center', va='bottom', fontweight='bold', fontsize=10)
        ax2.grid(True, alpha=0.3)
        # Add extra space at the top for labels
        ax2.set_ylim(0, pce * 1.15)
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/6_efficiency_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Created efficiency chart: {results_dir}/6_efficiency_predictions.png")

def create_recombination_chart(predictions, results_dir):
    """Create recombination predictions chart."""
    import numpy as np
    
    if 'IntSRHn_mean' not in predictions:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Recombination Rate Predictions', fontsize=14, fontweight='bold')
    
    recomb = predictions['IntSRHn_mean']
    
    # Linear scale
    ax1.bar(['IntSRHn_mean'], [recomb], color='lightcoral', alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Recombination Rate')
    ax1.set_title('Linear Scale')
    ax1.text(0, recomb + abs(recomb) * 0.02, f'{recomb:.2e}', 
            ha='center', va='bottom', fontweight='bold', fontsize=10)
    ax1.grid(True, alpha=0.3)
    # Add extra space at the top for labels
    ax1.set_ylim(0, recomb * 1.15)
    
    # Log scale
    log_recomb = np.log10(recomb) if recomb > 0 else 0
    ax2.bar(['IntSRHn_mean'], [log_recomb], color='orange', alpha=0.7, edgecolor='black')
    ax2.set_ylabel('log₁₀(Recombination Rate)')
    ax2.set_title('Logarithmic Scale')
    ax2.text(0, log_recomb + abs(log_recomb) * 0.02, f'1e{log_recomb:.1f}', 
            ha='center', va='bottom', fontweight='bold', fontsize=10)
    ax2.grid(True, alpha=0.3)
    # Add extra space at the top for labels
    ax2.set_ylim(log_recomb - abs(log_recomb) * 0.1, log_recomb * 1.15)
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/7_recombination_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Created recombination chart: {results_dir}/7_recombination_predictions.png")

def create_parameter_analysis(parameters, results_dir):
    """Create detailed parameter analysis chart."""
    import numpy as np
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Device Parameter Analysis', fontsize=16, fontweight='bold')
    
    # Thickness analysis
    ax = axes[0, 0]
    thickness_params = ['L1_L', 'L2_L', 'L3_L']
    thickness_values = [parameters[p] * 1e9 for p in thickness_params]
    thickness_labels = ['ETL', 'Active', 'HTL']
    
    bars = ax.bar(thickness_labels, thickness_values, color=['skyblue', 'lightgreen', 'lightsalmon'], alpha=0.7)
    ax.set_ylabel('Thickness (nm)')
    ax.set_title('Layer Thickness Distribution')
    
    for bar, value in zip(bars, thickness_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(thickness_values) * 0.02,
                f'{value:.1f}nm', ha='center', va='bottom', fontweight='bold', fontsize=10)
    ax.grid(True, alpha=0.3)
    # Add extra space at the top for labels
    ax.set_ylim(0, max(thickness_values) * 1.15)
    
    # Energy gap analysis
    ax = axes[0, 1]
    energy_gaps = []
    gap_labels = []
    for layer in ['L1', 'L2', 'L3']:
        gap = abs(parameters[f'{layer}_E_v'] - parameters[f'{layer}_E_c'])
        energy_gaps.append(gap)
        gap_labels.append(f'{layer}\nGap')
    
    bars = ax.bar(gap_labels, energy_gaps, color='gold', alpha=0.7)
    ax.set_ylabel('Energy Gap (eV)')
    ax.set_title('Energy Gap Analysis')
    
    for bar, gap in zip(bars, energy_gaps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(energy_gaps) * 0.02,
                f'{gap:.2f}eV', ha='center', va='bottom', fontweight='bold', fontsize=10)
    ax.grid(True, alpha=0.3)
    # Add extra space at the top for labels
    ax.set_ylim(0, max(energy_gaps) * 1.15)
    
    # Doping ratio analysis
    ax = axes[1, 0]
    doping_ratios = []
    ratio_labels = []
    for layer in ['L1', 'L2', 'L3']:
        ratio = parameters[f'{layer}_N_D'] / (parameters[f'{layer}_N_A'] + 1e-30)
        doping_ratios.append(np.log10(ratio) if ratio > 0 else 0)
        ratio_labels.append(f'{layer}\nN_D/N_A')
    
    bars = ax.bar(ratio_labels, doping_ratios, color='purple', alpha=0.7)
    ax.set_ylabel('log₁₀(N_D/N_A Ratio)')
    ax.set_title('Doping Ratio Analysis')
    
    for bar, ratio in zip(bars, doping_ratios):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(doping_ratios) * 0.02,
                f'1e{ratio:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    ax.grid(True, alpha=0.3)
    # Add extra space at the top for labels
    ax.set_ylim(min(doping_ratios) - abs(min(doping_ratios)) * 0.1, max(doping_ratios) * 1.15)
    
    # Band alignment analysis
    ax = axes[1, 1]
    band_offsets = [
        parameters['L2_E_c'] - parameters['L1_E_c'],  # ETL-Active
        parameters['L3_E_c'] - parameters['L2_E_c'],  # Active-HTL
        parameters['L3_E_v'] - parameters['L1_E_v']   # Overall valence offset
    ]
    offset_labels = ['ETL-Act\nE_c', 'Act-HTL\nE_c', 'Overall\nE_v']
    
    colors = ['green' if offset >= 0 else 'red' for offset in band_offsets]
    bars = ax.bar(offset_labels, band_offsets, color=colors, alpha=0.7)
    ax.set_ylabel('Band Offset (eV)')
    ax.set_title('Band Alignment Analysis')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    for bar, offset in zip(bars, band_offsets):
        ax.text(bar.get_x() + bar.get_width()/2, 
                bar.get_height() + (abs(offset) * 0.05 if offset > 0 else -abs(offset) * 0.05),
                f'{offset:+.3f}eV', ha='center', 
                va='bottom' if offset > 0 else 'top', fontweight='bold', fontsize=10)
    ax.grid(True, alpha=0.3)
    # Add extra space at the top and bottom for labels
    max_abs = max(abs(min(band_offsets)), abs(max(band_offsets)))
    ax.set_ylim(min(band_offsets) - max_abs * 0.15, max(band_offsets) + max_abs * 0.15)
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/8_parameter_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Created parameter analysis: {results_dir}/8_parameter_analysis.png")

def create_physics_validation_chart(validation_results, ax):
    """Create physics validation status chart."""
    if not validation_results['constraints']:
        ax.text(0.5, 0.5, 'No validation data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Physics Validation')
        return
    
    # Collect all constraints
    all_constraints = []
    all_status = []
    
    for category, constraints in validation_results['constraints'].items():
        if isinstance(constraints, dict):
            for name, status in constraints.items():
                all_constraints.append(name.replace('_', ' '))
                all_status.append(status)
    
    if all_constraints:
        colors = ['green' if status else 'red' for status in all_status]
        y_pos = range(len(all_constraints))
        
        bars = ax.barh(y_pos, [1] * len(all_constraints), color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(all_constraints)
        ax.set_xlabel('Constraint Status')
        ax.set_title('Physics Constraint Validation')
        ax.set_xlim(0, 1.2)
        
        # Add status labels
        for bar, status in zip(bars, all_status):
            label = 'PASS' if status else 'FAIL'
            ax.text(0.5, bar.get_y() + bar.get_height()/2, label, 
                    ha='center', va='center', fontweight='bold', color='white', fontsize=10)

def save_prediction_report(parameters, predictions, device_config, validation_results):
    """Save detailed prediction report."""
    results_dir = RESULTS_DIR
    
    report = {
        "prediction_date": datetime.now().isoformat(),
        "device_type": device_config.get('device_type', 'Unknown'),
        "experimental_parameters": parameters,
        "predictions": predictions,
        "physics_validation": validation_results,
        "performance_summary": {},
        "recommendations": []
    }
    
    # Add performance summary
    if 'MPP' in predictions:
        report["performance_summary"]["efficiency"] = {
            "MPP_W_per_cm2": predictions['MPP'],
            "PCE_percent": predictions.get('PCE', 0),
            "efficiency_category": "High" if predictions['MPP'] > 20 else "Moderate" if predictions['MPP'] > 10 else "Low"
        }
    
    if 'IntSRHn_mean' in predictions:
        report["performance_summary"]["recombination"] = {
            "IntSRHn_mean": predictions['IntSRHn_mean'],
            "recombination_category": "Low" if predictions['IntSRHn_mean'] < 1e29 else "Moderate" if predictions['IntSRHn_mean'] < 1e30 else "High"
        }
    
    # Add recommendations
    if validation_results['valid']:
        report["recommendations"].append("Physics constraints satisfied - device should be manufacturable")
    else:
        report["recommendations"].append("Physics constraints violated - device parameters need adjustment")
        report["recommendations"].extend(validation_results['warnings'])
    
    if 'MPP' in predictions:
        if predictions['MPP'] > 20:
            report["recommendations"].append("Excellent efficiency predicted - suitable for high-performance applications")
        elif predictions['MPP'] > 10:
            report["recommendations"].append("Good efficiency predicted - suitable for standard applications")
        else:
            report["recommendations"].append("Low efficiency predicted - consider parameter optimization")
    
    # Save report
    report_path = f'{results_dir}/prediction_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logging.info(f"Prediction report saved: {report_path}")
    return report

def main():
    """Main prediction workflow."""
    # Setup
    log_file = setup_logging()
    logging.info("=== EXPERIMENTAL DEVICE PERFORMANCE PREDICTION ===")
    logging.info(f"Log file: {log_file}")
    
    try:
        # 1. Load experimental parameters
        logging.info("\n=== Step 1: Loading Experimental Parameters ===")
        parameters, device_config = load_experimental_parameters()
        
        # 2. Load trained models
        logging.info("\n=== Step 2: Loading Trained Models ===")
        models_data = load_trained_models()
        
        # 3. Validate experimental parameters
        logging.info("\n=== Step 3: Validating Parameters ===")
        validation_results = validate_experimental_parameters(parameters)
        
        # 4. Predict device performance
        logging.info("\n=== Step 4: Predicting Device Performance ===")
        predictions = predict_device_performance(parameters, models_data)
        
        # 5. Create visualizations
        logging.info("\n=== Step 5: Creating Visualizations ===")
        create_prediction_visualizations(parameters, predictions, device_config, validation_results)
        
        # 6. Save prediction report
        logging.info("\n=== Step 6: Saving Prediction Report ===")
        report = save_prediction_report(parameters, predictions, device_config, validation_results)
        
        # Final summary
        logging.info("\n=== PREDICTION ANALYSIS COMPLETE ===")
        logging.info(f"Results saved to: {RESULTS_DIR}/")
        logging.info(f"Performance Metrics: 1_performance_metrics_summary.png")
        logging.info(f"Thickness: 2_thickness_parameters.png")
        logging.info(f"Energy: 3_energy_parameters.png")
        logging.info(f"Doping: 4_doping_parameters.png")
        logging.info(f"Physics: 5_physics_validation.png")
        logging.info(f"Efficiency: 6_efficiency_predictions.png")
        logging.info(f"Recombination: 7_recombination_predictions.png")
        logging.info(f"Parameters: 8_parameter_analysis.png")
        logging.info(f"Report: prediction_report.json")
        
        # Print key predictions
        if predictions:
            logging.info("\n=== PREDICTED PERFORMANCE ===")
            if 'MPP' in predictions:
                logging.info(f"MPP: {predictions['MPP']:.4f} W/m² (simulation units)")
            if 'PCE' in predictions:
                logging.info(f"PCE: {predictions['PCE']:.2f}%")
            if 'IntSRHn_mean' in predictions:
                logging.info(f"Recombination: {predictions['IntSRHn_mean']:.2e}")
        
        # Print validation status
        if validation_results['valid']:
            logging.info("Physics Status: VALID - Device is manufacturable")
        else:
            logging.info("Physics Status: INVALID - Parameter adjustment needed")
            
    except Exception as e:
        logging.error(f"Prediction analysis failed: {e}")
        raise

if __name__ == "__main__":
    main()
