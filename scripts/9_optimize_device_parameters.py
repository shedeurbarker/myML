#!/usr/bin/env python3
"""
Script 9: Optimize Device Parameters for Maximum Efficiency

This script takes device parameters from example_device_parameters.json,
predicts their current performance, optimizes them for maximum efficiency,
and provides a detailed comparison with visualizations.

WORKFLOW:
1. Load device parameters from example_device_parameters.json
2. Predict current efficiency and recombination rates
3. Optimize parameters for maximum efficiency while controlling recombination
4. Predict optimized performance
5. Create comparison visualizations and tables
6. Save results and recommendations

USAGE: python scripts/9_optimize_device_parameters.py [--fast] [--maxiter N] [--popsize N]

REQUIREMENTS:
- Trained models from Script 5 (results/train_optimization_models/)
- Example device parameters (example_device_parameters.json)

OUTPUT:
- results/optimize_device/
  ├── optimization_comparison.png      # Side-by-side parameter comparison
  ├── performance_comparison.png       # Efficiency and recombination comparison
  ├── parameter_changes_table.png      # Detailed changes table
  ├── optimization_report.json         # Detailed optimization results
  └── optimization_log.txt            # Process log
"""

import os
import sys
import json
import logging
import argparse
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution

def setup_logging():
    """Set up logging configuration."""
    # Create results directory
    results_dir = 'results/optimize_device'
    os.makedirs(results_dir, exist_ok=True)
    
    # Clear any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Set up logging
    log_filename = f'{results_dir}/optimization_log.txt'
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

def load_device_parameters():
    """Load device parameters from example_device_parameters.json."""
    param_file = 'example_device_parameters.json'
    
    if not os.path.exists(param_file):
        raise FileNotFoundError(f"Device parameters file not found: {param_file}")
    
    with open(param_file, 'r') as f:
        device_config = json.load(f)
    
    logging.info(f"Loaded device parameters from: {param_file}")
    logging.info(f"Device Type: {device_config.get('device_type', 'Unknown')}")
    
    return device_config['parameters'], device_config

def load_optimization_models():
    """Load trained models and scalers for optimization."""
    models_dir = 'results/train_optimization_models'
    
    # Load models
    import joblib
    
    # Load efficiency models (individual files)
    efficiency_models = {}
    efficiency_scalers = {}
    
    # Load MPP model
    mpp_model_path = f'{models_dir}/models/efficiency_MPP.joblib'
    mpp_scaler_path = f'{models_dir}/models/efficiency_MPP_scalers.joblib'
    
    if os.path.exists(mpp_model_path):
        efficiency_models['MPP'] = joblib.load(mpp_model_path)
        logging.info(f"Loaded MPP model from {mpp_model_path}")
    
    if os.path.exists(mpp_scaler_path):
        efficiency_scalers['MPP'] = joblib.load(mpp_scaler_path)
        logging.info(f"Loaded MPP scalers from {mpp_scaler_path}")
    
    # Load recombination models (individual files)
    recombination_models = {}
    recombination_scalers = {}
    
    # Load IntSRHn_mean model
    recomb_model_path = f'{models_dir}/models/recombination_IntSRHn_mean.joblib'
    recomb_scaler_path = f'{models_dir}/models/recombination_IntSRHn_mean_scalers.joblib'
    
    if os.path.exists(recomb_model_path):
        recombination_models['IntSRHn_mean'] = joblib.load(recomb_model_path)
        logging.info(f"Loaded IntSRHn_mean model from {recomb_model_path}")
    
    if os.path.exists(recomb_scaler_path):
        recombination_scalers['IntSRHn_mean'] = joblib.load(recomb_scaler_path)
        logging.info(f"Loaded IntSRHn_mean scalers from {recomb_scaler_path}")
    
    # Load metadata for feature information
    metadata_path = f'{models_dir}/training_metadata.json'
    metadata = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        logging.info("Loaded training metadata")
    
    return {
        'efficiency_models': efficiency_models,
        'efficiency_scalers': efficiency_scalers,
        'recombination_models': recombination_models,
        'recombination_scalers': recombination_scalers,
        'metadata': metadata
    }

def calculate_derived_features(df):
    """Calculate derived features from primary parameters.
    
    This function replicates the EXACT feature engineering from Script 4
    to ensure compatibility with trained models.
    """
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
    
    # Band alignment features (EXACT names from Script 4)
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
    
    # Enhanced physics-based features (for prediction mode, use default values)
    df['recombination_efficiency_ratio'] = 1e28  # Default typical value
    df['interface_quality_index'] = 1e-28  # Default typical value
    
    # Carrier transport efficiency features (EXACT from Script 4)
    if all(col in df.columns for col in ['band_offset_L1_L2', 'band_offset_L2_L3']):
        # Conduction band alignment quality (smooth transitions = better transport)
        df['conduction_band_alignment_quality'] = 1 / (1 + abs(df['band_offset_L1_L2']) + abs(df['band_offset_L2_L3']))
        
        # Valence band alignment quality
        if 'valence_band_offset' in df.columns:
            df['valence_band_alignment_quality'] = 1 / (1 + abs(df['valence_band_offset']))
    
    # Ensure valence_band_alignment_quality exists (default value if not calculated above)
    if 'valence_band_alignment_quality' not in df.columns:
        df['valence_band_alignment_quality'] = 0.5  # Default moderate alignment quality
    
    # Thickness optimization features (EXACT from Script 4)
    if all(col in df.columns for col in ['thickness_ratio_L2', 'thickness_ratio_ETL', 'thickness_ratio_HTL']):
        # Optimal thickness balance (active layer should be dominant)
        df['thickness_balance_quality'] = df['thickness_ratio_L2'] / (df['thickness_ratio_ETL'] + df['thickness_ratio_HTL'] + 1e-30)
        
        # Transport layer thickness ratio (should be balanced)
        df['transport_layer_balance'] = 1 / (1 + abs(df['thickness_ratio_ETL'] - df['thickness_ratio_HTL']))
    
    # Doping optimization features (EXACT from Script 4)
    if all(col in df.columns for col in ['doping_ratio_L1', 'doping_ratio_L2', 'doping_ratio_L3']):
        # Average doping ratio across layers
        df['average_doping_ratio'] = df[['doping_ratio_L1', 'doping_ratio_L2', 'doping_ratio_L3']].mean(axis=1)
        
        # Doping consistency across layers
        df['doping_consistency'] = 1 / (1 + df[['doping_ratio_L1', 'doping_ratio_L2', 'doping_ratio_L3']].var(axis=1))
    
    # Energy level optimization features (EXACT from Script 4)
    if all(col in df.columns for col in ['energy_gap_L1', 'energy_gap_L2', 'energy_gap_L3']):
        # Energy gap progression (absolute value to ensure positive)
        df['energy_gap_progression'] = abs((df['energy_gap_L2'] - df['energy_gap_L1']) * (df['energy_gap_L3'] - df['energy_gap_L2']))
        
        # Energy gap uniformity (for specific device types)
        df['energy_gap_uniformity'] = 1 / (1 + df[['energy_gap_L1', 'energy_gap_L2', 'energy_gap_L3']].var(axis=1))
    
    logging.info(f"Calculated all derived features. Total features: {len(df.columns)}")
    return df

def predict_performance(parameters, models_data):
    """Predict efficiency and recombination for given parameters."""
    # Create DataFrame with parameters
    df = pd.DataFrame([parameters])
    
    # Calculate derived features
    df = calculate_derived_features(df)
    
    # Get feature list from metadata (exact 38 features used during training)
    if 'efficiency_models' in models_data['metadata'] and 'MPP' in models_data['metadata']['efficiency_models']:
        training_features = models_data['metadata']['efficiency_models']['MPP']['feature_names']
        # Only use the exact features that were used during training
        available_features = [f for f in training_features if f in df.columns]
        df_features = df[available_features]
        
        # Log any missing features
        missing_features = [f for f in training_features if f not in df.columns]
        if missing_features:
            logging.warning(f"Missing features: {missing_features}")
    else:
        # Fallback: use all columns except targets
        df_features = df
        logging.warning("Could not find training feature names in metadata")
    
    predictions = {}
    
    # Predict efficiency
    for target, model in models_data['efficiency_models'].items():
        if target in models_data['efficiency_scalers']:
            scalers = models_data['efficiency_scalers'][target]
            feature_scaler = scalers['feature_scaler']
            X_scaled = feature_scaler.transform(df_features)
            pred = model.predict(X_scaled)[0]
            predictions[f'predicted_{target}'] = pred
            logging.debug(f"Predicted {target}: {pred:.4f}")
    
    # Predict recombination
    for target, model in models_data['recombination_models'].items():
        if target in models_data['recombination_scalers']:
            scalers = models_data['recombination_scalers'][target]
            feature_scaler = scalers['feature_scaler']
            X_scaled = feature_scaler.transform(df_features)
            pred = model.predict(X_scaled)[0]
            predictions[f'predicted_{target}'] = pred
            logging.debug(f"Predicted {target}: {pred:.2e}")
    
    return predictions

def validate_physics_constraints(params):
    """Validate physics constraints for device parameters."""
    # Extract parameters
    L1_E_c, L1_E_v = params[1], params[2]  # ETL
    L2_E_c, L2_E_v = params[6], params[7]  # Active
    L3_E_c, L3_E_v = params[11], params[12]  # HTL
    
    # Energy alignment constraints
    if L1_E_c < L2_E_c:  # ETL conduction band should be >= Active
        return False
    if L2_E_v < L3_E_v:  # Active valence band should be >= HTL
        return False
    
    # Energy gap constraints (must be positive)
    if L1_E_v <= L1_E_c or L2_E_v <= L2_E_c or L3_E_v <= L3_E_c:
        return False
    
    # Electrode work function compatibility (with small margin for numerical stability)
    W_L, W_R = 4.05, 5.2  # From simulation_setup.txt
    margin = 0.01  # Small margin to avoid boundary issues
    if W_L < (L1_E_c - margin) or W_R > (L3_E_v + margin):
        return False
    
    return True

def objective_function(params, models_data, parameter_names):
    """Objective function for optimization: maximize MPP while controlling recombination."""
    # Convert array to parameter dictionary
    param_dict = dict(zip(parameter_names, params))
    
    # Validate physics constraints
    if not validate_physics_constraints(params):
        return 1e6  # Large penalty for invalid physics
    
    try:
        # Predict performance
        predictions = predict_performance(param_dict, models_data)
        
        # Get efficiency (MPP) - we want to maximize this
        efficiency = predictions.get('predicted_MPP', 0)
        
        # Get recombination - we want to control this (not too high)
        recombination = predictions.get('predicted_IntSRHn_mean', 1e30)
        
        # Objective: minimize negative efficiency with recombination penalty
        # Penalize if recombination is too high (> 1e32)
        recomb_penalty = 0
        if recombination > 1e32:
            recomb_penalty = (recombination - 1e32) / 1e30
        
        objective = -efficiency + recomb_penalty
        
        return objective
        
    except Exception as e:
        logging.warning(f"Error in objective function: {e}")
        return 1e6  # Large penalty for errors

def optimize_parameters(original_params, models_data, maxiter=1000, popsize=15):
    """Optimize device parameters for maximum efficiency."""
    logging.info("Starting parameter optimization...")
    
    # Load parameter bounds from feature definitions
    bounds_file = 'results/features/parameter_bounds.json'
    if not os.path.exists(bounds_file):
        raise FileNotFoundError(f"Parameter bounds file not found: {bounds_file}")
    
    with open(bounds_file, 'r') as f:
        parameter_bounds = json.load(f)
    
    # Parameter names and bounds
    parameter_names = list(original_params.keys())
    bounds = []
    
    for param in parameter_names:
        if param in parameter_bounds:
            param_bounds = parameter_bounds[param]
            # Convert thickness bounds from nm to meters to match parameter units
            if 'L' in param and param.endswith('_L'):
                # Convert from nm to meters (bounds are in nm, parameters are in meters)
                bounds.append([param_bounds[0] * 1e-9, param_bounds[1] * 1e-9])
            else:
                bounds.append(param_bounds)
        else:
            # Fallback bounds
            current_value = original_params[param]
            bounds.append([current_value * 0.5, current_value * 2.0])
    
    logging.info(f"Optimizing {len(parameter_names)} parameters")
    logging.info(f"Using bounds from: {bounds_file}")
    
    # Initial guess (current parameters, but ensure physics compliance)
    x0 = [original_params[param] for param in parameter_names]
    
    # Adjust initial guess to satisfy electrode work function constraints if needed
    # Find L3_E_v index and ensure it's >= W_R
    if 'L3_E_v' in parameter_names:
        l3_ev_idx = parameter_names.index('L3_E_v')
        W_R = 5.2
        if x0[l3_ev_idx] < W_R:
            # Set to minimum compliant value within bounds
            l3_ev_bounds = bounds[l3_ev_idx]
            x0[l3_ev_idx] = max(W_R, l3_ev_bounds[0])
            logging.info(f"Adjusted initial L3_E_v to {x0[l3_ev_idx]:.3f} eV for electrode compatibility")
    
    # Optimization using Differential Evolution (global optimizer)
    logging.info("Running Differential Evolution optimization...")
    
    result = differential_evolution(
        objective_function,
        bounds,
        args=(models_data, parameter_names),
        maxiter=maxiter,
        popsize=popsize,
        seed=42,
        disp=True,
        atol=1e-6,
        tol=1e-6
    )
    
    if result.success:
        logging.info(f"Optimization successful!")
        logging.info(f"   Iterations: {result.nit}")
        logging.info(f"   Function evaluations: {result.nfev}")
        logging.info(f"   Final objective: {result.fun:.6f}")
    else:
        logging.warning(f"Optimization did not converge: {result.message}")
    
    # Convert result back to parameter dictionary
    optimized_params = dict(zip(parameter_names, result.x))
    
    return optimized_params, result

def create_comparison_visualizations(original_params, optimized_params, original_pred, optimized_pred, device_config):
    """Create comprehensive comparison visualizations."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        matplotlib_available = True
    except ImportError:
        logging.error("Matplotlib not available. Skipping visualizations.")
        return
    
    # Set up plotting
    plt.style.use('default')
    sns.set_palette("husl")
    
    results_dir = 'results/optimize_device'
    
    # 1. Parameter Comparison Plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Device Parameter Optimization Comparison', fontsize=16, fontweight='bold')
    
    # Organize parameters by type
    thickness_params = [p for p in original_params.keys() if 'L' in p and p.endswith('_L')]
    energy_params = [p for p in original_params.keys() if 'E_' in p]
    doping_params = [p for p in original_params.keys() if 'N_' in p]
    
    # Plot thickness parameters
    ax = axes[0, 0]
    thickness_orig = [original_params[p] * 1e9 for p in thickness_params]  # Convert to nm
    thickness_opt = [optimized_params[p] * 1e9 for p in thickness_params]
    
    x_pos = np.arange(len(thickness_params))
    width = 0.35
    
    ax.bar(x_pos - width/2, thickness_orig, width, label='Original', alpha=0.7, color='lightblue')
    ax.bar(x_pos + width/2, thickness_opt, width, label='Optimized', alpha=0.7, color='orange')
    
    ax.set_xlabel('Layer Thickness Parameters')
    ax.set_ylabel('Thickness (nm)')
    ax.set_title('Layer Thickness Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([p.replace('_', ' ') for p in thickness_params], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (orig, opt) in enumerate(zip(thickness_orig, thickness_opt)):
        ax.text(i - width/2, orig + max(thickness_orig) * 0.01, f'{orig:.1f}nm', 
                ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax.text(i + width/2, opt + max(thickness_opt) * 0.01, f'{opt:.1f}nm', 
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Plot energy parameters
    ax = axes[0, 1]
    energy_orig = [original_params[p] for p in energy_params]
    energy_opt = [optimized_params[p] for p in energy_params]
    
    x_pos = np.arange(len(energy_params))
    
    ax.bar(x_pos - width/2, energy_orig, width, label='Original', alpha=0.7, color='lightgreen')
    ax.bar(x_pos + width/2, energy_opt, width, label='Optimized', alpha=0.7, color='red')
    
    ax.set_xlabel('Energy Level Parameters')
    ax.set_ylabel('Energy (eV)')
    ax.set_title('Energy Level Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([p.replace('_', ' ') for p in energy_params], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (orig, opt) in enumerate(zip(energy_orig, energy_opt)):
        ax.text(i - width/2, orig + max(energy_orig) * 0.01, f'{orig:.2f}eV', 
                ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax.text(i + width/2, opt + max(energy_opt) * 0.01, f'{opt:.2f}eV', 
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Plot doping parameters (log scale)
    ax = axes[1, 0]
    doping_orig = [np.log10(original_params[p]) for p in doping_params]
    doping_opt = [np.log10(optimized_params[p]) for p in doping_params]
    
    x_pos = np.arange(len(doping_params))
    
    ax.bar(x_pos - width/2, doping_orig, width, label='Original', alpha=0.7, color='purple')
    ax.bar(x_pos + width/2, doping_opt, width, label='Optimized', alpha=0.7, color='gold')
    
    ax.set_xlabel('Doping Concentration Parameters')
    ax.set_ylabel('log₁₀(Concentration) [cm⁻³]')
    ax.set_title('Doping Concentration Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([p.replace('_', ' ') for p in doping_params], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (orig, opt) in enumerate(zip(doping_orig, doping_opt)):
        ax.text(i - width/2, orig + max(doping_orig) * 0.01, f'10^{orig:.1f}', 
                ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax.text(i + width/2, opt + max(doping_opt) * 0.01, f'10^{opt:.1f}', 
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Performance comparison
    ax = axes[1, 1]
    
    # Extract performance metrics
    orig_mpp = original_pred.get('predicted_MPP', 0)
    opt_mpp = optimized_pred.get('predicted_MPP', 0)
    orig_recomb = original_pred.get('predicted_IntSRHn_mean', 1e30)
    opt_recomb = optimized_pred.get('predicted_IntSRHn_mean', 1e30)
    
    # Create performance comparison
    metrics = ['Efficiency\n(MPP)', 'Recombination\n(log scale)']
    original_values = [orig_mpp, np.log10(orig_recomb)]
    optimized_values = [opt_mpp, np.log10(opt_recomb)]
    
    x_pos = np.arange(len(metrics))
    
    bars1 = ax.bar(x_pos - width/2, original_values, width, label='Original', alpha=0.7, color='lightcoral')
    bars2 = ax.bar(x_pos + width/2, optimized_values, width, label='Optimized', alpha=0.7, color='lightseagreen')
    
    ax.set_xlabel('Performance Metrics')
    ax.set_ylabel('Value')
    ax.set_title('Performance Comparison')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    ax.text(0 - width/2, orig_mpp + abs(orig_mpp) * 0.05, f'{orig_mpp:.2f} W/cm²', 
            ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.text(0 + width/2, opt_mpp + abs(opt_mpp) * 0.05, f'{opt_mpp:.2f} W/cm²', 
            ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.text(1 - width/2, np.log10(orig_recomb) + abs(np.log10(orig_recomb)) * 0.05, f'10^{np.log10(orig_recomb):.1f}', 
            ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.text(1 + width/2, np.log10(opt_recomb) + abs(np.log10(opt_recomb)) * 0.05, f'10^{np.log10(opt_recomb):.1f}', 
            ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/optimization_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Performance Improvement Summary
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Calculate improvements
    mpp_improvement = ((opt_mpp - orig_mpp) / orig_mpp) * 100 if orig_mpp > 0 else 0
    recomb_change = ((opt_recomb - orig_recomb) / orig_recomb) * 100 if orig_recomb > 0 else 0
    
    improvements = [mpp_improvement, -recomb_change]  # Negative recomb change is good
    labels = ['Efficiency\nImprovement (%)', 'Recombination\nReduction (%)']
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    
    bars = ax.bar(labels, improvements, color=colors, alpha=0.7)
    ax.set_ylabel('Percentage Change (%)')
    ax.set_title('Performance Improvement Summary')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Add value labels
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + (abs(height) * 0.05 if height > 0 else -abs(height) * 0.05),
                f'{imp:+.2f}%', ha='center', va='bottom' if height > 0 else 'top', 
                fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{results_dir}/performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info("Comparison visualizations created successfully!")
    logging.info(f"   Parameter comparison: {results_dir}/optimization_comparison.png")
    logging.info(f"   Performance comparison: {results_dir}/performance_comparison.png")

def create_parameter_changes_table(original_params, optimized_params, original_pred, optimized_pred):
    """Create a detailed table of parameter changes."""
    try:
        import matplotlib.pyplot as plt
        matplotlib_available = True
    except ImportError:
        logging.error("Matplotlib not available. Skipping table creation.")
        return
    
    results_dir = 'results/optimize_device'
    
    # Prepare data for table
    table_data = []
    
    for param in original_params.keys():
        original_val = original_params[param]
        optimized_val = optimized_params[param]
        change = optimized_val - original_val
        percent_change = (change / original_val) * 100 if original_val != 0 else 0
        
        # Format values based on parameter type with smart formatting
        if 'L' in param and param.endswith('_L'):  # Thickness
            # Convert to nm and format smartly
            orig_nm = original_val * 1e9
            opt_nm = optimized_val * 1e9
            change_nm = change * 1e9
            
            if orig_nm >= 100:
                orig_str = f"{orig_nm:.0f}nm"
            elif orig_nm >= 10:
                orig_str = f"{orig_nm:.1f}nm"
            else:
                orig_str = f"{orig_nm:.2f}nm"
                
            if opt_nm >= 100:
                opt_str = f"{opt_nm:.0f}nm"
            elif opt_nm >= 10:
                opt_str = f"{opt_nm:.1f}nm"
            else:
                opt_str = f"{opt_nm:.2f}nm"
                
            if abs(change_nm) >= 100:
                change_str = f"{change_nm:+.0f}nm"
            elif abs(change_nm) >= 10:
                change_str = f"{change_nm:+.1f}nm"
            else:
                change_str = f"{change_nm:+.2f}nm"
                
        elif 'E_' in param:  # Energy
            orig_str = f"{original_val:.2f}eV"
            opt_str = f"{optimized_val:.2f}eV"
            change_str = f"{change:+.3f}eV"
            
        elif 'N_' in param:  # Doping
            # Smart formatting for doping concentrations
            if original_val >= 1e20:
                orig_str = f"{original_val/1e20:.1f}e20"
            else:
                orig_str = f"{original_val:.1e}"
                
            if optimized_val >= 1e20:
                opt_str = f"{optimized_val/1e20:.1f}e20"
            else:
                opt_str = f"{optimized_val:.1e}"
                
            if abs(change) >= 1e20:
                change_str = f"{change/1e20:+.1f}e20"
            else:
                change_str = f"{change:+.1e}"
        else:
            orig_str = f"{original_val:.3f}"
            opt_str = f"{optimized_val:.3f}"
            change_str = f"{change:+.3f}"
        
        table_data.append([
            param.replace('_', ' '),
            orig_str,
            opt_str,
            change_str,
            f"{percent_change:+.1f}%"
        ])
    
    # Add performance metrics
    orig_mpp = original_pred.get('predicted_MPP', 0)
    opt_mpp = optimized_pred.get('predicted_MPP', 0)
    mpp_change = opt_mpp - orig_mpp
    mpp_percent = (mpp_change / orig_mpp) * 100 if orig_mpp > 0 else 0
    
    orig_recomb = original_pred.get('predicted_IntSRHn_mean', 1e30)
    opt_recomb = optimized_pred.get('predicted_IntSRHn_mean', 1e30)
    recomb_change = opt_recomb - orig_recomb
    recomb_percent = (recomb_change / orig_recomb) * 100 if orig_recomb > 0 else 0
    
    # Add separator and performance metrics
    table_data.append(['─' * 20, '─' * 15, '─' * 15, '─' * 15, '─' * 10])
    table_data.append([
        'MPP Efficiency',
        f"{orig_mpp:.2f} W/cm²",
        f"{opt_mpp:.2f} W/cm²",
        f"{mpp_change:+.2f} W/cm²",
        f"{mpp_percent:+.1f}%"
    ])
    
    # Smart format for recombination values
    if orig_recomb >= 1e10:
        orig_recomb_str = f"{orig_recomb:.1e}"
    else:
        orig_recomb_str = f"{orig_recomb:.2f}"
        
    if opt_recomb >= 1e10:
        opt_recomb_str = f"{opt_recomb:.1e}"
    else:
        opt_recomb_str = f"{opt_recomb:.2f}"
        
    if abs(recomb_change) >= 1e10:
        recomb_change_str = f"{recomb_change:+.1e}"
    else:
        recomb_change_str = f"{recomb_change:+.2f}"
    
    table_data.append([
        'Recombination',
        orig_recomb_str,
        opt_recomb_str,
        recomb_change_str,
        f"{recomb_percent:+.1f}%"
    ])
    
    # Create table plot
    fig, ax = plt.subplots(figsize=(14, len(table_data) * 0.5 + 2))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=['Parameter', 'Original', 'Optimized', 'Change', '% Change'],
        cellLoc='center',
        loc='center',
        colWidths=[0.25, 0.2, 0.2, 0.2, 0.15]
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.0)  # Make rows taller
    
    # Color code the header
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color code performance rows
    performance_start = len(table_data) - 2
    for i in range(performance_start, len(table_data)):
        for j in range(5):
            if i == performance_start - 1:  # Separator row
                table[(i + 1, j)].set_facecolor('#E0E0E0')
            else:
                table[(i + 1, j)].set_facecolor('#E8F5E8')
    
    plt.title('Parameter Optimization Summary Table', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(f'{results_dir}/parameter_changes_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Parameter changes table: {results_dir}/parameter_changes_table.png")

def save_optimization_report(original_params, optimized_params, original_pred, optimized_pred, 
                           device_config, optimization_result):
    """Save detailed optimization report to JSON."""
    results_dir = 'results/optimize_device'
    
    # Calculate improvements
    orig_mpp = original_pred.get('predicted_MPP', 0)
    opt_mpp = optimized_pred.get('predicted_MPP', 0)
    mpp_improvement = ((opt_mpp - orig_mpp) / orig_mpp) * 100 if orig_mpp != 0 else 0
    
    orig_recomb = original_pred.get('predicted_IntSRHn_mean', 1e30)
    opt_recomb = optimized_pred.get('predicted_IntSRHn_mean', 1e30)
    recomb_change = ((opt_recomb - orig_recomb) / orig_recomb) * 100 if orig_recomb > 0 else 0
    
    report = {
        "optimization_date": datetime.now().isoformat(),
        "device_type": device_config.get('device_type', 'Unknown'),
        "optimization_summary": {
            "success": optimization_result.success,
            "iterations": optimization_result.nit,
            "function_evaluations": optimization_result.nfev,
            "final_objective_value": optimization_result.fun
        },
        "performance_comparison": {
            "original": {
                "MPP": orig_mpp,
                "IntSRHn_mean": orig_recomb,
                "efficiency_percent": (orig_mpp / 1000) * 100  # Assuming 1000 W/m² reference
            },
            "optimized": {
                "MPP": opt_mpp,
                "IntSRHn_mean": opt_recomb,
                "efficiency_percent": (opt_mpp / 1000) * 100
            },
            "improvements": {
                "MPP_improvement_percent": mpp_improvement,
                "recombination_change_percent": recomb_change,
                "efficiency_improvement_absolute": (opt_mpp - orig_mpp) / 1000 * 100
            }
        },
        "parameter_changes": {},
        "original_parameters": original_params,
        "optimized_parameters": optimized_params,
        "optimization_recommendations": []
    }
    
    # Add parameter changes
    for param in original_params.keys():
        original_val = original_params[param]
        optimized_val = optimized_params[param]
        change = optimized_val - original_val
        percent_change = (change / original_val) * 100 if original_val != 0 else 0
        
        report["parameter_changes"][param] = {
            "original": original_val,
            "optimized": optimized_val,
            "absolute_change": change,
            "percent_change": percent_change
        }
    
    # Add recommendations
    if mpp_improvement > 5:
        report["optimization_recommendations"].append(
            f"✅ Significant efficiency improvement of {mpp_improvement:.1f}% achieved"
        )
    elif mpp_improvement > 0:
        report["optimization_recommendations"].append(
            f"✅ Modest efficiency improvement of {mpp_improvement:.1f}% achieved"
        )
    else:
        report["optimization_recommendations"].append(
            f"⚠️ No efficiency improvement achieved. Consider different optimization constraints."
        )
    
    if recomb_change < -5:
        report["optimization_recommendations"].append(
            f"✅ Recombination reduced by {abs(recomb_change):.1f}%"
        )
    elif recomb_change > 5:
        report["optimization_recommendations"].append(
            f"⚠️ Recombination increased by {recomb_change:.1f}%. Consider tighter constraints."
        )
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    # Save report
    report_path = f'{results_dir}/optimization_report.json'
    with open(report_path, 'w') as f:
        json.dump(convert_numpy_types(report), f, indent=2)
    
    logging.info(f"Optimization report saved: {report_path}")
    return report

def main():
    """Main optimization workflow."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Optimize device parameters for maximum efficiency')
    parser.add_argument('--maxiter', type=int, default=1000, help='Maximum iterations for optimization')
    parser.add_argument('--popsize', type=int, default=15, help='Population size for differential evolution')
    args = parser.parse_args()
    
    # Setup
    log_file = setup_logging()
    logging.info("=== DEVICE PARAMETER OPTIMIZATION ===")
    logging.info(f"Log file: {log_file}")
    
    try:
        # 1. Load device parameters
        logging.info("\n=== Step 1: Loading Device Parameters ===")
        original_params, device_config = load_device_parameters()
        
        # 2. Load trained models
        logging.info("\n=== Step 2: Loading Trained Models ===")
        models_data = load_optimization_models()
        
        # 3. Predict original performance
        logging.info("\n=== Step 3: Predicting Original Performance ===")
        original_pred = predict_performance(original_params, models_data)
        
        logging.info("Original Performance:")
        for key, value in original_pred.items():
            if 'MPP' in key:
                logging.info(f"   {key}: {value:.4f} W/cm²")
            else:
                logging.info(f"   {key}: {value:.2e}")
        
        # 4. Optimize parameters
        logging.info("\n=== Step 4: Optimizing Parameters ===")
        optimized_params, optimization_result = optimize_parameters(
            original_params, models_data, 
            maxiter=args.maxiter, popsize=args.popsize
        )
        
        # 5. Predict optimized performance
        logging.info("\n=== Step 5: Predicting Optimized Performance ===")
        optimized_pred = predict_performance(optimized_params, models_data)
        
        logging.info("Optimized Performance:")
        for key, value in optimized_pred.items():
            if 'MPP' in key:
                logging.info(f"   {key}: {value:.4f} W/cm²")
            else:
                logging.info(f"   {key}: {value:.2e}")
        
        # 6. Calculate improvements
        logging.info("\n=== Step 6: Performance Analysis ===")
        orig_mpp = original_pred.get('predicted_MPP', 0)
        opt_mpp = optimized_pred.get('predicted_MPP', 0)
        mpp_improvement = ((opt_mpp - orig_mpp) / orig_mpp) * 100 if orig_mpp != 0 else 0
        
        orig_recomb = original_pred.get('predicted_IntSRHn_mean', 1e30)
        opt_recomb = optimized_pred.get('predicted_IntSRHn_mean', 1e30)
        recomb_change = ((opt_recomb - orig_recomb) / orig_recomb) * 100 if orig_recomb > 0 else 0
        
        logging.info(f"Efficiency Improvement: {mpp_improvement:+.2f}%")
        logging.info(f"Recombination Change: {recomb_change:+.2f}%")
        
        # 7. Create visualizations
        logging.info("\n=== Step 7: Creating Visualizations ===")
        create_comparison_visualizations(original_params, optimized_params, 
                                       original_pred, optimized_pred, device_config)
        
        create_parameter_changes_table(original_params, optimized_params, 
                                     original_pred, optimized_pred)
        
        # 8. Save optimization report
        logging.info("\n=== Step 8: Saving Optimization Report ===")
        report = save_optimization_report(original_params, optimized_params, 
                                        original_pred, optimized_pred, 
                                        device_config, optimization_result)
        
        # Final summary
        logging.info("\n=== OPTIMIZATION COMPLETE ===")
        logging.info(f"Results saved to: results/optimize_device/")
        logging.info(f"Visualizations: 3 plots created")
        logging.info(f"Report: optimization_report.json")
        logging.info(f"Log: optimization_log.txt")
        
        if mpp_improvement > 0:
            logging.info(f"SUCCESS: {mpp_improvement:.1f}% efficiency improvement achieved!")
        else:
            logging.info("Consider adjusting optimization constraints for better results.")
            
    except Exception as e:
        logging.error(f"Optimization failed: {e}")
        raise

if __name__ == "__main__":
    main()
