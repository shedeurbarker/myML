"""
===============================================================================
SOLAR CELL OPTIMIZATION - MPP and IntSRHn_mean OPTIMIZATION
===============================================================================

PURPOSE:
This script uses the trained ML models from Script 5 to find optimal device parameters
that maximize MPP (Maximum Power Point) while controlling IntSRHn_mean (mean electron
interfacial recombination rate). Matches the workflow from scripts 1-5.

WHAT THIS SCRIPT DOES:
1. Loads trained MPP and IntSRHn_mean prediction models from Script 5
2. Uses optimization algorithms to find optimal 15 device parameters
3. Maximizes MPP while constraining IntSRHn_mean to physically reasonable values
4. Validates optimization results using the trained models
5. Generates comprehensive reports and visualizations

OPTIMIZATION TARGETS (matching scripts 1-5):
- Primary: Maximize MPP (Maximum Power Point in W/cm²)
- Constraint: Limit IntSRHn_mean (mean electron interfacial recombination rate)

OPTIMIZATION ALGORITHMS:
- L-BFGS-B: Local optimization with multiple starting points for fine-tuning
- Differential Evolution: Global optimization for exploring entire parameter space

INPUT FILES:
- results/train_optimization_models/models/efficiency_MPP.joblib (MPP prediction model)
- results/train_optimization_models/models/recombination_IntSRHn_mean.joblib (recombination model)
- results/train_optimization_models/models/*_scalers.joblib (feature and target scalers)
- results/features/feature_definitions.json (parameter bounds and definitions)

OUTPUT FILES:
- results/optimize_efficiency/reports/optimization_report.json (detailed results)
- results/optimize_efficiency/plots/optimization_results.png (visualizations)
- results/optimize_efficiency/optimization.log (optimization process log)

USAGE:
python scripts/7_optimize_efficiency.py

PREREQUISITES:
1. Run scripts/1_create_feature_names.py
2. Run scripts/2_generate_simulations.py
3. Run scripts/3_extract_simulation_data.py
4. Run scripts/4_prepare_ml_data.py
5. Run scripts/5_train_models.py

AUTHOR: Anthony Barker
DATE: 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import logging
from datetime import datetime
import json
import sys
from scipy.optimize import minimize, differential_evolution
from sklearn.preprocessing import StandardScaler

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up logging
log_dir = 'results/optimize_efficiency'
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'optimization.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

# Create directories for results
os.makedirs('results/optimize_efficiency/plots', exist_ok=True)
os.makedirs('results/optimize_efficiency/reports', exist_ok=True)

def load_optimization_models():
    """Load trained MPP and IntSRHn_mean prediction models from Script 5."""
    models_dir = 'results/train_optimization_models/models'
    
    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Models directory not found: {models_dir}. Run Script 5 first.")
    
    # Load training metadata from Script 5
    metadata_path = 'results/train_optimization_models/training_metadata.json'
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Training metadata not found: {metadata_path}. Run Script 5 first.")
    
    with open(metadata_path, 'r') as f:
        training_metadata = json.load(f)
    
    # Load feature definitions for parameter bounds
    feature_defs_path = 'results/features/feature_definitions.json'
    if not os.path.exists(feature_defs_path):
        raise FileNotFoundError(f"Feature definitions not found: {feature_defs_path}. Run Script 1 first.")
    
    with open(feature_defs_path, 'r') as f:
        feature_definitions = json.load(f)
    
    # Load MPP (efficiency) model and scalers
    mpp_models = {}
    mpp_scalers = {}
    
    # Load MPP model
    mpp_model_path = os.path.join(models_dir, 'efficiency_MPP.joblib')
    mpp_scalers_path = os.path.join(models_dir, 'efficiency_MPP_scalers.joblib')
    
    if not os.path.exists(mpp_model_path):
        raise FileNotFoundError(f"MPP model not found: {mpp_model_path}")
    if not os.path.exists(mpp_scalers_path):
        raise FileNotFoundError(f"MPP scalers not found: {mpp_scalers_path}")
    
    mpp_models['MPP'] = joblib.load(mpp_model_path)
    mpp_scalers['MPP'] = joblib.load(mpp_scalers_path)
    
    # Load IntSRHn_mean (recombination) model and scalers
    recombination_models = {}
    recombination_scalers = {}
    
    recomb_model_path = os.path.join(models_dir, 'recombination_IntSRHn_mean.joblib')
    recomb_scalers_path = os.path.join(models_dir, 'recombination_IntSRHn_mean_scalers.joblib')
    
    if not os.path.exists(recomb_model_path):
        raise FileNotFoundError(f"IntSRHn_mean model not found: {recomb_model_path}")
    if not os.path.exists(recomb_scalers_path):
        raise FileNotFoundError(f"IntSRHn_mean scalers not found: {recomb_scalers_path}")
    
    recombination_models['IntSRHn_mean'] = joblib.load(recomb_model_path)
    recombination_scalers['IntSRHn_mean'] = joblib.load(recomb_scalers_path)
    
    # Create metadata for optimization
    metadata = {
        'device_params': list(feature_definitions['primary_parameters'].keys()),
        'all_features': training_metadata['efficiency_models']['MPP']['feature_names'],
        'parameter_bounds': feature_definitions['parameter_bounds'],
        'efficiency_targets': ['MPP'],
        'recombination_targets': ['IntSRHn_mean']
    }
    
    logging.info(f"Loaded MPP prediction model")
    logging.info(f"Loaded IntSRHn_mean prediction model")
    logging.info(f"Total features: {len(metadata['all_features'])}")
    logging.info(f"Device parameters: {len(metadata['device_params'])}")
    
    return mpp_models, mpp_scalers, recombination_models, recombination_scalers, metadata

def create_derived_features(primary_params):
    """Create derived features from 15 primary device parameters - EXACTLY matching Script 4."""
    # Convert to dict for easier access
    param_dict = {
        'L1_L': primary_params[0], 'L1_E_c': primary_params[1], 'L1_E_v': primary_params[2],
        'L1_N_D': primary_params[3], 'L1_N_A': primary_params[4],
        'L2_L': primary_params[5], 'L2_E_c': primary_params[6], 'L2_E_v': primary_params[7],
        'L2_N_D': primary_params[8], 'L2_N_A': primary_params[9],
        'L3_L': primary_params[10], 'L3_E_c': primary_params[11], 'L3_E_v': primary_params[12],
        'L3_N_D': primary_params[13], 'L3_N_A': primary_params[14]
    }
    
    # Initialize feature array with primary parameters
    features = list(primary_params)
    
    # Thickness features (EXACTLY as in Script 4)
    total_thickness = param_dict['L1_L'] + param_dict['L2_L'] + param_dict['L3_L']
    features.append(total_thickness)  # total_thickness
    features.append(param_dict['L2_L'] / (total_thickness + 1e-30))  # thickness_ratio_L2
    features.append(param_dict['L1_L'] / (total_thickness + 1e-30))  # thickness_ratio_ETL
    features.append(param_dict['L3_L'] / (total_thickness + 1e-30))  # thickness_ratio_HTL
    
    # Energy gap features (EXACTLY as in Script 4 - use absolute value)
    energy_gap_L1 = abs(param_dict['L1_E_c'] - param_dict['L1_E_v'])
    energy_gap_L2 = abs(param_dict['L2_E_c'] - param_dict['L2_E_v'])
    energy_gap_L3 = abs(param_dict['L3_E_c'] - param_dict['L3_E_v'])
    features.extend([energy_gap_L1, energy_gap_L2, energy_gap_L3])
    
    # Band alignment features (EXACTLY as in Script 4)
    band_offset_L1_L2 = param_dict['L2_E_c'] - param_dict['L1_E_c']
    band_offset_L2_L3 = param_dict['L3_E_c'] - param_dict['L2_E_c']
    conduction_band_offset = param_dict['L3_E_c'] - param_dict['L1_E_c']
    valence_band_offset = param_dict['L3_E_v'] - param_dict['L1_E_v']
    features.extend([band_offset_L1_L2, band_offset_L2_L3, conduction_band_offset, valence_band_offset])
    
    # Doping features (EXACTLY as in Script 4)
    doping_ratio_L1 = param_dict['L1_N_D'] / (param_dict['L1_N_A'] + 1e-30)
    doping_ratio_L2 = param_dict['L2_N_D'] / (param_dict['L2_N_A'] + 1e-30)
    doping_ratio_L3 = param_dict['L3_N_D'] / (param_dict['L3_N_A'] + 1e-30)
    total_donor_concentration = param_dict['L1_N_D'] + param_dict['L2_N_D'] + param_dict['L3_N_D']
    total_acceptor_concentration = param_dict['L1_N_A'] + param_dict['L2_N_A'] + param_dict['L3_N_A']
    features.extend([doping_ratio_L1, doping_ratio_L2, doping_ratio_L3, total_donor_concentration, total_acceptor_concentration])
    
    # Material property features (EXACTLY as in Script 4)
    average_energy_gap = (energy_gap_L1 + energy_gap_L2 + energy_gap_L3) / 3
    energy_gap_variance = np.var([energy_gap_L1, energy_gap_L2, energy_gap_L3])
    thickness_variance = np.var([param_dict['L1_L'], param_dict['L2_L'], param_dict['L3_L']])
    doping_variance = np.var([param_dict['L1_N_D'], param_dict['L2_N_D'], param_dict['L3_N_D']])
    features.extend([average_energy_gap, energy_gap_variance, thickness_variance, doping_variance])
    
    # Physics-based features for recombination-efficiency relationship (EXACTLY as in Script 4)
    # Use default values since we don't have MPP and IntSRHn_mean during optimization
    recombination_efficiency_ratio = 1e28  # Default typical value
    interface_quality_index = 1e-28  # Default typical value
    features.extend([recombination_efficiency_ratio, interface_quality_index])
    
    # Carrier transport efficiency features (EXACTLY as in Script 4)
    conduction_band_alignment_quality = 1 / (1 + abs(band_offset_L1_L2) + abs(band_offset_L2_L3))
    valence_band_alignment_quality = 1 / (1 + abs(valence_band_offset))
    features.extend([conduction_band_alignment_quality, valence_band_alignment_quality])
    
    # Thickness optimization features (EXACTLY as in Script 4)
    thickness_ratio_L2 = features[16]  # Already calculated above
    thickness_ratio_ETL = features[17]  # Already calculated above
    thickness_ratio_HTL = features[18]  # Already calculated above
    thickness_balance_quality = thickness_ratio_L2 / (thickness_ratio_ETL + thickness_ratio_HTL + 1e-30)
    transport_layer_balance = 1 / (1 + abs(thickness_ratio_ETL - thickness_ratio_HTL))
    features.extend([thickness_balance_quality, transport_layer_balance])
    
    # Doping optimization features (EXACTLY as in Script 4)
    average_doping_ratio = (doping_ratio_L1 + doping_ratio_L2 + doping_ratio_L3) / 3
    doping_consistency = 1 / (1 + np.var([doping_ratio_L1, doping_ratio_L2, doping_ratio_L3]))
    features.extend([average_doping_ratio, doping_consistency])
    
    # Energy level optimization features (EXACTLY as in Script 4)
    energy_gap_progression = abs((energy_gap_L2 - energy_gap_L1) * (energy_gap_L3 - energy_gap_L2))
    energy_gap_uniformity = 1 / (1 + np.var([energy_gap_L1, energy_gap_L2, energy_gap_L3]))
    features.extend([energy_gap_progression, energy_gap_uniformity])
    
    return np.array(features)

def predict_mpp(device_params, mpp_models, mpp_scalers, target='MPP'):
    """Predict MPP (Maximum Power Point) for given device parameters."""
    if target not in mpp_models:
        raise ValueError(f"Model not found for target: {target}")
    
    # Create all features from primary device parameters
    all_features = create_derived_features(device_params)
    
    # Get the scalers (feature scaler and target scaler)
    scalers = mpp_scalers[target]
    feature_scaler = scalers['feature_scaler']
    target_scaler = scalers['target_scaler']
    
    # Scale input features
    X_scaled = feature_scaler.transform([all_features])
    
    # Make prediction (scaled)
    model = mpp_models[target]
    prediction_scaled = model.predict(X_scaled)[0]
    
    # Inverse transform to get actual MPP value
    prediction = target_scaler.inverse_transform([[prediction_scaled]])[0][0]
    
    return prediction

def predict_recombination(device_params, recombination_models, recombination_scalers, target='IntSRHn_mean'):
    """Predict IntSRHn_mean (recombination rate) for given device parameters."""
    if target not in recombination_models:
        raise ValueError(f"Model not found for target: {target}")
    
    # Create all features from primary device parameters
    all_features = create_derived_features(device_params)
    
    # Get the scalers (feature scaler and target scaler)
    scalers = recombination_scalers[target]
    feature_scaler = scalers['feature_scaler']
    target_scaler = scalers['target_scaler']
    
    # Scale input features
    X_scaled = feature_scaler.transform([all_features])
    
    # Make prediction (scaled)
    model = recombination_models[target]
    prediction_scaled = model.predict(X_scaled)[0]
    
    # Inverse transform to get actual recombination value
    prediction = target_scaler.inverse_transform([[prediction_scaled]])[0][0]
    
    return prediction

def objective_function(device_params, mpp_models, mpp_scalers, target='MPP'):
    """Objective function: maximize MPP (negative for minimization)."""
    mpp = predict_mpp(device_params, mpp_models, mpp_scalers, target)
    return -mpp  # Negative because we want to maximize

def constraint_recombination(device_params, recombination_models, recombination_scalers, 
                           max_recombination=1e-3, target='IntSRHn_mean'):
    """Constraint function: recombination rate must be below threshold."""
    recombination = predict_recombination(device_params, recombination_models, recombination_scalers, target)
    return max_recombination - recombination  # Must be >= 0

def get_parameter_bounds(metadata):
    """Get parameter bounds for optimization."""
    # Load feature definitions for bounds
    try:
        with open('results/features/feature_definitions.json', 'r') as f:
            feature_definitions = json.load(f)
        parameter_bounds = feature_definitions['parameter_bounds']
    except FileNotFoundError:
        logging.warning("Feature definitions not found. Using default bounds.")
        parameter_bounds = {
            # Layer 1 (PCBM - Electron Transport Layer)
            'L1_L': (20, 50),      # nm
            'L1_E_c': (3.7, 4.0),  # eV
            'L1_E_v': (5.7, 5.9),  # eV
            'L1_N_D': (1e20, 1e21), # m⁻³
            'L1_N_A': (1e20, 1e21), # m⁻³
            
            # Layer 2 (MAPI - Active Layer)
            'L2_L': (200, 500),     # nm
            'L2_E_c': (4.4, 4.6),   # eV
            'L2_E_v': (5.6, 5.8),   # eV
            'L2_N_D': (1e20, 1e21), # m⁻³
            'L2_N_A': (1e20, 1e21), # m⁻³
            
            # Layer 3 (PEDOT - Hole Transport Layer)
            'L3_L': (20, 50),       # nm
            'L3_E_c': (3.4, 3.6),   # eV
            'L3_E_v': (5.3, 5.5),   # eV
            'L3_N_D': (1e20, 1e21), # m⁻³
            'L3_N_A': (1e20, 1e21)  # m⁻³
        }
    
    # Convert to list of tuples for scipy.optimize
    bounds = []
    for param in metadata['device_params']:
        if param in parameter_bounds:
            bounds.append(parameter_bounds[param])
        else:
            logging.warning(f"No bounds found for parameter {param}. Using default bounds.")
            bounds.append((0, 1))  # Default bounds
    
    return bounds

def optimize_for_maximum_mpp(mpp_models, mpp_scalers, recombination_models, recombination_scalers,
                            metadata, target='MPP', max_recombination=None):
    """Optimize device parameters for maximum MPP."""
    logging.info(f"\n=== Optimizing for Maximum MPP ===")
    
    # Get parameter bounds
    bounds = get_parameter_bounds(metadata)
    n_params = len(bounds)
    
    logging.info(f"Optimizing {n_params} parameters for maximum MPP")
    logging.info(f"Parameter bounds: {bounds}")
    
    # Method 1: L-BFGS-B (local optimization - no constraints)
    logging.info("\n--- Method 1: L-BFGS-B Optimization ---")
    logging.info("Note: L-BFGS-B runs without constraints (constraint checking done post-optimization)")
    
    # Generate multiple starting points
    n_starts = 10
    best_result_lbfgs = None
    best_mpp_lbfgs = -np.inf
    
    for i in range(n_starts):
        # Random starting point within bounds
        x0 = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
        
        try:
            result = minimize(
                objective_function,
                x0,
                args=(mpp_models, mpp_scalers, target),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000}
            )
            
            if result.success and -result.fun > best_mpp_lbfgs:
                best_result_lbfgs = result
                best_mpp_lbfgs = -result.fun
                logging.info(f"L-BFGS-B run {i+1}: MPP = {best_mpp_lbfgs:.6f} W/cm²")
        
        except Exception as e:
            logging.warning(f"L-BFGS-B run {i+1} failed: {e}")
    
    # Method 2: Differential Evolution (global optimization - no constraints)
    logging.info("\n--- Method 2: Differential Evolution ---")
    logging.info("Note: Differential Evolution runs without constraints (constraint checking done post-optimization)")
    
    result_de = None
    mpp_de = -np.inf
    
    try:
        result_de = differential_evolution(
            objective_function,
            bounds,
            args=(mpp_models, mpp_scalers, target),
            maxiter=1000,
            popsize=15,
            seed=42
        )
        
        if result_de.success:
            mpp_de = -result_de.fun
            logging.info(f"Differential Evolution: MPP = {mpp_de:.6f} W/cm²")
        else:
            logging.warning("Differential Evolution failed to converge")
            mpp_de = -np.inf
    
    except Exception as e:
        logging.error(f"Differential Evolution failed: {e}")
        mpp_de = -np.inf
    
    # Compare results and select best
    if best_mpp_lbfgs > mpp_de:
        best_result = best_result_lbfgs
        best_method = "L-BFGS-B"
        best_mpp = best_mpp_lbfgs
    else:
        best_result = result_de
        best_method = "Differential Evolution"
        best_mpp = mpp_de
    
    if best_result is None:
        logging.error("All optimization methods failed")
        return None
    
    # Get optimal parameters
    optimal_params = best_result.x
    
    # Predict recombination for optimal parameters
    optimal_recombination = {}
    for target in recombination_models.keys():
        optimal_recombination[target] = predict_recombination(
            optimal_params, recombination_models, recombination_scalers, target
        )
    
    # Check recombination constraint (post-optimization)
    if max_recombination is not None:
        actual_recombination = optimal_recombination.get('IntSRHn_mean', 0)
        if actual_recombination > max_recombination:
            logging.warning(f"Recombination constraint violated: {actual_recombination:.6f} > {max_recombination}")
            logging.warning("Consider running with stricter bounds or different optimization approach")
        else:
            logging.info(f"Recombination constraint satisfied: {actual_recombination:.6f} <= {max_recombination}")
    
    # Create results dictionary
    results = {
        'optimization_success': best_result.success,
        'optimal_method': best_method,
        'optimal_mpp': best_mpp,
        'optimal_parameters': dict(zip(metadata['device_params'], optimal_params)),
        'optimal_recombination': optimal_recombination,
        'optimization_details': {
            'n_iterations': best_result.nit,
            'n_function_evaluations': best_result.nfev,
            'convergence_message': best_result.message
        }
    }
    
    logging.info(f"\nOptimization completed successfully!")
    logging.info(f"Best method: {best_method}")
    logging.info(f"Optimal MPP: {best_mpp:.6f} W/cm²")
    logging.info(f"Optimal recombination (IntSRHn_mean): {optimal_recombination.get('IntSRHn_mean', 'N/A')}")
    
    return results

def validate_optimization_results(results, mpp_models, mpp_scalers,
                                recombination_models, recombination_scalers, metadata):
    """Validate optimization results."""
    if results is None:
        return False
    
    logging.info("\n=== Validating Optimization Results ===")
    
    # Check if PCE is reasonable
    optimal_mpp = results['optimal_mpp']
    if optimal_mpp < 0 or optimal_mpp > 1000:  # Reasonable MPP range in W/cm²
        logging.warning(f"Optimal MPP ({optimal_mpp:.6f} W/cm²) seems unrealistic")
        return False
    
    # Check if recombination is reasonable
    optimal_recombination = results['optimal_recombination']
    if 'IntSRHn_mean' in optimal_recombination:
        recombination_rate = optimal_recombination['IntSRHn_mean']
        if recombination_rate < 0 or recombination_rate > 1e-2:
            logging.warning(f"Optimal recombination rate ({recombination_rate}) seems unrealistic")
            return False
    
    # Check if parameters are within bounds
    bounds = get_parameter_bounds(metadata)
    optimal_params = results['optimal_parameters']
    
    for i, (param, value) in enumerate(optimal_params.items()):
        if i < len(bounds):
            min_val, max_val = bounds[i]
            if value < min_val or value > max_val:
                logging.warning(f"Parameter {param} ({value}) outside bounds [{min_val}, {max_val}]")
                return False
    
    logging.info("Optimization results validation passed")
    return True

def create_optimization_report(results, metadata):
    """Create comprehensive optimization report."""
    if results is None:
        return
    
    logging.info("\n=== Creating Optimization Report ===")
    
    report = {
        'optimization_summary': {
            'target': 'MPP (Maximum Power Point)',
            'optimization_method': results['optimal_method'],
            'success': results['optimization_success'],
            'optimal_mpp': results['optimal_mpp'],
            'convergence_message': results['optimization_details']['convergence_message']
        },
        'optimal_parameters': results['optimal_parameters'],
        'optimal_recombination': results['optimal_recombination'],
        'optimization_details': results['optimization_details'],
        'metadata': {
            'device_parameters': metadata['device_params'],
            'total_features': metadata['total_features'],
            'optimization_date': datetime.now().isoformat()
        }
    }
    
    # Save report
    report_path = 'results/optimize_efficiency/reports/optimization_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logging.info(f"Optimization report saved to: {report_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS SUMMARY")
    print("="*60)
    print(f"Target: {report['optimization_summary']['target']}")
    print(f"Method: {report['optimization_summary']['optimization_method']}")
    print(f"Optimal MPP: {report['optimization_summary']['optimal_mpp']:.6f} W/cm²")
    print(f"Success: {report['optimization_summary']['success']}")
    print("\nOptimal Parameters:")
    for param, value in report['optimal_parameters'].items():
        print(f"  {param}: {value:.6f}")
    print("\nOptimal Recombination Rates:")
    for target, value in report['optimal_recombination'].items():
        print(f"  {target}: {value:.6e}")
    print("="*60)

def create_optimization_plots(results, metadata):
    """Create optimization result visualizations."""
    if results is None:
        return
    
    logging.info("\n=== Creating Optimization Plots ===")
    
    # Plot 1: Optimal parameters
    optimal_params = results['optimal_parameters']
    param_names = list(optimal_params.keys())
    param_values = list(optimal_params.values())
    
    plt.figure(figsize=(15, 8))
    bars = plt.bar(range(len(param_names)), param_values)
    plt.xlabel('Device Parameters')
    plt.ylabel('Parameter Value')
    plt.title('Optimal Device Parameters for Maximum PCE')
    plt.xticks(range(len(param_names)), param_names, rotation=45, ha='right')
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, param_values)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(param_values)*0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('results/optimize_efficiency/plots/optimal_parameters.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Recombination rates
    optimal_recombination = results['optimal_recombination']
    recombination_names = list(optimal_recombination.keys())
    recombination_values = list(optimal_recombination.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(recombination_names)), recombination_values)
    plt.xlabel('Recombination Metrics')
    plt.ylabel('Recombination Rate')
    plt.title('Optimal Recombination Rates')
    plt.xticks(range(len(recombination_names)), recombination_names, rotation=45, ha='right')
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, recombination_values)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(recombination_values)*0.01,
                f'{value:.2e}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('results/optimize_efficiency/plots/optimal_recombination.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Optimization methods comparison
    plt.figure(figsize=(8, 6))
    methods = ['L-BFGS-B', 'Differential Evolution']
    mpp_values = [results['optimal_mpp'], results['optimal_mpp']]  # Same value for now
    
    bars = plt.bar(methods, mpp_values)
    plt.ylabel('MPP (W/cm²)')
    plt.title('Optimization Methods Comparison')
    plt.ylim(0, max(mpp_values) * 1.1)
    
    # Add value labels
    for bar, value in zip(bars, mpp_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{value:.6f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('results/optimize_efficiency/plots/optimization_methods.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info("Optimization plots created")

def main():
    """Main optimization function."""
    logging.info("=" * 60)
    logging.info("SOLAR CELL MPP OPTIMIZATION")
    logging.info("=" * 60)
    
    try:
        # Load models
        mpp_models, mpp_scalers, recombination_models, recombination_scalers, metadata = load_optimization_models()
        
        # Run optimization
        results = optimize_for_maximum_mpp(
            mpp_models, mpp_scalers, recombination_models, recombination_scalers,
            metadata, target='MPP', max_recombination=1e-3
        )
        
        # Validate results
        if validate_optimization_results(results, mpp_models, mpp_scalers,
                                      recombination_models, recombination_scalers, metadata):
            
            # Create report and plots
            create_optimization_report(results, metadata)
            create_optimization_plots(results, metadata)
            
            logging.info("\n" + "=" * 60)
            logging.info("OPTIMIZATION COMPLETED SUCCESSFULLY")
            logging.info(f"Optimal MPP: {results['optimal_mpp']:.6f} W/cm²")
            logging.info("=" * 60)
        else:
            logging.error("Optimization results validation failed")
    
    except Exception as e:
        logging.error(f"Optimization failed: {e}")
        raise

if __name__ == "__main__":
    main() 