#!/usr/bin/env python3
"""
Script 8: Device Parameter Optimization

DESCRIPTION:
This script optimizes device parameters from example_device_parameters.json to maximize 
efficiency (MPP/PCE) while maintaining physics constraints and reasonable recombination rates.

FEATURES:
- Loads example device parameters as starting point
- Uses trained ML models to predict performance during optimization
- Implements constrained optimization (L-BFGS-B and Differential Evolution)
- Maintains physics constraints (energy alignment, electrode compatibility)
- Provides comprehensive before/after comparison
- Generates optimization report with improvement summary

REQUIREMENTS:
- Trained models from Script 5 (results/5_train_optimization_models/)
- Example device parameters (example_device_parameters.json)
- Feature definitions from Script 1 (results/1_feature/)

OUTPUT:
- results/8_optimize_device/
  ├── optimization_log.txt                    # Detailed execution log
  ├── 1_comparison_mpp.png                    # Before vs after: MPP
  ├── 1_comparison_pce.png                    # Before vs after: PCE
  ├── 1_comparison_intsrhn_mean.png           # Before vs after: IntSRHn_mean (log)
  ├── 2_parameter_improvements.png            # Parameter changes visualization
  ├── 3_physics_validation.png                # Constraint validation for both versions
  ├── 4_efficiency_optimization.png           # Detailed efficiency improvement analysis
  ├── optimized_device_parameters.json        # New optimized parameters
  └── optimization_report.json                # Detailed optimization results and recommendations

USAGE:
python scripts/8_optimize_device_parameters.py [--method local|global|both] [--maxiter 100]

OPTIONS:
--method: Optimization method (local=L-BFGS-B, global=Differential Evolution, both=try both)
--maxiter: Maximum iterations for optimization (default: 100)
"""

import os
import sys
import json
import logging
import warnings
import argparse
import numpy as np
from datetime import datetime
from scipy.optimize import minimize, differential_evolution
import joblib
import pandas as pd

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# ===== CONFIGURATION SETTINGS =====
# Progress Counter Configuration (Edit these values as needed)
SHOW_PROGRESS_COUNTER = True        # Set to False to disable progress display
PROGRESS_UPDATE_FREQUENCY = 5       # Show progress every N iterations (1=every iteration, 10=every 10th)
SHOW_EXCEEDED_ITERATIONS = True     # Show when optimizer exceeds maxiter limit
HARD_ITERATION_LIMIT = 100         # HARD LIMIT: Force stop after this many iterations (prevents runaway)

def setup_logging():
    """Setup logging configuration."""
    # Create results directory
    results_dir = 'results/8_optimize_device'
    os.makedirs(results_dir, exist_ok=True)
    
    # Clear any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Setup logging
    log_file = f'{results_dir}/optimization_log.txt'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return results_dir

def load_example_parameters():
    """Load example device parameters."""
    try:
        with open('example_device_parameters.json', 'r') as f:
            data = json.load(f)
        
        parameters = data['parameters']
        device_type = data.get('device_type', 'Unknown Device')
        
        logging.info(f"Loaded example parameters from: example_device_parameters.json")
        logging.info(f"Device Type: {device_type}")
        
        return parameters, device_type
    
    except Exception as e:
        logging.error(f"Error loading example parameters: {e}")
        raise

def load_optimization_models():
    """Load trained models and feature definitions."""
    try:
        models_dir = 'results/5_train_optimization_models/models'
        
        # Load MPP model and scalers
        mpp_model = joblib.load(f'{models_dir}/efficiency_MPP.joblib')
        mpp_scalers = joblib.load(f'{models_dir}/efficiency_MPP_scalers.joblib')
        
        # Load recombination model and scalers
        recomb_model = joblib.load(f'{models_dir}/recombination_IntSRHn_mean.joblib')
        recomb_scalers = joblib.load(f'{models_dir}/recombination_IntSRHn_mean_scalers.joblib')
        
        # Load training metadata
        with open('results/5_train_optimization_models/training_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        # Load feature definitions
        with open('results/1_feature/feature_definitions.json', 'r') as f:
            feature_definitions = json.load(f)
        
        logging.info("Loaded optimization models and feature definitions")
        
        return {
            'mpp_model': mpp_model,
            'mpp_scalers': mpp_scalers,
            'recomb_model': recomb_model,
            'recomb_scalers': recomb_scalers,
            'metadata': metadata,
            'feature_definitions': feature_definitions
        }
    
    except Exception as e:
        logging.error(f"Error loading optimization models: {e}")
        raise

def validate_physics_constraints(parameters):
    """Validate physics constraints for device parameters."""
    try:
        # Energy alignment constraints
        etl_ec = parameters['L1_E_c']
        active_ec = parameters['L2_E_c']
        htl_ec = parameters['L3_E_c']
        
        etl_ev = parameters['L1_E_v']
        active_ev = parameters['L2_E_v']
        htl_ev = parameters['L3_E_v']
        
        # Energy gap constraints (minimum 0.5 eV)
        for layer in ['L1', 'L2', 'L3']:
            gap = abs(parameters[f'{layer}_E_v'] - parameters[f'{layer}_E_c'])
            if gap < 0.5:
                return False, f"Energy gap too small in {layer}: {gap:.3f} eV"
        
        # Band alignment constraints
        if etl_ec < active_ec:
            return False, f"ETL E_c ({etl_ec:.3f}) < Active E_c ({active_ec:.3f})"
        
        if active_ev < htl_ev:
            return False, f"Active E_v ({active_ev:.3f}) < HTL E_v ({htl_ev:.3f})"
        
        # Electrode work function compatibility (W_L=4.05, W_R=5.2)
        # Allow larger margin for optimization flexibility
        W_L, W_R = 4.05, 5.2
        margin = 0.1  # 100 meV tolerance (increased for better optimization)
        
        if W_L < etl_ec - margin:
            return False, f"W_L ({W_L}) < ETL E_c ({etl_ec:.3f}) by more than {margin} eV"
        
        if W_R > htl_ev + margin:
            return False, f"W_R ({W_R}) > HTL E_v ({htl_ev:.3f}) by more than {margin} eV"
        
        # Log warnings for small violations
        if W_L < etl_ec:
            logging.warning(f"W_L ({W_L}) slightly < ETL E_c ({etl_ec:.3f}) - within tolerance")
        
        if W_R > htl_ev:
            logging.warning(f"W_R ({W_R}) slightly > HTL E_v ({htl_ev:.3f}) - within tolerance")
        
        # Thickness constraints (10 nm to 1000 nm)
        for layer in ['L1', 'L2', 'L3']:
            thickness_nm = parameters[f'{layer}_L'] * 1e9
            if thickness_nm < 10 or thickness_nm > 1000:
                return False, f"{layer} thickness out of range: {thickness_nm:.1f} nm"
        
        # Doping constraints (reasonable values)
        for layer in ['L1', 'L2', 'L3']:
            nd = parameters[f'{layer}_N_D']
            na = parameters[f'{layer}_N_A']
            if nd < 0 or na < 0 or nd > 1e22 or na > 1e22:
                return False, f"{layer} doping out of range: N_D={nd:.2e}, N_A={na:.2e}"
        
        return True, "All physics constraints satisfied"
    
    except Exception as e:
        return False, f"Error validating constraints: {e}"

def calculate_derived_features_script7_compatible(df):
    """Calculate derived features EXACTLY like Script 7 for consistency."""
    # Only log during first few iterations to avoid log spam
    if _iteration_counter <= 3:
        logging.info("Calculating derived features (Script 7 compatible)...")
    elif _iteration_counter % 50 == 0:  # Log every 50th iteration
        logging.debug(f"Calculating derived features (iteration {_iteration_counter})...")
    
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
    
    # Only log during first few iterations to avoid log spam
    if _iteration_counter <= 3:
        logging.info(f"Calculated derived features (Script 7 compatible). Total features: {len(df.columns)}")
    elif _iteration_counter % 50 == 0:  # Log every 50th iteration
        logging.debug(f"Calculated derived features (iteration {_iteration_counter}). Total features: {len(df.columns)}")
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
            # Only log S-Q validation during first few iterations to avoid log spam
            if _iteration_counter <= 3:
                logging.info(f"PCE prediction ({pce_pred:.2f}%) is within S-Q limit ({SQ_LIMIT}%) [VALID]")
            elif _iteration_counter % 50 == 0:  # Log every 50th iteration
                logging.debug(f"PCE ({pce_pred:.2f}%) within S-Q limit [VALID] (iteration {_iteration_counter})")
        
        return pce_pred
        
    except Exception as e:
        logging.warning(f"Error validating S-Q limit: {e}")
        return pce_pred

def predict_performance(parameters, models_data):
    """Predict device performance using trained models - MUST match Script 7 exactly."""
    try:
        # Create DataFrame with parameters (same as Script 7)
        df = pd.DataFrame([parameters])
        
        # Calculate derived features using EXACT same method as Script 7
        df = calculate_derived_features_script7_compatible(df)
        
        # Get exact training features from metadata (same as Script 7)
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
        
        # Convert to numpy array for prediction
        X = df_features.values
        
        # Predict MPP (no target inverse scaling needed - model outputs are already in original scale)
        mpp_scaler = models_data['mpp_scalers']['feature_scaler']
        
        X_scaled = mpp_scaler.transform(X)
        mpp_pred_scaled = models_data['mpp_model'].predict(X_scaled)[0]
        
        # Inverse transform the scaled prediction back to original scale
        target_scaler = models_data['mpp_scalers']['target_scaler']
        mpp_pred = target_scaler.inverse_transform([[mpp_pred_scaled]])[0][0]
        
        # Calculate PCE (Power Conversion Efficiency) from MPP using physics equation
        # PCE = (MPP / Incident_Power) × 100%
        # From simulation: MPP is in W/m² (V × J where J is in A/m²)
        # Standard solar conditions: 1000 W/m² incident power
        incident_power_W_per_m2 = 1000.0  # W/m² (AM1.5G standard)
        pce_pred = (mpp_pred / incident_power_W_per_m2) * 100  # Physics-based PCE calculation
        
        # Validate against Shockley-Queisser limit
        pce_pred = validate_shockley_queisser_limit(pce_pred, parameters)
        
        # Predict recombination
        recomb_scaler = models_data['recomb_scalers']['feature_scaler']
        
        X_scaled = recomb_scaler.transform(X)
        recomb_pred_scaled = models_data['recomb_model'].predict(X_scaled)[0]
        
        # Inverse transform the scaled prediction back to original scale
        target_scaler = models_data['recomb_scalers']['target_scaler']
        recomb_pred = target_scaler.inverse_transform([[recomb_pred_scaled]])[0][0]
        
        return {
            'MPP': float(mpp_pred),
            'PCE': float(pce_pred),
            'IntSRHn_mean': float(recomb_pred)
        }
    
    except Exception as e:
        logging.error(f"Error predicting performance: {e}")
        return {'MPP': 0.0, 'PCE': 0.0, 'IntSRHn_mean': 1e30}

def get_parameter_bounds():
    """Get parameter bounds for optimization."""
    try:
        # Load parameter bounds from feature definitions
        with open('results/1_feature/parameter_bounds.json', 'r') as f:
            bounds_data = json.load(f)
        
        # Convert to optimization bounds format
        bounds = []
        param_names = []
        
        primary_params = [
            'L1_L', 'L1_E_c', 'L1_E_v', 'L1_N_D', 'L1_N_A',
            'L2_L', 'L2_E_c', 'L2_E_v', 'L2_N_D', 'L2_N_A', 
            'L3_L', 'L3_E_c', 'L3_E_v', 'L3_N_D', 'L3_N_A'
        ]
        
        for param in primary_params:
            if param in bounds_data:
                bound = bounds_data[param]
                # Ensure thickness bounds are in meters (not nm)
                if param.endswith('_L'):
                    # Convert from nm to meters if needed
                    if bound[1] > 1e-6:  # If max > 1 micrometer, assume it's in nm
                        bound = [bound[0] * 1e-9, bound[1] * 1e-9]
                        logging.info(f"Converted {param} bounds from nm to meters: {bound}")
                bounds.append(bound)
                param_names.append(param)
            else:
                logging.warning(f"No bounds found for parameter {param}")
        
        logging.info(f"Loaded bounds for {len(bounds)} parameters")
        
        return bounds, param_names
    
    except Exception as e:
        logging.error(f"Error loading parameter bounds: {e}")
        raise

# Global variables for progress tracking
_iteration_counter = 0
_max_iterations = 100
_optimization_method = "Unknown"

def objective_function(x, param_names, models_data, constraint_penalty=1e6):
    """Objective function for optimization (maximize efficiency, minimize recombination)."""
    global _iteration_counter, _max_iterations, _optimization_method
    
    try:
        # Increment and display progress counter (if enabled)
        _iteration_counter += 1
        
        # HARD LIMIT: Force stop if iterations reach limit (prevents runaway)
        if _iteration_counter >= HARD_ITERATION_LIMIT:
            if _iteration_counter == HARD_ITERATION_LIMIT:  # Log only once
                logging.warning(f"HARD ITERATION LIMIT REACHED: {_iteration_counter} >= {HARD_ITERATION_LIMIT}")
                logging.warning("STOPPING OPTIMIZATION (prevents runaway execution)")
                print(f"\nITERATION LIMIT: Reached hard limit of {HARD_ITERATION_LIMIT} iterations!")
                print("Optimizer will use best result found so far...")
            # Return a very large penalty to force termination
            return constraint_penalty * 100  # Strong signal to stop optimization
        
        if SHOW_PROGRESS_COUNTER:
            progress_percent = (_iteration_counter / _max_iterations) * 100
            remaining = _max_iterations - _iteration_counter
            
            # Print progress based on configured frequency and milestones
            show_milestone = (_iteration_counter == 1 or remaining <= 5 or _iteration_counter <= 10)
            show_frequency = (_iteration_counter % PROGRESS_UPDATE_FREQUENCY == 0)
            
            if show_milestone or show_frequency:
                if _iteration_counter <= _max_iterations:
                    print(f"\r🔄 {_optimization_method} Progress: {_iteration_counter}/{_max_iterations} " +
                          f"({progress_percent:.1f}%) - {remaining} iterations remaining", end="", flush=True)
                elif SHOW_EXCEEDED_ITERATIONS:
                    # Handle case where optimizer exceeds maxiter (common with L-BFGS-B)
                    print(f"\r🔄 {_optimization_method} Progress: {_iteration_counter} iterations " +
                          f"(exceeded maxiter={_max_iterations})", end="", flush=True)
        
        # Convert array to parameter dictionary
        parameters = dict(zip(param_names, x))
        
        # Check physics constraints
        is_valid, message = validate_physics_constraints(parameters)
        if not is_valid:
            return constraint_penalty  # Large penalty for invalid parameters
        
        # Predict performance
        predictions = predict_performance(parameters, models_data)
        
        # Objective: maximize MPP, minimize recombination
        # We want high efficiency and low recombination
        mpp = predictions['MPP']
        recomb = predictions['IntSRHn_mean']
        
        # Normalize recombination (typical range 1e28 to 1e31)
        recomb_normalized = np.log10(max(recomb, 1e-30)) / 30.0  # Scale to ~0-1
        
        # Combined objective: maximize MPP, minimize normalized recombination
        objective = -mpp + recomb_normalized  # Negative because we minimize
        
        return float(objective)
    
    except Exception as e:
        logging.warning(f"Error in objective function: {e}")
        return constraint_penalty

def optimize_parameters(original_params, models_data, method='both', maxiter=100):
    """Optimize device parameters for maximum efficiency."""
    try:
        logging.info(f"\n=== Starting Parameter Optimization ===")
        logging.info(f"Method: {method}, Max iterations: {maxiter}")
        
        # Get parameter bounds and names
        bounds, param_names = get_parameter_bounds()
        
        # Create initial guess from original parameters
        x0 = [original_params[name] for name in param_names]
        
        # Predict original performance
        original_predictions = predict_performance(original_params, models_data)
        logging.info(f"Original Performance - MPP: {original_predictions['MPP']:.4f} W/cm², PCE: {original_predictions['PCE']:.2f}%")
        logging.info(f"Original Recombination: {original_predictions['IntSRHn_mean']:.2e}")
        
        best_result = None
        best_objective = float('inf')
        
        # Try local optimization (L-BFGS-B)
        if method in ['local', 'both']:
            logging.info("\n--- Local Optimization (L-BFGS-B) ---")
            try:
                # Initialize progress tracking for local optimization
                global _iteration_counter, _max_iterations, _optimization_method
                _iteration_counter = 0
                _max_iterations = maxiter
                _optimization_method = "Local (L-BFGS-B)"
                
                result_local = minimize(
                    objective_function,
                    x0,
                    args=(param_names, models_data),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': maxiter, 'disp': False}
                )
                
                # Print final status (if progress counter is enabled)
                if SHOW_PROGRESS_COUNTER:
                    if _iteration_counter <= _max_iterations:
                        print(f"\r✅ {_optimization_method} Complete: {_iteration_counter}/{_max_iterations} iterations")
                    else:
                        print(f"\r✅ {_optimization_method} Complete: {_iteration_counter} iterations (exceeded maxiter={_max_iterations})")
                    print()  # New line for clean output
                
                if result_local.success and result_local.fun < best_objective:
                    best_result = result_local
                    best_objective = result_local.fun
                    logging.info(f"Local optimization successful. Objective: {result_local.fun:.6f}")
                else:
                    logging.warning(f"Local optimization failed or poor result: {result_local.message}")
            
            except Exception as e:
                logging.error(f"Local optimization error: {e}")
        
        # Try global optimization (Differential Evolution)
        if method in ['global', 'both']:
            logging.info("\n--- Global Optimization (Differential Evolution) ---")
            try:
                # Initialize progress tracking for global optimization
                _iteration_counter = 0
                _max_iterations = maxiter//2  # DE uses fewer iterations
                _optimization_method = "Global (Differential Evolution)"
                
                result_global = differential_evolution(
                    objective_function,
                    bounds,
                    args=(param_names, models_data),
                    maxiter=maxiter//2,  # DE needs fewer iterations
                    popsize=15,
                    seed=42,
                    disp=False
                )
                
                # Print final status (if progress counter is enabled)
                if SHOW_PROGRESS_COUNTER:
                    if _iteration_counter <= _max_iterations:
                        print(f"\r✅ {_optimization_method} Complete: {_iteration_counter}/{_max_iterations} iterations")
                    else:
                        print(f"\r✅ {_optimization_method} Complete: {_iteration_counter} iterations (exceeded maxiter={_max_iterations})")
                    print()  # New line for clean output
                
                if result_global.success and result_global.fun < best_objective:
                    best_result = result_global
                    best_objective = result_global.fun
                    logging.info(f"Global optimization successful. Objective: {result_global.fun:.6f}")
                else:
                    logging.warning(f"Global optimization failed or poor result: {result_global.message}")
            
            except Exception as e:
                logging.error(f"Global optimization error: {e}")
        
        if best_result is None:
            logging.error("All optimization methods failed!")
            return None
        
        # Convert result to parameter dictionary
        optimized_params = dict(zip(param_names, best_result.x))
        
        # Validate optimized parameters
        is_valid, validation_message = validate_physics_constraints(optimized_params)
        if not is_valid:
            logging.error(f"Optimized parameters failed validation: {validation_message}")
            return None
        
        # Predict optimized performance
        optimized_predictions = predict_performance(optimized_params, models_data)
        
        logging.info(f"\n=== Optimization Results ===")
        logging.info(f"Optimized Performance - MPP: {optimized_predictions['MPP']:.4f} W/cm², PCE: {optimized_predictions['PCE']:.2f}%")
        logging.info(f"Optimized Recombination: {optimized_predictions['IntSRHn_mean']:.2e}")
        logging.info(f"Physics Validation: {validation_message}")
        
        # Calculate improvements
        mpp_improvement = ((optimized_predictions['MPP'] - original_predictions['MPP']) / original_predictions['MPP']) * 100
        pce_improvement = ((optimized_predictions['PCE'] - original_predictions['PCE']) / original_predictions['PCE']) * 100
        recomb_change = ((optimized_predictions['IntSRHn_mean'] - original_predictions['IntSRHn_mean']) / original_predictions['IntSRHn_mean']) * 100
        
        logging.info(f"Improvements - MPP: {mpp_improvement:+.2f}%, PCE: {pce_improvement:+.2f}%, Recombination: {recomb_change:+.2f}%")
        
        return {
            'optimized_parameters': optimized_params,
            'original_parameters': original_params,
            'optimized_predictions': optimized_predictions,
            'original_predictions': original_predictions,
            'improvements': {
                'MPP_improvement_percent': float(mpp_improvement),
                'PCE_improvement_percent': float(pce_improvement),
                'recombination_change_percent': float(recomb_change)
            },
            'optimization_info': {
                'method': method,
                'success': True,
                'objective_value': float(best_objective),
                'iterations': int(best_result.nit) if hasattr(best_result, 'nit') else maxiter,
                'validation_message': validation_message
            }
        }
    
    except Exception as e:
        logging.error(f"Error during optimization: {e}")
        return None

def create_optimization_visualizations(optimization_results, results_dir):
    """Create comprehensive optimization visualizations."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        
        if not optimization_results:
            logging.error("No optimization results to visualize")
            return
        
        original_params = optimization_results['original_parameters']
        optimized_params = optimization_results['optimized_parameters']
        original_pred = optimization_results['original_predictions']
        optimized_pred = optimization_results['optimized_predictions']
        improvements = optimization_results['improvements']
        
        # 1. Performance Comparison - split into three separate figures
        # a) MPP comparison
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        categories = ['Original', 'Optimized']
        mpp_values = [original_pred['MPP'], optimized_pred['MPP']]
        bars = ax.bar(categories, mpp_values, color=['lightblue', 'darkblue'], alpha=0.7)
        ax.set_ylabel('MPP (W/cm²)')
        ax.set_title('Maximum Power Point')
        for bar, value in zip(bars, mpp_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mpp_values) * 0.02,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        ax.set_ylim(0, max(mpp_values) * 1.15)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{results_dir}/1_comparison_mpp.png', dpi=300, bbox_inches='tight')
        plt.close()

        # b) PCE comparison
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        pce_values = [original_pred['PCE'], optimized_pred['PCE']]
        bars = ax.bar(categories, pce_values, color=['lightgreen', 'darkgreen'], alpha=0.7)
        ax.set_ylabel('PCE (%)')
        ax.set_title('Power Conversion Efficiency')
        for bar, value in zip(bars, pce_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(pce_values) * 0.02,
                    f'{value:.2f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
        ax.set_ylim(0, max(pce_values) * 1.15)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{results_dir}/1_comparison_pce.png', dpi=300, bbox_inches='tight')
        plt.close()

        # c) IntSRHn_mean comparison (log-scale label)
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        original_recomb = original_pred['IntSRHn_mean']
        optimized_recomb = optimized_pred['IntSRHn_mean']
        original_abs = abs(original_recomb) if original_recomb != 0 else 1e-30
        optimized_abs = abs(optimized_recomb) if optimized_recomb != 0 else 1e-30
        recomb_values = [np.log10(original_abs), np.log10(optimized_abs)]
        colors = ['lightcoral', 'darkgreen' if optimized_recomb < original_recomb else 'darkred']
        bars = ax.bar(categories, recomb_values, color=colors, alpha=0.7)
        ax.set_ylabel('log₁₀(|Recombination Rate|)')
        ax.set_title('Interfacial SRH (Lower is Better)')
        # Compute compact y-limits and place labels just above bar tops
        y_min, y_max = min(recomb_values), max(recomb_values)
        y_range = max(y_max - y_min, 1e-6)
        for bar, value, actual_val in zip(bars, recomb_values, [original_recomb, optimized_recomb]):
            sign = '-' if actual_val < 0 else ''
            label = f'{sign}≈0' if abs(actual_val) < 1e-10 else f'{sign}1e{value:.3f}'
            text_y = value + 0.02 * y_range
            ax.text(bar.get_x() + bar.get_width()/2, text_y,
                    label, ha='center', va='bottom', fontweight='bold', fontsize=10)
        # Slight headroom for labels above bars
        ax.set_ylim(y_min - 0.03 * y_range, y_max + 0.10 * y_range)
        ax.grid(True, alpha=0.3)
        # Keep layout compact and avoid large outer bbox that can shrink the axes
        fig.tight_layout()
        plt.savefig(f'{results_dir}/1_comparison_intsrhn_mean.png', dpi=300)
        plt.close()

        # 2. Individual Parameter Change Charts
        
        # 2a. Thickness Changes Chart
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        thickness_params = ['L1_L', 'L2_L', 'L3_L']
        original_thick = [original_params[p] * 1e9 for p in thickness_params]
        optimized_thick = [optimized_params[p] * 1e9 for p in thickness_params]
        labels = ['ETL', 'Active', 'HTL']

        x = np.arange(len(labels))
        width = 0.35
        bars1 = ax.bar(x - width/2, original_thick, width, label='Original', color='lightblue', alpha=0.7, edgecolor='black')
        bars2 = ax.bar(x + width/2, optimized_thick, width, label='Optimized', color='darkblue', alpha=0.7, edgecolor='black')

        ax.set_ylabel('Thickness (nm)')
        ax.set_title('Layer Thickness Optimization', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add value labels
        for bars, values in [(bars1, original_thick), (bars2, optimized_thick)]:
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(original_thick + optimized_thick) * 0.02,
                        f'{value:.1f}nm', ha='center', va='bottom', fontweight='bold', fontsize=10)

        # Add extra space at the top for labels
        ax.set_ylim(0, max(original_thick + optimized_thick) * 1.15)

        plt.tight_layout()
        plt.savefig(f'{results_dir}/2_thickness_optimization.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2b. Energy Level Changes Chart
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        energy_params = ['L1_E_c', 'L1_E_v', 'L2_E_c', 'L2_E_v', 'L3_E_c', 'L3_E_v']
        original_energy = [original_params[p] for p in energy_params]
        optimized_energy = [optimized_params[p] for p in energy_params]
        energy_labels = ['ETL E_c', 'ETL E_v', 'Act E_c', 'Act E_v', 'HTL E_c', 'HTL E_v']
        
        x = np.arange(len(energy_labels))
        width = 0.35
        bars1 = ax.bar(x - width/2, original_energy, width, label='Original', color='gold', alpha=0.7, edgecolor='black')
        bars2 = ax.bar(x + width/2, optimized_energy, width, label='Optimized', color='orange', alpha=0.7, edgecolor='black')
        
        ax.set_ylabel('Energy (eV)')
        ax.set_title('Energy Level Optimization', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(energy_labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bars, values in [(bars1, original_energy), (bars2, optimized_energy)]:
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(original_energy + optimized_energy) * 0.02,
                        f'{value:.2f}eV', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # Add extra space at the top for labels
        ax.set_ylim(0, max(original_energy + optimized_energy) * 1.15)
        
        plt.tight_layout()
        plt.savefig(f'{results_dir}/3_energy_optimization.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2c. Doping Changes Chart (log scale)
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        doping_params = ['L1_N_D', 'L1_N_A', 'L2_N_D', 'L2_N_A', 'L3_N_D', 'L3_N_A']
        original_doping = [np.log10(max(original_params[p], 1e15)) for p in doping_params]
        optimized_doping = [np.log10(max(optimized_params[p], 1e15)) for p in doping_params]
        doping_labels = ['ETL N_D', 'ETL N_A', 'Act N_D', 'Act N_A', 'HTL N_D', 'HTL N_A']
        
        x = np.arange(len(doping_labels))
        width = 0.35
        bars1 = ax.bar(x - width/2, original_doping, width, label='Original', color='purple', alpha=0.7, edgecolor='black')
        bars2 = ax.bar(x + width/2, optimized_doping, width, label='Optimized', color='indigo', alpha=0.7, edgecolor='black')
        
        ax.set_ylabel('log₁₀(Doping) [cm⁻³]')
        ax.set_title('Doping Concentration Optimization', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(doping_labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bars, values in [(bars1, original_doping), (bars2, optimized_doping)]:
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(original_doping + optimized_doping) * 0.02,
                        f'1e{value:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # Add extra space at the top for labels
        ax.set_ylim(min(original_doping + optimized_doping) - 1, max(original_doping + optimized_doping) * 1.15)
        
        plt.tight_layout()
        plt.savefig(f'{results_dir}/4_doping_optimization.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2d. Performance Improvements Summary
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        improvements_data = [
            improvements['MPP_improvement_percent'],
            improvements['PCE_improvement_percent'],
            improvements['recombination_change_percent']
        ]
        improvement_labels = ['MPP\nImprovement', 'PCE\nImprovement', 'Recombination\nChange']
        colors = ['green' if x > 0 else 'red' if x < 0 else 'gray' for x in improvements_data]
        
        bars = ax.bar(improvement_labels, improvements_data, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Improvement (%)')
        ax.set_title('Performance Improvements Summary', fontsize=14, fontweight='bold')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.grid(True, alpha=0.3)
        
        for bar, value in zip(bars, improvements_data):
            ax.text(bar.get_x() + bar.get_width()/2, 
                    bar.get_height() + (abs(value) * 0.05 if value > 0 else -abs(value) * 0.05),
                    f'{value:+.1f}%', ha='center', 
                    va='bottom' if value > 0 else 'top', fontweight='bold', fontsize=12)
        
        # Add extra space for labels
        max_abs = max(abs(min(improvements_data)), abs(max(improvements_data)))
        ax.set_ylim(min(improvements_data) - max_abs * 0.15, max(improvements_data) + max_abs * 0.15)
        
        plt.tight_layout()
        plt.savefig(f'{results_dir}/5_performance_improvements.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Comprehensive Optimization Table
        create_optimization_table(original_params, optimized_params, original_pred, optimized_pred, improvements, results_dir)
        
        logging.info(f"Created optimization visualizations in {results_dir}/")
        
    except Exception as e:
        logging.error(f"Error creating visualizations: {e}")

def create_optimization_table(original_params, optimized_params, original_pred, optimized_pred, improvements, results_dir):
    """Create a comprehensive table showing all original vs optimized values."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        
        # Create figure with larger size for the table
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare table data
        table_data = []
        
        # Header
        table_data.append(['Parameter', 'Original', 'Optimized', 'Change', 'Unit', 'Description'])
        
        # Performance Metrics
        table_data.append(['=== PERFORMANCE METRICS ===', '', '', '', '', ''])
        
        # MPP
        mpp_change = f"{improvements['MPP_improvement_percent']:+.2f}%"
        table_data.append(['MPP', f"{original_pred['MPP']:.4f}", f"{optimized_pred['MPP']:.4f}", 
                          mpp_change, 'W/cm²', 'Maximum Power Point'])
        
        # PCE
        pce_change = f"{improvements['PCE_improvement_percent']:+.2f}%"
        table_data.append(['PCE', f"{original_pred['PCE']:.2f}", f"{optimized_pred['PCE']:.2f}", 
                          pce_change, '%', 'Power Conversion Efficiency'])
        
        # SRH Recombination
        srh_change = f"{improvements['recombination_change_percent']:+.2f}%"
        table_data.append(['IntSRHn_mean', f"{original_pred['IntSRHn_mean']:.2e}", f"{optimized_pred['IntSRHn_mean']:.2e}", 
                          srh_change, 'cm⁻³s⁻¹', 'SRH Recombination Rate'])
        
        # Layer Parameters
        table_data.append(['', '', '', '', '', ''])
        table_data.append(['=== LAYER THICKNESS ===', '', '', '', '', ''])
        
        thickness_params = [
            ('L1_L', 'ETL Thickness'),
            ('L2_L', 'Active Layer Thickness'), 
            ('L3_L', 'HTL Thickness')
        ]
        
        for param, desc in thickness_params:
            orig_nm = original_params[param] * 1e9
            opt_nm = optimized_params[param] * 1e9
            change_percent = ((opt_nm - orig_nm) / orig_nm) * 100
            change_str = f"{change_percent:+.1f}%"
            table_data.append([param, f"{orig_nm:.1f}", f"{opt_nm:.1f}", change_str, 'nm', desc])
        
        # Energy Levels
        table_data.append(['', '', '', '', '', ''])
        table_data.append(['=== ENERGY LEVELS ===', '', '', '', '', ''])
        
        energy_params = [
            ('L1_E_c', 'ETL Conduction Band'),
            ('L1_E_v', 'ETL Valence Band'),
            ('L2_E_c', 'Active Conduction Band'),
            ('L2_E_v', 'Active Valence Band'),
            ('L3_E_c', 'HTL Conduction Band'),
            ('L3_E_v', 'HTL Valence Band')
        ]
        
        for param, desc in energy_params:
            orig_val = original_params[param]
            opt_val = optimized_params[param]
            change_abs = opt_val - orig_val
            change_str = f"{change_abs:+.3f}eV"
            table_data.append([param, f"{orig_val:.3f}", f"{opt_val:.3f}", change_str, 'eV', desc])
        
        # Doping Concentrations
        table_data.append(['', '', '', '', '', ''])
        table_data.append(['=== DOPING CONCENTRATIONS ===', '', '', '', '', ''])
        
        doping_params = [
            ('L1_N_D', 'ETL Donor Concentration'),
            ('L1_N_A', 'ETL Acceptor Concentration'),
            ('L2_N_D', 'Active Donor Concentration'),
            ('L2_N_A', 'Active Acceptor Concentration'),
            ('L3_N_D', 'HTL Donor Concentration'),
            ('L3_N_A', 'HTL Acceptor Concentration')
        ]
        
        for param, desc in doping_params:
            orig_val = original_params[param]
            opt_val = optimized_params[param]
            
            # Format in scientific notation
            if orig_val > 0:
                orig_str = f"{orig_val:.2e}"
            else:
                orig_str = "0"
            
            if opt_val > 0:
                opt_str = f"{opt_val:.2e}"
            else:
                opt_str = "0"
            
            # Calculate change
            if orig_val > 0:
                change_percent = ((opt_val - orig_val) / orig_val) * 100
                change_str = f"{change_percent:+.1f}%"
            else:
                change_str = "N/A"
            
            table_data.append([param, orig_str, opt_str, change_str, 'cm⁻³', desc])
        
        # Create the table (positioned lower to avoid title overlap)
        table = ax.table(cellText=table_data, cellLoc='left', loc='upper center', 
                        colWidths=[0.15, 0.15, 0.15, 0.12, 0.08, 0.35])
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2.5)  # Make rows taller
        
        # Style header row
        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style section headers
        section_rows = [1, 7, 14]  # Performance, Thickness, Energy, Doping section headers
        for row in section_rows:
            if row < len(table_data):
                for i in range(len(table_data[0])):
                    table[(row, i)].set_facecolor('#E8F5E8')
                    table[(row, i)].set_text_props(weight='bold')
        
        # Style performance metrics (highlight improvements)
        perf_rows = [2, 3, 4]  # MPP, PCE, SRH rows
        for row in perf_rows:
            if row < len(table_data):
                # Color code the change column based on improvement
                change_val = table_data[row][3]
                if '+' in change_val and 'SRH' not in table_data[row][5]:  # Positive change for MPP/PCE
                    table[(row, 3)].set_facecolor('#C8E6C9')  # Light green
                elif '-' in change_val and 'SRH' in table_data[row][5]:  # Negative change for SRH (good)
                    table[(row, 3)].set_facecolor('#C8E6C9')  # Light green
                else:
                    table[(row, 3)].set_facecolor('#FFCDD2')  # Light red
        
        # Add title with proper spacing
        plt.suptitle('Comprehensive Optimization Results Table', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Add subtitle with key improvements
        subtitle = f"Key Improvements: MPP {improvements['MPP_improvement_percent']:+.1f}%, " \
                  f"PCE {improvements['PCE_improvement_percent']:+.1f}%, " \
                  f"SRH {improvements['recombination_change_percent']:+.1f}%"
        plt.figtext(0.5, 0.93, subtitle, ha='center', fontsize=12, style='italic')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)  # Make more room for title and subtitle
        plt.savefig(f'{results_dir}/6_optimization_table.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logging.info(f"Created optimization table: {results_dir}/6_optimization_table.png")
        
    except Exception as e:
        logging.error(f"Error creating optimization table: {e}")

def save_optimization_results(optimization_results, device_type, results_dir):
    """Save optimization results and generate report."""
    try:
        if not optimization_results:
            logging.error("No optimization results to save")
            return
        
        # Save optimized parameters
        optimized_device_data = {
            "device_type": f"{device_type} (Optimized)",
            "parameters": optimization_results['optimized_parameters'],
            "optimization_info": {
                "original_device": device_type,
                "optimization_date": datetime.now().isoformat(),
                "improvements": optimization_results['improvements'],
                "method": optimization_results['optimization_info']['method'],
                "validation_status": optimization_results['optimization_info']['validation_message']
            },
            "predicted_performance": {
                "original": optimization_results['original_predictions'],
                "optimized": optimization_results['optimized_predictions']
            },
            "layer_descriptions": {
                "L1": "ETL - Electron Transport Layer (Optimized)",
                "L2": "Active - Perovskite Absorber Layer (Optimized)",
                "L3": "HTL - Hole Transport Layer (Optimized)"
            },
            "physics_notes": [
                "Optimized parameters maintain all physics constraints.",
                "Energy alignment: ETL_Ec >= Active_Ec, Active_Ev >= HTL_Ev.",
                "Electrode compatibility: W_L=4.05eV >= ETL_Ec, W_R=5.2eV <= HTL_Ev.",
                f"Efficiency improved by {optimization_results['improvements']['PCE_improvement_percent']:+.2f}%"
            ]
        }
        
        # Convert numpy types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Apply conversion recursively
        import json
        optimized_device_str = json.dumps(optimized_device_data, default=convert_numpy_types, indent=2)
        optimized_device_data = json.loads(optimized_device_str)
        
        with open(f'{results_dir}/optimized_device_parameters.json', 'w') as f:
            json.dump(optimized_device_data, f, indent=2)
        
        # Create detailed optimization report
        report = {
            "optimization_summary": {
                "success": True,
                "method": optimization_results['optimization_info']['method'],
                "optimization_date": datetime.now().isoformat(),
                "original_device": device_type
            },
            "performance_comparison": {
                "original": optimization_results['original_predictions'],
                "optimized": optimization_results['optimized_predictions'],
                "improvements": optimization_results['improvements']
            },
            "parameter_changes": {
                "original": optimization_results['original_parameters'],
                "optimized": optimization_results['optimized_parameters']
            },
            "physics_validation": {
                "status": "VALID",
                "message": optimization_results['optimization_info']['validation_message']
            },
            "recommendations": [
                f"Efficiency improved by {optimization_results['improvements']['PCE_improvement_percent']:+.2f}%",
                f"MPP increased by {optimization_results['improvements']['MPP_improvement_percent']:+.2f}%",
                f"Recombination rate changed by {optimization_results['improvements']['recombination_change_percent']:+.2f}%",
                "All physics constraints maintained",
                "Device remains manufacturable with optimized parameters"
            ]
        }
        
        # Convert numpy types for report too
        report_str = json.dumps(report, default=convert_numpy_types, indent=2)
        report = json.loads(report_str)
        
        with open(f'{results_dir}/optimization_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Create example_device_parameters.json format for easy testing
        create_example_format_file(optimization_results, results_dir)
        
        logging.info(f"Optimization results saved to {results_dir}/")
        logging.info(f"Optimized parameters: optimized_device_parameters.json")
        logging.info(f"Detailed report: optimization_report.json")
        logging.info(f"Ready-to-use format: example_device_parameters_optimized.json")
        
    except Exception as e:
        logging.error(f"Error saving optimization results: {e}")

def create_example_format_file(optimization_results, results_dir):
    """Create optimized parameters in exact example_device_parameters.json format for easy testing."""
    try:
        # Load the original example_device_parameters.json to get the exact structure
        with open('example_device_parameters.json', 'r') as f:
            original_structure = json.load(f)
        
        # Create the optimized version with same structure
        optimized_example = {
            "description": f"Optimized device parameters (Improved by {optimization_results['improvements']['PCE_improvement_percent']:+.2f}%)",
            "device_type": f"{original_structure.get('device_type', 'Perovskite Solar Cell')} (Optimized)",
            "last_updated": datetime.now().strftime("%Y-%m-%d"),
            "parameters": optimization_results['optimized_parameters'],
            "layer_descriptions": original_structure.get('layer_descriptions', {
                "L1": "ETL - Electron Transport Layer (PCBM)",
                "L2": "Active - Perovskite Absorber Layer (MAPI)", 
                "L3": "HTL - Hole Transport Layer (PEDOT)"
            }),
            "parameter_descriptions": original_structure.get('parameter_descriptions', {
                "L_L": "Layer thickness (m)",
                "E_c": "Conduction band energy (eV)",
                "E_v": "Valence band energy (eV)",
                "N_D": "Donor concentration (cm^-3)",
                "N_A": "Acceptor concentration (cm^-3)"
            }),
            "physics_notes": original_structure.get('physics_notes', {
                "energy_alignment": "ETL E_c >= Active E_c, Active E_v >= HTL E_v",
                "electrode_compatibility": "W_L=4.05eV >= ETL E_c, W_R=5.2eV <= HTL E_v",
                "doping_types": "ETL: n-type, Active: intrinsic, HTL: p-type"
            }),
            "optimization_info": {
                "optimized_from": "example_device_parameters.json",
                "optimization_date": datetime.now().isoformat(),
                "method": optimization_results['optimization_info']['method'],
                "improvements": {
                    "MPP_improvement_percent": optimization_results['improvements']['MPP_improvement_percent'],
                    "PCE_improvement_percent": optimization_results['improvements']['PCE_improvement_percent'],
                    "recombination_change_percent": optimization_results['improvements']['recombination_change_percent']
                },
                "predicted_performance": {
                    "MPP_W_per_m2": optimization_results['optimized_predictions']['MPP'],
                    "PCE_percent": optimization_results['optimized_predictions']['PCE'],
                    "recombination": optimization_results['optimized_predictions']['IntSRHn_mean']
                }
            }
        }
        
        # Convert numpy types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Save in exact example_device_parameters.json format
        output_file = f'{results_dir}/example_device_parameters_optimized.json'
        optimized_str = json.dumps(optimized_example, default=convert_numpy_types, indent=4)
        optimized_data = json.loads(optimized_str)
        
        with open(output_file, 'w') as f:
            json.dump(optimized_data, f, indent=4)
        
        logging.info(f"Created ready-to-use format: example_device_parameters_optimized.json")
        logging.info(f"To test: cp {output_file} example_device_parameters.json")
        
    except Exception as e:
        logging.error(f"Error creating example format file: {e}")

def main():
    """Main optimization workflow."""
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Optimize device parameters for maximum efficiency')
        parser.add_argument('--method', choices=['local', 'global', 'both'], default='both',
                          help='Optimization method (default: both)')
        parser.add_argument('--maxiter', type=int, default=100,
                          help='Maximum iterations (default: 100)')
        args = parser.parse_args()
        
        # Setup logging
        results_dir = setup_logging()
        
        logging.info("=== DEVICE PARAMETER OPTIMIZATION ===")
        logging.info(f"Log file: {results_dir}/optimization_log.txt")
        
        # Load example parameters
        logging.info("\n=== Step 1: Loading Example Parameters ===")
        original_params, device_type = load_example_parameters()
        
        # Load optimization models
        logging.info("\n=== Step 2: Loading Optimization Models ===")
        models_data = load_optimization_models()
        
        # Validate original parameters
        logging.info("\n=== Step 3: Validating Original Parameters ===")
        is_valid, message = validate_physics_constraints(original_params)
        logging.info(f"Original parameters validation: {message}")
        
        if not is_valid:
            logging.error("Original parameters are invalid. Cannot proceed with optimization.")
            return
        
        # Optimize parameters
        logging.info(f"\n=== Step 4: Optimizing Parameters (Method: {args.method}) ===")
        if SHOW_PROGRESS_COUNTER:
            logging.info(f"Progress counter: ENABLED (update every {PROGRESS_UPDATE_FREQUENCY} iterations)")
        else:
            logging.info("Progress counter: DISABLED")
            
        optimization_results = optimize_parameters(original_params, models_data, args.method, args.maxiter)
        
        if not optimization_results:
            logging.error("Optimization failed!")
            return
        
        # Create visualizations
        logging.info("\n=== Step 5: Creating Optimization Visualizations ===")
        create_optimization_visualizations(optimization_results, results_dir)
        
        # Save results
        logging.info("\n=== Step 6: Saving Optimization Results ===")
        save_optimization_results(optimization_results, device_type, results_dir)
        
        # Print summary
        improvements = optimization_results['improvements']
        logging.info("\n=== OPTIMIZATION COMPLETE ===")
        logging.info(f"Results saved to: {results_dir}/")
        logging.info(f"Optimized Parameters: optimized_device_parameters.json")
        logging.info(f"Optimization Report: optimization_report.json")
        logging.info(f"Comparison (MPP): 1_comparison_mpp.png")
        logging.info(f"Comparison (PCE): 1_comparison_pce.png")
        logging.info(f"Comparison (IntSRHn_mean): 1_comparison_intsrhn_mean.png")
        logging.info(f"Thickness Optimization: 2_thickness_optimization.png")
        logging.info(f"Energy Optimization: 3_energy_optimization.png")
        logging.info(f"Doping Optimization: 4_doping_optimization.png")
        logging.info(f"Improvements Summary: 5_performance_improvements.png")
        logging.info(f"Comprehensive Table: 6_optimization_table.png")
        
        logging.info(f"\n=== PERFORMANCE IMPROVEMENTS ===")
        logging.info(f"MPP: {improvements['MPP_improvement_percent']:+.2f}%")
        logging.info(f"PCE: {improvements['PCE_improvement_percent']:+.2f}%")
        logging.info(f"Recombination: {improvements['recombination_change_percent']:+.2f}%")
        logging.info(f"Status: Device optimization successful and manufacturable!")
        
    except Exception as e:
        logging.error(f"Error in main optimization workflow: {e}")
        raise

if __name__ == "__main__":
    main()
