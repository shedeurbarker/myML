"""
Solar Cell Efficiency Optimization Script - MODIFIED FOR PCE OPTIMIZATION
===============================================================================

PURPOSE:
--------
Finds optimal device parameters that maximize PCE (Power Conversion Efficiency) 
while respecting recombination constraints.

MODIFIED VERSION:
- Optimizes for PCE instead of MPP
- Uses PCE prediction models from modified training script
- Focuses on recombination-PCE relationship

OPTIMIZATION TARGETS:
- Primary: Maximize PCE (Power Conversion Efficiency)
- Constraint: Limit recombination rates to physically meaningful values

ALGORITHMS:
- L-BFGS-B (local optimization for fine-tuning)
- Differential Evolution (global optimization for exploration)

INPUT:
- Trained PCE and recombination models from script 5_modified
- Parameter bounds from feature definitions

OUTPUT:
- Optimal device parameters
- Predicted optimal PCE and recombination rates
- Optimization reports and visualizations

USAGE:
python scripts/6_optimize_efficiency_modified.py
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
log_file = os.path.join(log_dir, f'optimization_modified.log')

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
    """Load all trained optimization models - MODIFIED FOR PCE."""
    models_dir = 'results/train_optimization_models/models'
    
    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Models directory not found: {models_dir}")
    
    # Load metadata
    metadata_path = os.path.join(models_dir, 'metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Load PCE models
    pce_models = {}
    pce_scalers = {}
    for target in metadata['pce_targets']:
        model_path = os.path.join(models_dir, f'pce_{target}.joblib')
        scaler_path = os.path.join(models_dir, f'pce_{target}_scaler.joblib')
        
        if os.path.exists(model_path):
            pce_models[target] = joblib.load(model_path)
            pce_scalers[target] = joblib.load(scaler_path)
    
    # Load recombination models
    recombination_models = {}
    recombination_scalers = {}
    for target in metadata['recombination_targets']:
        model_path = os.path.join(models_dir, f'recombination_{target}.joblib')
        scaler_path = os.path.join(models_dir, f'recombination_{target}_scaler.joblib')
        
        if os.path.exists(model_path):
            recombination_models[target] = joblib.load(model_path)
            recombination_scalers[target] = joblib.load(scaler_path)
    
    logging.info(f"Loaded {len(pce_models)} PCE models")
    logging.info(f"Loaded {len(recombination_models)} recombination models")
    
    return pce_models, pce_scalers, recombination_models, recombination_scalers, metadata

def predict_pce(device_params, pce_models, pce_scalers, target='PCE'):
    """Predict PCE for given device parameters."""
    if target not in pce_models:
        raise ValueError(f"Model not found for target: {target}")
    
    # Scale input parameters
    scaler = pce_scalers[target]
    X_scaled = scaler.transform([device_params])
    
    # Make prediction
    model = pce_models[target]
    prediction = model.predict(X_scaled)[0]
    
    return prediction

def predict_recombination(device_params, recombination_models, recombination_scalers, target='IntSRHn_mean'):
    """Predict recombination rate for given device parameters."""
    if target not in recombination_models:
        raise ValueError(f"Model not found for target: {target}")
    
    # Scale input parameters
    scaler = recombination_scalers[target]
    X_scaled = scaler.transform([device_params])
    
    # Make prediction
    model = recombination_models[target]
    prediction = model.predict(X_scaled)[0]
    
    return prediction

def objective_function(device_params, pce_models, pce_scalers, target='PCE'):
    """Objective function: maximize PCE (negative for minimization)."""
    pce = predict_pce(device_params, pce_models, pce_scalers, target)
    return -pce  # Negative because we want to maximize

def constraint_recombination(device_params, recombination_models, recombination_scalers, 
                           max_recombination=1e-3, target='IntSRHn_mean'):
    """Constraint function: recombination rate must be below threshold."""
    recombination = predict_recombination(device_params, recombination_models, recombination_scalers, target)
    return max_recombination - recombination  # Must be >= 0

def get_parameter_bounds(metadata):
    """Get parameter bounds for optimization."""
    # Load feature definitions for bounds
    try:
        with open('results/feature_definitions.json', 'r') as f:
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

def optimize_for_maximum_pce(pce_models, pce_scalers, recombination_models, recombination_scalers,
                            metadata, target='PCE', max_recombination=None):
    """Optimize device parameters for maximum PCE."""
    logging.info(f"\n=== Optimizing for Maximum PCE ===")
    
    # Get parameter bounds
    bounds = get_parameter_bounds(metadata)
    n_params = len(bounds)
    
    logging.info(f"Optimizing {n_params} parameters for maximum PCE")
    logging.info(f"Parameter bounds: {bounds}")
    
    # Define constraints
    constraints = []
    if max_recombination is not None:
        constraints.append({
            'type': 'ineq',
            'fun': lambda x: constraint_recombination(x, recombination_models, recombination_scalers, 
                                                    max_recombination, 'IntSRHn_mean')
        })
        logging.info(f"Recombination constraint: max_recombination = {max_recombination}")
    
    # Method 1: L-BFGS-B (local optimization)
    logging.info("\n--- Method 1: L-BFGS-B Optimization ---")
    
    # Generate multiple starting points
    n_starts = 10
    best_result_lbfgs = None
    best_pce_lbfgs = -np.inf
    
    for i in range(n_starts):
        # Random starting point within bounds
        x0 = np.array([np.random.uniform(b[0], b[1]) for b in bounds])
        
        try:
            result = minimize(
                objective_function,
                x0,
                args=(pce_models, pce_scalers, target),
                method='L-BFGS-B',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success and -result.fun > best_pce_lbfgs:
                best_result_lbfgs = result
                best_pce_lbfgs = -result.fun
                logging.info(f"L-BFGS-B run {i+1}: PCE = {best_pce_lbfgs:.4f}%")
        
        except Exception as e:
            logging.warning(f"L-BFGS-B run {i+1} failed: {e}")
    
    # Method 2: Differential Evolution (global optimization)
    logging.info("\n--- Method 2: Differential Evolution ---")
    
    try:
        result_de = differential_evolution(
            objective_function,
            bounds,
            args=(pce_models, pce_scalers, target),
            constraints=constraints,
            maxiter=1000,
            popsize=15,
            seed=42
        )
        
        if result_de.success:
            pce_de = -result_de.fun
            logging.info(f"Differential Evolution: PCE = {pce_de:.4f}%")
        else:
            logging.warning("Differential Evolution failed to converge")
            pce_de = -np.inf
    
    except Exception as e:
        logging.error(f"Differential Evolution failed: {e}")
        pce_de = -np.inf
    
    # Compare results and select best
    if best_pce_lbfgs > pce_de:
        best_result = best_result_lbfgs
        best_method = "L-BFGS-B"
        best_pce = best_pce_lbfgs
    else:
        best_result = result_de
        best_method = "Differential Evolution"
        best_pce = pce_de
    
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
    
    # Create results dictionary
    results = {
        'optimization_success': best_result.success,
        'optimal_method': best_method,
        'optimal_pce': best_pce,
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
    logging.info(f"Optimal PCE: {best_pce:.4f}%")
    logging.info(f"Optimal recombination (IntSRHn_mean): {optimal_recombination.get('IntSRHn_mean', 'N/A')}")
    
    return results

def validate_optimization_results(results, pce_models, pce_scalers,
                                recombination_models, recombination_scalers, metadata):
    """Validate optimization results."""
    if results is None:
        return False
    
    logging.info("\n=== Validating Optimization Results ===")
    
    # Check if PCE is reasonable
    optimal_pce = results['optimal_pce']
    if optimal_pce < 0 or optimal_pce > 50:  # Reasonable PCE range
        logging.warning(f"Optimal PCE ({optimal_pce:.4f}%) seems unrealistic")
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
            'target': 'PCE (Power Conversion Efficiency)',
            'optimization_method': results['optimal_method'],
            'success': results['optimization_success'],
            'optimal_pce': results['optimal_pce'],
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
    print(f"Optimal PCE: {report['optimization_summary']['optimal_pce']:.4f}%")
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
    pce_values = [results['optimal_pce'], results['optimal_pce']]  # Same value for now
    
    bars = plt.bar(methods, pce_values)
    plt.ylabel('PCE (%)')
    plt.title('Optimization Methods Comparison')
    plt.ylim(0, max(pce_values) * 1.1)
    
    # Add value labels
    for bar, value in zip(bars, pce_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{value:.4f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('results/optimize_efficiency/plots/optimization_methods.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info("Optimization plots created")

def main():
    """Main optimization function."""
    logging.info("=" * 60)
    logging.info("SOLAR CELL PCE OPTIMIZATION - MODIFIED VERSION")
    logging.info("=" * 60)
    
    try:
        # Load models
        pce_models, pce_scalers, recombination_models, recombination_scalers, metadata = load_optimization_models()
        
        # Run optimization
        results = optimize_for_maximum_pce(
            pce_models, pce_scalers, recombination_models, recombination_scalers,
            metadata, target='PCE', max_recombination=1e-3
        )
        
        # Validate results
        if validate_optimization_results(results, pce_models, pce_scalers,
                                      recombination_models, recombination_scalers, metadata):
            
            # Create report and plots
            create_optimization_report(results, metadata)
            create_optimization_plots(results, metadata)
            
            logging.info("\n" + "=" * 60)
            logging.info("OPTIMIZATION COMPLETED SUCCESSFULLY")
            logging.info(f"Optimal PCE: {results['optimal_pce']:.4f}%")
            logging.info("=" * 60)
        else:
            logging.error("Optimization results validation failed")
    
    except Exception as e:
        logging.error(f"Optimization failed: {e}")
        raise

if __name__ == "__main__":
    main() 