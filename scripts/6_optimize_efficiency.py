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
    """Load all trained optimization models."""
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
    
    # Load inverse models
    inverse_models = {}
    for target in metadata['efficiency_targets']:
        inverse_models[target] = {}
        for param in metadata['device_params']:
            model_path = os.path.join(models_dir, f'inverse_{target}_{param}.joblib')
            if os.path.exists(model_path):
                inverse_models[target][param] = joblib.load(model_path)
    
    logging.info(f"Loaded {len(efficiency_models)} efficiency models")
    logging.info(f"Loaded {len(recombination_models)} recombination models")
    logging.info(f"Loaded inverse models for {len(inverse_models)} targets")
    
    return efficiency_models, efficiency_scalers, recombination_models, recombination_scalers, inverse_models, metadata

def predict_efficiency(device_params, efficiency_models, efficiency_scalers, target='MPP'):
    """Predict efficiency for given device parameters."""
    if target not in efficiency_models:
        raise ValueError(f"Model not found for target: {target}")
    
    # Scale input parameters
    scaler = efficiency_scalers[target]
    X_scaled = scaler.transform([device_params])
    
    # Make prediction
    model = efficiency_models[target]
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

def objective_function(device_params, efficiency_models, efficiency_scalers, target='MPP'):
    """Objective function for optimization (maximize efficiency)."""
    try:
        efficiency = predict_efficiency(device_params, efficiency_models, efficiency_scalers, target)
        # Return negative efficiency since we're minimizing
        return -efficiency
    except Exception as e:
        logging.warning(f"Error in objective function: {e}")
        return 1e6  # Return large value for failed predictions

def constraint_recombination(device_params, recombination_models, recombination_scalers, 
                           max_recombination=1e-3, target='IntSRHn_mean'):
    """Constraint function to limit recombination rate."""
    try:
        recombination = predict_recombination(device_params, recombination_models, recombination_scalers, target)
        return max_recombination - recombination  # Must be >= 0
    except Exception as e:
        logging.warning(f"Error in recombination constraint: {e}")
        return -1e6  # Return negative value for failed predictions

def get_parameter_bounds(metadata):
    """Get parameter bounds for optimization."""
    # Load original data to get parameter ranges
    data_path = 'results/generate_enhanced/combined_output_with_efficiency.csv'
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        bounds = []
        
        for param in metadata['device_params']:
            if param in df.columns:
                min_val = df[param].min()
                max_val = df[param].max()
                bounds.append((min_val, max_val))
            else:
                # Default bounds if parameter not found
                bounds.append((0.0, 1.0))
        
        return bounds
    else:
        # Default bounds based on parameter types
        bounds = []
        for param in metadata['device_params']:
            if 'L' in param:  # Layer thickness
                bounds.append((1e-9, 1e-6))  # 1 nm to 1 μm
            elif 'E_' in param:  # Energy levels
                bounds.append((1.0, 10.0))  # 1-10 eV
            elif 'N_' in param:  # Doping concentrations
                bounds.append((1e18, 1e22))  # 1e18 to 1e22 m⁻³
            else:
                bounds.append((0.0, 1.0))
        
        return bounds

def optimize_for_maximum_efficiency(efficiency_models, efficiency_scalers, 
                                  recombination_models, recombination_scalers,
                                  metadata, target='MPP', max_recombination=None):
    """Optimize device parameters for maximum efficiency."""
    logging.info(f"\n=== Optimizing for maximum {target} ===")
    
    # Get parameter bounds
    bounds = get_parameter_bounds(metadata)
    device_params = metadata['device_params']
    
    logging.info(f"Optimizing {len(device_params)} parameters")
    logging.info(f"Parameter bounds: {bounds}")
    
    # Define constraints
    constraints = []
    if max_recombination is not None:
        constraint = {
            'type': 'ineq',
            'fun': lambda x: constraint_recombination(x, recombination_models, recombination_scalers, max_recombination)
        }
        constraints.append(constraint)
        logging.info(f"Added recombination constraint: max IntSRHn = {max_recombination:.2e}")
    
    # Initial guess (middle of bounds)
    x0 = [(b[0] + b[1]) / 2 for b in bounds]
    
    # Optimize using different methods
    optimization_results = {}
    
    # Method 1: L-BFGS-B (local optimization)
    try:
        logging.info("Running L-BFGS-B optimization...")
        result_lbfgs = minimize(
            objective_function, x0, args=(efficiency_models, efficiency_scalers, target),
            method='L-BFGS-B', bounds=bounds, constraints=constraints,
            options={'maxiter': 1000}
        )
        optimization_results['L-BFGS-B'] = result_lbfgs
        logging.info(f"L-BFGS-B result: success={result_lbfgs.success}, efficiency={-result_lbfgs.fun:.4f}")
    except Exception as e:
        logging.error(f"L-BFGS-B optimization failed: {e}")
    
    # Method 2: Differential Evolution (global optimization)
    try:
        logging.info("Running Differential Evolution optimization...")
        result_de = differential_evolution(
            objective_function, bounds, args=(efficiency_models, efficiency_scalers, target),
            constraints=constraints, maxiter=100, popsize=10, seed=42
        )
        optimization_results['Differential Evolution'] = result_de
        logging.info(f"Differential Evolution result: success={result_de.success}, efficiency={-result_de.fun:.4f}")
    except Exception as e:
        logging.error(f"Differential Evolution optimization failed: {e}")
    
    # Find best result
    best_result = None
    best_efficiency = -np.inf
    
    for method, result in optimization_results.items():
        if result.success and -result.fun > best_efficiency:
            best_result = result
            best_efficiency = -result.fun
    
    if best_result is None:
        logging.error("All optimization methods failed")
        return None
    
    # Get optimal parameters
    optimal_params = best_result.x
    optimal_efficiency = -best_result.fun
    
    # Predict recombination for optimal parameters
    optimal_recombination = predict_recombination(optimal_params, recombination_models, recombination_scalers)
    
    # Create results dictionary
    results = {
        'optimal_parameters': dict(zip(device_params, optimal_params)),
        'optimal_efficiency': optimal_efficiency,
        'optimal_recombination': optimal_recombination,
        'optimization_method': [k for k, v in optimization_results.items() if v == best_result][0],
        'all_results': optimization_results
    }
    
    logging.info(f"\nOptimization Results:")
    logging.info(f"Optimal efficiency ({target}): {optimal_efficiency:.4f}")
    logging.info(f"Optimal recombination (IntSRHn): {optimal_recombination:.2e}")
    logging.info(f"Optimization method: {results['optimization_method']}")
    
    return results

def validate_optimization_results(results, efficiency_models, efficiency_scalers,
                                recombination_models, recombination_scalers, metadata):
    """Validate optimization results with physics constraints."""
    logging.info("\n=== Validating Optimization Results ===")
    
    optimal_params = list(results['optimal_parameters'].values())
    
    # Check parameter bounds
    bounds = get_parameter_bounds(metadata)
    violations = []
    
    for i, (param, value) in enumerate(results['optimal_parameters'].items()):
        if i < len(bounds):
            min_val, max_val = bounds[i]
            if value < min_val or value > max_val:
                violations.append(f"{param}: {value:.2e} (bounds: {min_val:.2e} - {max_val:.2e})")
    
    if violations:
        logging.warning(f"Parameter bound violations: {violations}")
    else:
        logging.info("All parameters within bounds")
    
    # Check recombination constraint
    recombination = results['optimal_recombination']
    if recombination < 0:
        logging.warning(f"Negative recombination rate: {recombination:.2e}")
    
    # Check efficiency prediction
    predicted_efficiency = predict_efficiency(optimal_params, efficiency_models, efficiency_scalers)
    actual_efficiency = results['optimal_efficiency']
    
    if abs(predicted_efficiency - actual_efficiency) > 1e-6:
        logging.warning(f"Efficiency prediction mismatch: {predicted_efficiency:.4f} vs {actual_efficiency:.4f}")
    
    return len(violations) == 0

def create_optimization_report(results, metadata):
    """Create comprehensive optimization report."""
    logging.info("\n=== Creating Optimization Report ===")
    
    report = {
        'optimization_date': datetime.now().isoformat(),
        'target_efficiency': 'MPP',
        'optimal_efficiency': results['optimal_efficiency'],
        'optimal_recombination': results['optimal_recombination'],
        'optimization_method': results['optimization_method'],
        'optimal_parameters': results['optimal_parameters'],
        'parameter_bounds': get_parameter_bounds(metadata),
        'validation_passed': True,  # Will be updated by validation function
        'recommendations': []
    }
    
    # Add recommendations based on results
    optimal_params = results['optimal_parameters']
    
    # Analyze layer thicknesses
    thickness_params = [p for p in optimal_params.keys() if 'L' in p and 'L_' in p]
    for param in thickness_params:
        value = optimal_params[param]
        if 'L1' in param:  # PCBM layer
            if value < 30e-9:
                report['recommendations'].append(f"Consider increasing {param} (current: {value:.1f} nm) for better electron transport")
            elif value > 50e-9:
                report['recommendations'].append(f"Consider decreasing {param} (current: {value:.1f} nm) to reduce recombination")
        elif 'L2' in param:  # MAPI layer
            if value < 300e-9:
                report['recommendations'].append(f"Consider increasing {param} (current: {value:.1f} nm) for better light absorption")
            elif value > 500e-9:
                report['recommendations'].append(f"Consider decreasing {param} (current: {value:.1f} nm) to reduce bulk recombination")
        elif 'L3' in param:  # PEDOT layer
            if value < 30e-9:
                report['recommendations'].append(f"Consider increasing {param} (current: {value:.1f} nm) for better hole transport")
            elif value > 50e-9:
                report['recommendations'].append(f"Consider decreasing {param} (current: {value:.1f} nm) to reduce recombination")
    
    # Analyze energy levels
    energy_params = [p for p in optimal_params.keys() if 'E_' in p]
    for param in energy_params:
        value = optimal_params[param]
        if 'E_c' in param:  # Conduction band
            if value < 3.5:
                report['recommendations'].append(f"Consider increasing {param} (current: {value:.2f} eV) for better electron blocking")
            elif value > 4.5:
                report['recommendations'].append(f"Consider decreasing {param} (current: {value:.2f} eV) to reduce energy barriers")
        elif 'E_v' in param:  # Valence band
            if value < 5.0:
                report['recommendations'].append(f"Consider increasing {param} (current: {value:.2f} eV) to reduce hole barriers")
            elif value > 6.0:
                report['recommendations'].append(f"Consider decreasing {param} (current: {value:.2f} eV) for better hole transport")
    
    # Save report
    report_path = 'results/optimize_efficiency/reports/optimization_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logging.info(f"Optimization report saved to: {report_path}")
    
    return report

def create_optimization_plots(results, metadata):
    """Create visualization plots for optimization results."""
    logging.info("\n=== Creating Optimization Plots ===")
    
    plots_dir = 'results/optimize_efficiency/plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Parameter comparison plot
    optimal_params = results['optimal_parameters']
    
    # Group parameters by type
    thickness_params = [p for p in optimal_params.keys() if 'L' in p and 'L_' in p]
    energy_params = [p for p in optimal_params.keys() if 'E_' in p]
    doping_params = [p for p in optimal_params.keys() if 'N_' in p]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Thickness parameters
    if thickness_params:
        thickness_values = [optimal_params[p] for p in thickness_params]
        thickness_labels = [p.replace('_', ' ') for p in thickness_params]
        
        axes[0, 0].bar(range(len(thickness_params)), thickness_values)
        axes[0, 0].set_xticks(range(len(thickness_params)))
        axes[0, 0].set_xticklabels(thickness_labels, rotation=45)
        axes[0, 0].set_ylabel('Thickness (m)')
        axes[0, 0].set_title('Optimal Layer Thicknesses')
        axes[0, 0].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # Energy parameters
    if energy_params:
        energy_values = [optimal_params[p] for p in energy_params]
        energy_labels = [p.replace('_', ' ') for p in energy_params]
        
        axes[0, 1].bar(range(len(energy_params)), energy_values)
        axes[0, 1].set_xticks(range(len(energy_params)))
        axes[0, 1].set_xticklabels(energy_labels, rotation=45)
        axes[0, 1].set_ylabel('Energy (eV)')
        axes[0, 1].set_title('Optimal Energy Levels')
    
    # Doping parameters
    if doping_params:
        doping_values = [optimal_params[p] for p in doping_params]
        doping_labels = [p.replace('_', ' ') for p in doping_params]
        
        axes[1, 0].bar(range(len(doping_params)), doping_values)
        axes[1, 0].set_xticks(range(len(doping_params)))
        axes[1, 0].set_xticklabels(doping_labels, rotation=45)
        axes[1, 0].set_ylabel('Concentration (m⁻³)')
        axes[1, 0].set_title('Optimal Doping Concentrations')
        axes[1, 0].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # Efficiency vs Recombination
    axes[1, 1].scatter(results['optimal_recombination'], results['optimal_efficiency'], 
                       s=100, color='red', alpha=0.7)
    axes[1, 1].set_xlabel('IntSRHn (A/m²)')
    axes[1, 1].set_ylabel('MPP (W/m²)')
    axes[1, 1].set_title('Optimal Point')
    axes[1, 1].set_xscale('log')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/optimization_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Parameter importance plot (if available)
    if 'all_results' in results:
        # Create a summary of all optimization attempts
        methods = list(results['all_results'].keys())
        efficiencies = [-results['all_results'][m].fun for m in methods]
        success = [results['all_results'][m].success for m in methods]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Efficiency comparison
        colors = ['green' if s else 'red' for s in success]
        ax1.bar(methods, efficiencies, color=colors, alpha=0.7)
        ax1.set_ylabel('MPP (W/m²)')
        ax1.set_title('Optimization Method Comparison')
        ax1.tick_params(axis='x', rotation=45)
        
        # Success rate
        success_rate = [1 if s else 0 for s in success]
        ax2.bar(methods, success_rate, color=colors, alpha=0.7)
        ax2.set_ylabel('Success (1=Yes, 0=No)')
        ax2.set_title('Optimization Success')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{plots_dir}/optimization_methods.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    logging.info(f"Optimization plots saved to {plots_dir}")

def main():
    """Main optimization function."""
    logging.info("Starting efficiency optimization...")
    
    # Load models
    efficiency_models, efficiency_scalers, recombination_models, recombination_scalers, inverse_models, metadata = load_optimization_models()
    
    # Run optimization
    results = optimize_for_maximum_efficiency(
        efficiency_models, efficiency_scalers,
        recombination_models, recombination_scalers,
        metadata, target='MPP', max_recombination=1e-3
    )
    
    if results is None:
        logging.error("Optimization failed")
        return
    
    # Validate results
    validation_passed = validate_optimization_results(
        results, efficiency_models, efficiency_scalers,
        recombination_models, recombination_scalers, metadata
    )
    
    # Create report
    report = create_optimization_report(results, metadata)
    report['validation_passed'] = validation_passed
    
    # Create plots
    create_optimization_plots(results, metadata)
    
    # Print summary
    logging.info("\n=== Optimization Complete ===")
    logging.info(f"Optimal efficiency: {results['optimal_efficiency']:.4f} W/m²")
    logging.info(f"Optimal recombination: {results['optimal_recombination']:.2e} A/m²")
    logging.info(f"Validation passed: {validation_passed}")
    logging.info(f"Results saved to: results/optimize_efficiency/")
    
    # Print optimal parameters
    logging.info("\nOptimal Device Parameters:")
    for param, value in results['optimal_parameters'].items():
        if 'L' in param and 'L_' in param:
            logging.info(f"  {param}: {value:.1f} nm")
        elif 'E_' in param:
            logging.info(f"  {param}: {value:.2f} eV")
        elif 'N_' in param:
            logging.info(f"  {param}: {value:.2e} m⁻³")
        else:
            logging.info(f"  {param}: {value:.2e}")

if __name__ == "__main__":
    main() 