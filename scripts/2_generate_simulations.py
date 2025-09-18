"""
===============================================================================
GENERATE PHYSICS-VALIDATED SIMULATIONS
===============================================================================

PURPOSE:
This script generates parameter combinations with physics validation, creates simulation directories, 
and runs physics-based simulations for solar cell optimization. It includes validation to prevent 
unphysical parameter combinations that lead to unrealistic results.

WHAT THIS SCRIPT DOES:
1. Reads parameter ranges from sim/parameters.txt
2. Generates parameter combinations for each device layer
3. VALIDATES PHYSICS CONSTRAINTS (energy gaps, thicknesses, doping, band alignment)
4. Rejects unphysical combinations (negative energy gaps, extreme values)
5. Creates simulation directories only for valid parameter sets
6. Runs physics simulations (simss.exe) for validated parameter combinations
7. Logs validation statistics and simulation results

PHYSICS VALIDATION CHECKS:
- Energy gaps must be positive (accounting for SimSalabim's negative energy convention)
- Layer thicknesses must be reasonable (>1 nm)
- Doping concentrations must be realistic (<1e22 m^-3)
- Doping imbalances must be moderate (<100:1 ratio)
- Energy gaps must be in semiconductor range (0.5-4.0 eV)
- Layer thickness ratios must be reasonable (active layer >> transport layers)
- Transport layer thickness balance (<10:1 ratio)

BENEFITS:
- Prevents numerical instabilities that cause extreme MPP values (>1000 W/cm²)
- Eliminates failed device configurations (negative MPP)
- Ensures realistic solar cell parameter combinations
- Improves simulation stability and data quality
- Reduces computational waste on unphysical/unstable simulations

IMPORTANT:
- This script DOES NOT extract or combine simulation results.
- To extract MPP and IntSRHn_mean data, run scripts/3_extract_simulation_data.py after simulations complete.

INPUT FILES:
- sim/parameters.txt (parameter ranges)
- sim/simulation_setup.txt, L1_parameters.txt, L2_parameters.txt, L3_parameters.txt (simulation setup)
- sim/Data/ (material property files)
- sim/simss.exe (simulation executable)

OUTPUT FILES:
- sim/simulations/sim_XXXX/ (simulation output folders)
- results/generate_enhanced/simulation_enhanced.log (log file)

USAGE:
python scripts/2_generate_simulations_enhanced.py

AUTHOR: [ANTHONY BAKER]
DATE: 2025
===============================================================================
"""
import numpy as np
import os
import shutil
import logging
from datetime import datetime
import itertools
import subprocess
import json
import pandas as pd

# Define base paths - updated for scripts folder location
SCRIPT_DIR = os.path.dirname(__file__)  # scripts folder
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # project root
SIM_DIR = os.path.join(PROJECT_ROOT, 'sim')  # sim folder
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'generated_simulations')

# Configure logging
os.makedirs(RESULTS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(RESULTS_DIR, 'generated_simulations.log')),
        logging.StreamHandler()
    ]
)

# Maximum number of parameter combinations to generate
MAX_COMBINATIONS = 10

def parse_parameters(param_file):
    """Parse parameters from file, organizing them by layer and making names unique per layer."""
    params = []
    current_layer = None
    
    with open(param_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.lower().startswith('layer'):
                current_layer = int(line.split()[1])
                continue
            if current_layer is not None:
                try:
                    name, values_str = line.split('=')
                    name = name.strip()
                    # Add layer number to parameter name to make it unique
                    name = f"L{current_layer}_{name}"
                    values = eval(values_str.strip())
                    param = {
                        'layer': current_layer,
                        'name': name,
                        'min': float(values[0]),
                        'max': float(values[1]),
                        'points': int(values[2]),
                        'log_scale': bool(values[3])
                    }
                    params.append(param)
                except (ValueError, SyntaxError) as e:
                    logging.error(f"Error parsing line '{line}': {e}")
    return params

def generate_parameter_values(params):
    """Generate parameter values based on ranges and points, organized by layer."""
    layer_params = {}
    
    # First pass: collect all parameters and their ranges by layer
    for param in params:
        layer = param['layer']
        if layer not in layer_params:
            layer_params[layer] = {'names': [], 'values': []}
            
        name = param['name']
        min_val = float(param['min'])
        max_val = float(param['max'])
        points = int(param['points'])
        
        layer_params[layer]['names'].append(name)
        if param['log_scale'] and min_val > 0 and max_val > 0:
            values = np.logspace(np.log10(min_val), np.log10(max_val), points)
        else:
            values = np.linspace(min_val, max_val, points)
        layer_params[layer]['values'].append(values)
        logging.info(f"Layer {layer} - {name}: {len(values)} values from {min_val} to {max_val}")
    
    return layer_params

def update_layer_file(layer_file, param_values, layer_num):
    """Update a layer parameter file with new values for the specific layer only."""
    with open(layer_file, 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        line_strip = line.strip()
        if not line_strip or line_strip.startswith('#'):
            continue
        parts = line_strip.split('=')
        if len(parts) != 2:
            continue
        param_name = parts[0].strip()
        comment = parts[1].split('*')[1] if '*' in parts[1] else ''
        # Look for the parameter with the correct layer prefix
        key = f"L{layer_num}_{param_name}"
        if key in param_values:
            value = param_values[key]
            formatted_value = f"{value:.2E}"
            lines[i] = f"{param_name} = {formatted_value}                    * {comment}\n"
            logging.info(f"Updated {param_name} in L{layer_num}_parameters.txt to {formatted_value}")
    with open(layer_file, 'w') as f:
        f.writelines(lines)

def create_simulation_directory(sim_id, param_values):
    """Create a directory for a simulation and copy necessary files."""
    # Create simulations directory in sim folder
    base_dir = os.path.join(SIM_DIR, 'simulations')
    os.makedirs(base_dir, exist_ok=True)
    
    # Find the next available simulation number
    existing = [d for d in os.listdir(base_dir) if d.startswith('sim_') and d[4:8].isdigit()]
    if existing:
        max_index = max(int(d[4:8]) for d in existing)
        next_index = max_index + 1
    else:
        next_index = 1
    
    sim_dir = os.path.join(base_dir, f'sim_{next_index:04d}')
    os.makedirs(sim_dir, exist_ok=True)
    data_dir = os.path.join(sim_dir, 'Data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Files to copy from sim directory
    files_to_copy = ['simss.exe', 'simulation_setup.txt', 'L1_parameters.txt', 'L2_parameters.txt', 'L3_parameters.txt']
    for file in files_to_copy:
        source_path = os.path.join(SIM_DIR, file)
        dest_path = os.path.join(sim_dir, file)
        try:
            shutil.copy2(source_path, dest_path)
            logging.info(f"Copied {file} to {sim_dir}")
        except Exception as e:
            logging.error(f"Failed to copy {file}: {e}")
    
    # Data files to copy from sim/Data directory
    data_files = [
        'nk_SiO2.txt', 'nk_ITO.txt', 'nk_PEDOT.txt', 'nk_Au.txt',
        'AM15G.txt', 'nk_PCBM.txt', 'nk_MAPI.txt'
    ]
    for file in data_files:
        source_path = os.path.join(SIM_DIR, 'Data', file)
        dest_path = os.path.join(data_dir, file)
        try:
            shutil.copy2(source_path, dest_path)
            logging.info(f"Copied {file} to {data_dir}")
        except Exception as e:
            logging.error(f"Failed to copy data file {file}: {e}")
    
    # Update each layer file with only its parameters
    for layer_num in [1, 2, 3]:
        update_layer_file(os.path.join(sim_dir, f"L{layer_num}_parameters.txt"), param_values, layer_num)
    
    with open(os.path.join(sim_dir, 'parameters.json'), 'w') as f:
        json.dump(param_values, f, indent=4)
    
    return sim_dir

def run_simulation(sim_dir):
    """Run the simulation in the specified directory."""
    try:
        # Store current directory
        original_dir = os.getcwd()
        
        # Change to simulation directory
        os.chdir(sim_dir)
        
        # Run simulation
        result = subprocess.run(['./simss.exe'], 
                              capture_output=True, 
                              text=True, 
                              check=False)
        
        # Change back to original directory
        os.chdir(original_dir)
        
        # Consider return code 0, empty return code, and return code 95 (non-convergence) as success
        if result.returncode in [0, 95] or result.returncode is None:
            return result
        else:
            raise subprocess.CalledProcessError(result.returncode, './simss.exe', result.stdout, result.stderr)
    except Exception as e:
        # Change back to original directory in case of error
        os.chdir(original_dir)
        raise e

def validate_physics_constraints(params_dict):
    """Validate that parameter combination satisfies basic physics constraints.
    
    NOTE: SimSalabim uses NEGATIVE energy values internally, so:
    - Input E_c = 3.7 becomes E_c = -3.7 eV in simulation
    - Input E_v = 5.7 becomes E_v = -5.7 eV in simulation
    - Energy gap = (-3.7) - (-5.7) = +2.0 eV (positive, as expected)
    """
    
    # Check 1: Energy gaps must be positive (accounting for SimSalabim convention)
    # SimSalabim: Gap = (-E_c_input) - (-E_v_input) = E_v_input - E_c_input
    for layer in ['L1', 'L2', 'L3']:
        ec_input = params_dict[f'{layer}_E_c']  # Input value (becomes -E_c in sim)
        ev_input = params_dict[f'{layer}_E_v']  # Input value (becomes -E_v in sim)
        
        # In SimSalabim: actual_gap = (-ec_input) - (-ev_input) = ev_input - ec_input
        actual_gap = ev_input - ec_input
        
        if actual_gap <= 0:
            return False, f"{layer} has negative/zero energy gap ({actual_gap:.3f} eV) in SimSalabim"
    
    # Check 2: Layer thicknesses must be reasonable (not too thin)
    min_thickness = 1e-9  # 1 nm minimum
    for layer in ['L1', 'L2', 'L3']:
        thickness = params_dict[f'{layer}_L']
        if thickness < min_thickness:
            return False, f"{layer} thickness too small ({thickness*1e9:.2f} nm)"
    
    # Check 3: Doping concentrations must be reasonable
    max_doping = 5e21  # More restrictive maximum doping to prevent instabilities
    for layer in ['L1', 'L2', 'L3']:
        nd = params_dict[f'{layer}_N_D']
        na = params_dict[f'{layer}_N_A']
        if nd > max_doping or na > max_doping:
            return False, f"{layer} has unrealistic doping concentration (max: 5e21 m^-3)"
    
    # Check 4: Energy level ordering should make sense (in SimSalabim convention)
    # In SimSalabim: E_c = -E_c_input, E_v = -E_v_input
    # For proper band alignment: E_c should be higher than E_v (less negative)
    # This means E_c_input should be smaller than E_v_input (which we already check above)
    
    # Check 5: Reasonable energy level ranges
    for layer in ['L1', 'L2', 'L3']:
        ec_input = params_dict[f'{layer}_E_c']
        ev_input = params_dict[f'{layer}_E_v']
        
        # Energy levels should be in reasonable range for semiconductors
        if ec_input < 1.0 or ec_input > 6.0:
            return False, f"{layer} E_c ({ec_input:.2f} eV) outside reasonable range [1.0, 6.0]"
        if ev_input < 1.0 or ev_input > 8.0:
            return False, f"{layer} E_v ({ev_input:.2f} eV) outside reasonable range [1.0, 8.0]"
    
    # Check 6: Prevent extreme doping imbalances that can cause numerical instabilities
    for layer in ['L1', 'L2', 'L3']:
        nd = params_dict[f'{layer}_N_D']
        na = params_dict[f'{layer}_N_A']
        doping_ratio = max(nd/na, na/nd)  # Always >= 1
        
        if doping_ratio > 3.5:  # Balanced constraint - prevent extreme imbalances while allowing some variation
            return False, f"{layer} has doping imbalance (ratio: {doping_ratio:.1f}:1) - may cause extreme MPP"
    
    # Check 7: Energy gap constraints for stable devices
    for layer in ['L1', 'L2', 'L3']:
        ec_input = params_dict[f'{layer}_E_c']
        ev_input = params_dict[f'{layer}_E_v']
        actual_gap = ev_input - ec_input
        
        # Energy gap constraints for stable devices (relaxed to allow valid combinations)
        if actual_gap < 1.0:  # Prevent very small gaps that cause extreme instabilities
            return False, f"{layer} energy gap too small ({actual_gap:.2f} eV) - causes extreme MPP"
        if actual_gap > 3.0:  # Upper limit for realistic solar cells
            return False, f"{layer} energy gap too large ({actual_gap:.2f} eV) - unrealistic for solar cells"
    
    # Check 8: Layer thickness ratios should be reasonable
    l1_thickness = params_dict['L1_L']
    l2_thickness = params_dict['L2_L'] 
    l3_thickness = params_dict['L3_L']
    
    # Active layer should be much thicker than transport layers
    if l2_thickness < 5 * max(l1_thickness, l3_thickness):
        return False, f"Active layer too thin relative to transport layers"
    
    # Transport layers shouldn't be too thin relative to each other
    transport_ratio = max(l1_thickness/l3_thickness, l3_thickness/l1_thickness)
    if transport_ratio > 10:
        return False, f"Transport layer thickness imbalance (ratio: {transport_ratio:.1f}:1)"
    
    return True, "Valid physics"

def generate_parameter_combinations():
    """Generate parameter combinations for simulations with physics validation."""
    # Parse parameters from file in sim directory
    params = parse_parameters(os.path.join(SIM_DIR, 'parameters.txt'))
    layer_params = generate_parameter_values(params)
    
    # Calculate total possible combinations per layer
    total_possible = 1
    for layer in layer_params:
        layer_total = 1
        logging.info(f"\nLayer {layer} parameters:")
        for name, values in zip(layer_params[layer]['names'], layer_params[layer]['values']):
            num_values = len(values)
            layer_total *= num_values
            logging.info(f"{name}: {num_values} values")
        logging.info(f"Layer {layer} total combinations: {layer_total}")
        total_possible *= layer_total
    
    logging.info(f"\nTotal possible combinations: {total_possible}")
    
    # Determine how many combinations to generate
    num_combinations = min(total_possible, MAX_COMBINATIONS)
    logging.info(f"Generating {num_combinations} combinations")
    
    if total_possible <= MAX_COMBINATIONS:
        # If we have fewer combinations than MAX_COMBINATIONS, generate all possible combinations
        logging.info("Generating all possible combinations")
        combinations = []
        
        # Generate all possible combinations for each layer
        layer_combinations = {}
        for layer in layer_params:
            # Generate all possible combinations of values for this layer
            layer_combinations[layer] = list(itertools.product(*layer_params[layer]['values']))
            logging.info(f"Layer {layer} generated {len(layer_combinations[layer])} combinations")
        
        # Generate all possible combinations across layers with physics validation
        valid_combinations = 0
        invalid_combinations = 0
        
        for layer_combo in itertools.product(*[layer_combinations[layer] for layer in sorted(layer_params.keys())]):
            params_dict = {}
            for layer, values in zip(sorted(layer_params.keys()), layer_combo):
                for name, value in zip(layer_params[layer]['names'], values):
                    params_dict[name] = value
            
            # Validate physics constraints
            is_valid, reason = validate_physics_constraints(params_dict)
            if is_valid:
                combinations.append(params_dict)
                valid_combinations += 1
            else:
                invalid_combinations += 1
                if invalid_combinations <= 10:  # Log first 10 invalid combinations
                    logging.debug(f"Invalid combination rejected: {reason}")
        
        logging.info(f"Physics validation results:")
        logging.info(f"  Valid combinations: {valid_combinations}")
        logging.info(f"  Invalid combinations rejected: {invalid_combinations}")
        if valid_combinations + invalid_combinations > 0:
            logging.info(f"  Physics validation rate: {valid_combinations/(valid_combinations+invalid_combinations)*100:.1f}%")
        else:
            logging.warning("No combinations were generated - check parameter ranges and constraints")
    else:
        # If we have more combinations than MAX_COMBINATIONS, randomly sample with physics validation
        logging.info("Randomly sampling combinations with physics validation")
        combinations = []
        valid_combinations = 0
        invalid_combinations = 0
        attempts = 0
        max_attempts = num_combinations * 10  # Allow up to 10x attempts to find valid combinations
        
        while len(combinations) < num_combinations and attempts < max_attempts:
            attempts += 1
            
            # Randomly select one value from each parameter's range for each layer
            combo = {}
            for layer in sorted(layer_params.keys()):
                for name, values in zip(layer_params[layer]['names'], layer_params[layer]['values']):
                    combo[name] = np.random.choice(values)
            
            # Validate physics constraints
            is_valid, reason = validate_physics_constraints(combo)
            if is_valid:
                combinations.append(combo)
                valid_combinations += 1
            else:
                invalid_combinations += 1
                if invalid_combinations <= 10:  # Log first 10 invalid combinations
                    logging.debug(f"Invalid combination rejected: {reason}")
        
        logging.info(f"Random sampling with physics validation results:")
        logging.info(f"  Attempts: {attempts}")
        logging.info(f"  Valid combinations: {valid_combinations}")
        logging.info(f"  Invalid combinations rejected: {invalid_combinations}")
        if valid_combinations + invalid_combinations > 0:
            logging.info(f"  Physics validation rate: {valid_combinations/(valid_combinations+invalid_combinations)*100:.1f}%")
        else:
            logging.warning("No valid combinations found - constraints may be too restrictive")
        
        if len(combinations) < num_combinations:
            logging.warning(f"Could only generate {len(combinations)} valid combinations out of {num_combinations} requested")
            logging.warning("Consider relaxing physics constraints or increasing parameter ranges")
    
    # Final summary
    total_generated = len(combinations)
    logging.info(f"\n=== PARAMETER GENERATION SUMMARY ===")
    logging.info(f"Total valid combinations generated: {total_generated}")
    if total_generated > 0:
        logging.info("Physics validation prevented generation of unphysical devices")
        logging.info("Expected improvements:")
        logging.info("  - No negative energy gaps → No extreme MPP values (>1000 W/cm²)")
        logging.info("  - No failed devices → No negative MPP values")
        logging.info("  - Better data quality → Improved ML model performance")
    
    return combinations

def main():
    """Main function to generate and run enhanced simulations."""
    # Create simulations directory in sim folder
    sim_dir = os.path.join(SIM_DIR, 'simulations')
    os.makedirs(sim_dir, exist_ok=True)
    
    # Create results/generate_enhanced directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Generate parameter combinations
    param_combinations = generate_parameter_combinations()
    total_sims = len(param_combinations)
    
    logging.info(f"Starting generation of {total_sims} enhanced simulations")
    
    successful = 0
    failed = 0
    
    for i, params in enumerate(param_combinations, 1):
        logging.info(f"\nSimulation {i}/{total_sims} parameters:")
        for param, value in params.items():
            logging.info(f"{param}: {value}")
        
        # Create simulation directory and copy files
        sim_path = create_simulation_directory(i, params)
        
        # Run simulation
        logging.info(f"Running simulation {i}/{total_sims}")
        try:
            result = run_simulation(sim_path)
            if result.returncode in [0, 95] or result.returncode is None:
                successful += 1
                if result.returncode == 95:
                    logging.info(f"Simulation {i} completed with non-convergence (return code 95)")
                else:
                    logging.info(f"Simulation {i} completed successfully")
            else:
                failed += 1
                logging.error(f"Simulation {i} failed with return code {result.returncode}")
                logging.error(f"Error: {result.stderr}")
        except Exception as e:
            failed += 1
            logging.error(f"Error running simulation {i}: {str(e)}")
    
    # Print summary
    logging.info("\nEnhanced Simulation Summary:")
    logging.info(f"Total simulations: {total_sims}")
    logging.info(f"Successful: {successful}")
    logging.info(f"Failed: {failed}")
    logging.info(f"Success rate: {(successful/total_sims)*100:.2f}%")

if __name__ == "__main__":
    main()