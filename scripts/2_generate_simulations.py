"""
===============================================================================
GENERATE PHYSICS-VALIDATED SIMULATIONS
===============================================================================

PURPOSE:
This script generates parameter combinations with physics validation, creates simulation directories, 
and runs physics-based simulations for solar cell optimization. It includes validation to prevent 
unphysical parameter combinations that lead to unrealistic results.

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
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', '2_generated_simulations')

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
MAX_COMBINATIONS = 20000

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
                    # Split on '=' and handle inline comments
                    if '=' in line:
                        name, rest = line.split('=', 1)  # Split only on first '='
                        name = name.strip()
                        
                        # Extract values part (before any comment)
                        values_str = rest.split('#')[0].strip()  # Remove inline comments
                        
                        # Add layer number to parameter name to make it unique
                        name = f"L{current_layer}_{name}"
                        values = eval(values_str)
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
    """Validate that parameter combination satisfies comprehensive solar cell physics constraints.
    
    Device Structure (ETL/Active/HTL):
    - L1: Electron Transport Layer (PCBM) - n-type, facilitates electron transport
    - L2: Active Layer (MAPI) - intrinsic/undoped, light absorption  
    - L3: Hole Transport Layer (PEDOT:PSS) - p-type, facilitates hole transport
    
    SimSalabim Energy Convention:
    - Input E_c = 3.7 becomes E_c = -3.7 eV in simulation
    - Input E_v = 5.7 becomes E_v = -5.7 eV in simulation
    - Energy gap = (-3.7) - (-5.7) = +2.0 eV (positive, as expected)
    """
    
    # Extract parameters for clarity
    # L1 = ETL (Electron Transport Layer - PCBM)
    etl_thickness = params_dict['L1_L']
    etl_ec = params_dict['L1_E_c']  # Will become -E_c in SimSalabim
    etl_ev = params_dict['L1_E_v']  # Will become -E_v in SimSalabim
    etl_nd = params_dict['L1_N_D']
    etl_na = params_dict['L1_N_A']
    
    # L2 = Active Layer (MAPI Perovskite)
    active_thickness = params_dict['L2_L']
    active_ec = params_dict['L2_E_c']
    active_ev = params_dict['L2_E_v']
    active_nd = params_dict['L2_N_D']
    active_na = params_dict['L2_N_A']
    
    # L3 = HTL (Hole Transport Layer - PEDOT:PSS)
    htl_thickness = params_dict['L3_L']
    htl_ec = params_dict['L3_E_c']
    htl_ev = params_dict['L3_E_v']
    htl_nd = params_dict['L3_N_D']
    htl_na = params_dict['L3_N_A']
    
    # CONSTRAINT 1: ENERGY BAND ALIGNMENT
    # In SimSalabim: E_CB = -E_c_input, E_VB = -E_v_input
    
    # 1a. Internal consistency: E_CB > E_VB for each layer (positive bandgap)
    for layer, ec, ev in [('ETL', etl_ec, etl_ev), ('Active', active_ec, active_ev), ('HTL', htl_ec, htl_ev)]:
        bandgap = ev - ec  # In SimSalabim: (-E_v) - (-E_c) = E_c - E_v = -(E_v - E_c)
        if bandgap <= 0:
            return False, f"{layer} has non-positive bandgap ({bandgap:.3f} eV)"
        if bandgap < 1.0 or bandgap > 4.0:
            return False, f"{layer} bandgap ({bandgap:.2f} eV) outside realistic range [1.0, 4.0] eV"
    
    # 1b. Electron transport: E_CB_ETL ≤ E_CB_Active (downhill for electrons)
    # In SimSalabim: -etl_ec ≤ -active_ec → etl_ec ≥ active_ec
    if etl_ec < active_ec:
        return False, f"Poor electron alignment: ETL E_c ({etl_ec:.2f}) < Active E_c ({active_ec:.2f}), blocks electrons"
    
    # 1c. Hole transport: E_VB_HTL ≥ E_VB_Active (downhill for holes)
    # In SimSalabim: -htl_ev ≥ -active_ev → htl_ev ≤ active_ev
    if htl_ev > active_ev:
        return False, f"Poor hole alignment: HTL E_v ({htl_ev:.2f}) > Active E_v ({active_ev:.2f}), blocks holes"
    
    # CONSTRAINT 2: DOPING AND CARRIER TYPE
    
    # 2a. ETL (L1) must be n-type: N_D >> N_A
    if etl_nd <= etl_na:
        return False, f"ETL must be n-type: N_D ({etl_nd:.2e}) ≤ N_A ({etl_na:.2e})"
    etl_n_ratio = etl_nd / etl_na
    if etl_n_ratio < 10:  # Strong n-type requirement
        return False, f"ETL not sufficiently n-type: N_D/N_A ratio ({etl_n_ratio:.1f}) < 10"
    
    # 2b. Active Layer (L2) should be intrinsic/undoped: low doping levels
    max_active_doping = 1e18  # Much lower than transport layers
    if active_nd > max_active_doping or active_na > max_active_doping:
        return False, f"Active layer over-doped: N_D ({active_nd:.2e}) or N_A ({active_na:.2e}) > {max_active_doping:.2e}"
    
    # 2c. HTL (L3) must be p-type: N_A >> N_D
    if htl_na <= htl_nd:
        return False, f"HTL must be p-type: N_A ({htl_na:.2e}) ≤ N_D ({htl_nd:.2e})"
    htl_p_ratio = htl_na / htl_nd
    if htl_p_ratio < 10:  # Strong p-type requirement
        return False, f"HTL not sufficiently p-type: N_A/N_D ratio ({htl_p_ratio:.1f}) < 10"
    
    # CONSTRAINT 3: LAYER THICKNESS
    
    # 3a. ETL thickness: 10-50 nm (thin for low resistance)
    etl_thickness_nm = etl_thickness * 1e9
    if etl_thickness_nm < 10 or etl_thickness_nm > 50:
        return False, f"ETL thickness ({etl_thickness_nm:.1f} nm) outside realistic range [10, 50] nm"
    
    # 3b. Active layer thickness: 200-600 nm (thick for light absorption, relaxed minimum)
    active_thickness_nm = active_thickness * 1e9
    if active_thickness_nm < 200 or active_thickness_nm > 600:
        return False, f"Active layer thickness ({active_thickness_nm:.1f} nm) outside realistic range [200, 600] nm"
    
    # 3c. HTL thickness: 10-50 nm (thin for low resistance)
    htl_thickness_nm = htl_thickness * 1e9
    if htl_thickness_nm < 10 or htl_thickness_nm > 50:
        return False, f"HTL thickness ({htl_thickness_nm:.1f} nm) outside realistic range [10, 50] nm"
    
    # ADDITIONAL SAFETY CHECKS
    
    # 4. Reasonable doping concentrations
    max_transport_doping = 1e22  # Maximum for transport layers
    min_doping = 1e16  # Minimum to avoid numerical issues
    
    for layer, nd, na in [('ETL', etl_nd, etl_na), ('HTL', htl_nd, htl_na)]:
        if nd < min_doping or na < min_doping:
            return False, f"{layer} doping too low: N_D ({nd:.2e}) or N_A ({na:.2e}) < {min_doping:.2e}"
        if nd > max_transport_doping or na > max_transport_doping:
            return False, f"{layer} doping too high: N_D ({nd:.2e}) or N_A ({na:.2e}) > {max_transport_doping:.2e}"
    
    # 5. Energy level ranges (realistic for common materials)
    for layer, ec, ev in [('ETL', etl_ec, etl_ev), ('Active', active_ec, active_ev), ('HTL', htl_ec, htl_ev)]:
        if ec < 2.0 or ec > 6.0:
            return False, f"{layer} E_c ({ec:.2f} eV) outside realistic range [2.0, 6.0] eV"
        if ev < 4.0 or ev > 8.0:
            return False, f"{layer} E_v ({ev:.2f} eV) outside realistic range [4.0, 8.0] eV"
    
    # 6. Electrode work function compatibility
    # Fixed electrode work functions from simulation_setup.txt
    W_L = 4.05  # eV, left electrode work function
    W_R = 5.2   # eV, right electrode work function
    
    # Physics constraint: W_L must be >= E_c of leftmost layer (ETL)
    if W_L < etl_ec:
        return False, f"Left electrode work function ({W_L:.2f} eV) < ETL conduction band ({etl_ec:.2f} eV)"
    
    # Physics constraint: W_R must be <= E_v of rightmost layer (HTL)  
    if W_R > htl_ev:
        return False, f"Right electrode work function ({W_R:.2f} eV) > HTL valence band ({htl_ev:.2f} eV)"
    
    # Additional electrode alignment checks for reasonable energy differences
    if W_L - etl_ec > 0.5:  # Too large energy barrier for electron injection
        return False, f"Left electrode barrier too high: W_L - E_c(ETL) = {W_L - etl_ec:.2f} eV > 0.5 eV"
    
    if htl_ev - W_R > 0.5:  # Too large energy barrier for hole injection
        return False, f"Right electrode barrier too high: E_v(HTL) - W_R = {htl_ev - W_R:.2f} eV > 0.5 eV"
    
    return True, "Valid solar cell physics"

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
                    logging.info(f"Invalid combination rejected: {reason}")
        
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
                    logging.info(f"Invalid combination rejected: {reason}")
        
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