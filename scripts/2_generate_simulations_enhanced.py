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
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'generate_enhanced')

# Configure logging
os.makedirs(RESULTS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(RESULTS_DIR, 'simulation_enhanced.log')),
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

def extract_efficiency_metrics(sim_path):
    """Extract efficiency metrics from simulation output files."""
    efficiency_data = {}
    
    # Extract from output_scPars.dat
    scpars_file = os.path.join(sim_path, 'output_scPars.dat')
    if os.path.exists(scpars_file):
        try:
            with open(scpars_file, 'r') as f:
                lines = f.readlines()
                if len(lines) >= 2:
                    header = lines[0].strip().split()
                    values = lines[1].strip().split()
                    
                    for i, param in enumerate(header):
                        if i < len(values):
                            try:
                                efficiency_data[param] = float(values[i])
                            except ValueError:
                                efficiency_data[param] = 0.0
                                
            logging.info(f"Extracted efficiency metrics: {efficiency_data}")
        except Exception as e:
            logging.error(f"Error extracting efficiency metrics: {e}")
    
    return efficiency_data

def extract_recombination_data(sim_path):
    """Extract IntSRHn recombination data from simulation output."""
    recombination_data = {}
    
    # Extract from output_Var.dat
    var_file = os.path.join(sim_path, 'output_Var.dat')
    if os.path.exists(var_file):
        try:
            with open(var_file, 'r') as f:
                lines = f.readlines()
                
            if len(lines) > 1:
                # Find IntSRHn column
                header = lines[0].strip().split()
                intsrhn_col = None
                for i, col in enumerate(header):
                    if col == 'IntSRHn':
                        intsrhn_col = i
                        break
                
                if intsrhn_col is not None:
                    # Calculate average IntSRHn across all data points
                    intsrhn_values = []
                    for line in lines[1:]:
                        values = line.strip().split()
                        if len(values) > intsrhn_col:
                            try:
                                intsrhn_values.append(float(values[intsrhn_col]))
                            except ValueError:
                                continue
                    
                    if intsrhn_values:
                        recombination_data['IntSRHn_mean'] = np.mean(intsrhn_values)
                        recombination_data['IntSRHn_std'] = np.std(intsrhn_values)
                        recombination_data['IntSRHn_min'] = np.min(intsrhn_values)
                        recombination_data['IntSRHn_max'] = np.max(intsrhn_values)
                        
            logging.info(f"Extracted recombination data: {recombination_data}")
        except Exception as e:
            logging.error(f"Error extracting recombination data: {e}")
    
    return recombination_data

def extract_and_combine_enhanced_data(sim_path, combined_csv_path, is_first_simulation, param_values):
    """Extract both efficiency and recombination data and combine with parameters."""
    # Extract efficiency metrics
    efficiency_metrics = extract_efficiency_metrics(sim_path)
    
    # Extract recombination data
    recombination_data = extract_recombination_data(sim_path)
    
    # Combine all data
    combined_data = {**param_values, **efficiency_metrics, **recombination_data}
    
    # Convert to DataFrame
    df_row = pd.DataFrame([combined_data])
    
    # Save to CSV
    if is_first_simulation:
        df_row.to_csv(combined_csv_path, index=False)
    else:
        df_row.to_csv(combined_csv_path, mode='a', header=False, index=False)
    
    logging.info(f"Combined data saved: {len(combined_data)} parameters")
    return combined_data

def generate_parameter_combinations():
    """Generate parameter combinations for simulations."""
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
        
        # Generate all possible combinations across layers
        for layer_combo in itertools.product(*[layer_combinations[layer] for layer in sorted(layer_params.keys())]):
            params_dict = {}
            for layer, values in zip(sorted(layer_params.keys()), layer_combo):
                for name, value in zip(layer_params[layer]['names'], values):
                    params_dict[name] = value
            combinations.append(params_dict)
    else:
        # If we have more combinations than MAX_COMBINATIONS, randomly sample
        logging.info("Randomly sampling combinations")
        combinations = []
        for _ in range(num_combinations):
            # Randomly select one value from each parameter's range for each layer
            combo = {}
            for layer in sorted(layer_params.keys()):
                for name, values in zip(layer_params[layer]['names'], layer_params[layer]['values']):
                    combo[name] = np.random.choice(values)
            combinations.append(combo)
    
    return combinations

def main():
    """Main function to generate and run enhanced simulations."""
    # Create simulations directory in sim folder
    sim_dir = os.path.join(SIM_DIR, 'simulations')
    os.makedirs(sim_dir, exist_ok=True)
    
    # Create results/generate_enhanced directory
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Create combined CSV file path in results/generate_enhanced
    combined_csv_path = os.path.join(RESULTS_DIR, 'combined_output_with_efficiency.csv')
    
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
                
                # Extract and combine enhanced data
                is_first_simulation = (i == 1)
                combined_data = extract_and_combine_enhanced_data(sim_path, combined_csv_path, is_first_simulation, params)
                
                # Log key metrics
                if 'MPP' in combined_data and 'IntSRHn_mean' in combined_data:
                    logging.info(f"Simulation {i} results:")
                    logging.info(f"  - MPP: {combined_data['MPP']:.2f} W/m²")
                    logging.info(f"  - IntSRHn: {combined_data['IntSRHn_mean']:.2e} A/m²")
                    logging.info(f"  - Jsc: {combined_data.get('Jsc', 0):.2f} A/m²")
                    logging.info(f"  - Voc: {combined_data.get('Voc', 0):.2f} V")
                    logging.info(f"  - FF: {combined_data.get('FF', 0):.2f}")
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
    logging.info(f"Enhanced data saved to: {combined_csv_path}")
    
    # Load and analyze the combined data
    if os.path.exists(combined_csv_path):
        df = pd.read_csv(combined_csv_path)
        logging.info(f"\nData Analysis:")
        logging.info(f"Total data points: {len(df)}")
        logging.info(f"Columns: {list(df.columns)}")
        
        # Analyze efficiency vs recombination
        if 'MPP' in df.columns and 'IntSRHn_mean' in df.columns:
            best_efficiency = df['MPP'].max()
            best_recombination = df.loc[df['MPP'].idxmax(), 'IntSRHn_mean']
            logging.info(f"Best efficiency: {best_efficiency:.2f} W/m²")
            logging.info(f"Corresponding IntSRHn: {best_recombination:.2e} A/m²")

if __name__ == "__main__":
    main() 