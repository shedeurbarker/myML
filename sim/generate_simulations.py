import numpy as np
import os
import shutil
import logging
from datetime import datetime
import itertools
import subprocess
import json

# Configure logging
results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'generate')
os.makedirs(results_dir, exist_ok=True)  # Ensure the results/generate directory exists

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(results_dir, 'simulation.log')),
        logging.StreamHandler()
    ]
)

# Maximum number of parameter combinations to generate
MAX_COMBINATIONS = 100

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
        logging.info(f"Values: {values}")
    
    # Calculate total possible combinations per layer
    total_possible = 1
    for layer in layer_params:
        layer_total = 1
        for values in layer_params[layer]['values']:
            layer_total *= len(values)
        total_possible *= layer_total
        logging.info(f"Layer {layer} total combinations: {layer_total}")    
    
    logging.info(f"Total possible combinations: {total_possible}")
    
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
    base_dir = os.path.join(os.path.dirname(__file__), 'simulations')
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
    files_to_copy = ['simss.exe', 'simulation_setup.txt', 'L1_parameters.txt', 'L2_parameters.txt', 'L3_parameters.txt']
    for file in files_to_copy:
        source_path = os.path.join(os.path.dirname(__file__), file)
        dest_path = os.path.join(sim_dir, file)
        try:
            shutil.copy2(source_path, dest_path)
            logging.info(f"Copied {file} to {sim_dir}")
        except Exception as e:
            logging.error(f"Failed to copy {file}: {e}")
    data_files = [
        'nk_SiO2.txt', 'nk_ITO.txt', 'nk_PEDOT.txt', 'nk_Au.txt',
        'AM15G.txt', 'nk_PCBM.txt', 'nk_MAPI.txt'
    ]
    for file in data_files:
        source_path = os.path.join(os.path.dirname(__file__), 'Data', file)
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
        # Change to simulation directory
        os.chdir(sim_dir)
        
        # Run simulation
        result = subprocess.run(['./simss.exe'], 
                              capture_output=True, 
                              text=True, 
                              check=False)
        
        # Change back to original directory
        os.chdir(os.path.dirname(os.path.dirname(__file__)))
        
        # Consider return code 0, empty return code, and return code 95 (non-convergence) as success
        if result.returncode in [0, 95] or result.returncode is None:
            return result
        else:
            raise subprocess.CalledProcessError(result.returncode, './simss.exe', result.stdout, result.stderr)
    except Exception as e:
        # Change back to original directory in case of error
        os.chdir(os.path.dirname(os.path.dirname(__file__)))
        raise e

def generate_parameter_combinations():
    """Generate parameter combinations for simulations."""
    # Parse parameters from file
    params = parse_parameters(os.path.join(os.path.dirname(__file__), 'parameters.txt'))
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
            # Log first few combinations for debugging
            for i, combo in enumerate(layer_combinations[layer][:3]):
                logging.info(f"Layer {layer} combination {i}: {dict(zip(layer_params[layer]['names'], combo))}")
        
        # Generate all possible combinations across layers
        for layer_combo in itertools.product(*[layer_combinations[layer] for layer in sorted(layer_params.keys())]):
            params_dict = {}
            for layer, values in zip(sorted(layer_params.keys()), layer_combo):
                for name, value in zip(layer_params[layer]['names'], values):
                    params_dict[name] = value
            combinations.append(params_dict)
            # Log first few complete combinations for debugging
            if len(combinations) <= 3:
                logging.info(f"Complete combination {len(combinations)}: {params_dict}")
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
            # Log first few random combinations for debugging
            if len(combinations) <= 3:
                logging.info(f"Random combination {len(combinations)}: {combo}")
    
    return combinations

def extract_and_combine_data(sim_path, combined_csv_path, is_first_simulation):
    """Extract data from output_Var.dat and append to combined CSV file."""
    var_file = os.path.join(sim_path, 'output_Var.dat')
    if not os.path.exists(var_file):
        logging.warning(f"No output_Var.dat found in {sim_path}")
        return

    try:
        with open(var_file, 'r') as f:
            lines = f.readlines()

        if not lines:
            logging.warning(f"Empty output_Var.dat in {sim_path}")
            return

        file_exists = os.path.exists(combined_csv_path)
        # Only write header if file does not exist, always skip header when appending
        if file_exists:
            data_lines = lines[1:]
            mode = 'a'
        else:
            data_lines = lines
            mode = 'w'

        with open(combined_csv_path, mode) as f:
            f.writelines(data_lines)

        logging.info(f"Successfully combined data from {sim_path}")
    except Exception as e:
        logging.error(f"Error combining data from {sim_path}: {str(e)}")

def main():
    """Main function to generate and run simulations."""
    # Create simulations directory
    sim_dir = os.path.join(os.path.dirname(__file__), 'simulations')
    os.makedirs(sim_dir, exist_ok=True)
    
    # Create results/generate directory
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results', 'generate')
    os.makedirs(results_dir, exist_ok=True)
    
    # Create combined CSV file path in results/generate
    combined_csv_path = os.path.join(results_dir, 'combined_output.csv')
    
   
    # Generate parameter combinations
    param_combinations = generate_parameter_combinations()
    total_sims = len(param_combinations)
    
    logging.info(f"Starting generation of {total_sims} simulations")
    
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
                logging.info(f"Output: {result.stdout}")
                
                # Extract and combine data
                is_first_simulation = (i == 1)
                extract_and_combine_data(sim_path, combined_csv_path, is_first_simulation)
            else:
                failed += 1
                logging.error(f"Simulation {i} failed with return code {result.returncode}")
                logging.error(f"Error: {result.stderr}")
        except Exception as e:
            failed += 1
            logging.error(f"Error running simulation {i}: {str(e)}")
    
    # Print summary
    logging.info("\nSimulation Summary:")
    logging.info(f"Total simulations: {total_sims}")
    logging.info(f"Successful: {successful}")
    logging.info(f"Failed: {failed}")
    logging.info(f"Success rate: {(successful/total_sims)*100:.2f}%")
    logging.info(f"Combined data saved to: {combined_csv_path}")

if __name__ == "__main__":
    main() 