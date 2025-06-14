import numpy as np
import itertools
import subprocess
import os
import json
from datetime import datetime

# Define base paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SIM_DIR = os.path.join(SCRIPT_DIR, 'sim')

def create_parameter_combinations():
    """Create systematic combinations of parameters"""
    # Interface parameters
    interface_params = {
        'N_t_int': [1E11, 1E12, 1E13, 1E14],  # Interface trap density (m^-2)
        'E_t_int': [4.0, 4.3, 4.7, 5.0],      # Trap energy level (eV)
        'C_n_int': [1E-15, 1E-14, 1E-13],     # Electron capture coefficient (m^3/s)
        'C_p_int': [1E-15, 1E-14, 1E-13],     # Hole capture coefficient (m^3/s)
        'nu_int_n': [1E2, 1E3, 1E4],          # Interface transfer velocity (m/s)
        'nu_int_p': [1E2, 1E3, 1E4]           # Interface transfer velocity (m/s)
    }
    
    # Bulk parameters
    bulk_params = {
        'N_t_bulk': [1E18, 1E19, 1E20],       # Bulk trap density (m^-3)
        'E_t_bulk': [4.0, 4.3, 4.7, 5.0],     # Bulk trap energy (eV)
        'mu_n': [1E-7, 1E-6, 1E-5],           # Electron mobility (m^2/Vs)
        'mu_p': [1E-7, 1E-6, 1E-5]            # Hole mobility (m^2/Vs)
    }
    
    # Generate combinations
    interface_combinations = list(itertools.product(*interface_params.values()))
    bulk_combinations = list(itertools.product(*bulk_params.values()))
    
    return interface_combinations, bulk_combinations, interface_params.keys(), bulk_params.keys()

def update_layer_parameters(layer_file, params, param_names):
    """Update layer parameter file with new values"""
    file_path = os.path.join(SIM_DIR, layer_file)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Layer parameter file not found: {file_path}")
        
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        for param_name, param_value in zip(param_names, params):
            if param_name in line and '*' in line:
                # Update the parameter value
                parts = line.split('*')
                parts[0] = f"{param_value:10.3e}"
                lines[i] = '*'.join(parts)
    
    with open(file_path, 'w') as f:
        f.writelines(lines)

def run_simulation():
    """Run the simulation and collect results"""
    try:
        # Change to sim directory and run simulation
        original_dir = os.getcwd()
        os.chdir(SIM_DIR)
        subprocess.run(['simss.exe'], check=True)
        os.chdir(original_dir)
        return True
    except subprocess.CalledProcessError:
        print("Simulation failed to run")
        return False
    except Exception as e:
        print(f"Error running simulation: {str(e)}")
        return False

def extract_interface_data(var_file, interface_positions):
    """Extract interface data from var file"""
    # This is a placeholder - you'll need to implement the actual data extraction
    # based on your var file format
    data = {
        'position': [],
        'electric_field': [],
        'carrier_densities': [],
        'quasi_fermi_levels': [],
        'interface_currents': []
    }
    return data

def main():
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(SCRIPT_DIR, f"training_data_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Get parameter combinations
    interface_combinations, bulk_combinations, interface_param_names, bulk_param_names = create_parameter_combinations()
    
    # Save parameter combinations
    with open(os.path.join(output_dir, "parameter_combinations.json"), 'w') as f:
        json.dump({
            'interface_params': list(interface_param_names),
            'bulk_params': list(bulk_param_names),
            'total_combinations': len(interface_combinations) * len(bulk_combinations)
        }, f, indent=4)
    
    # Interface positions (in meters)
    interface_positions = {
        'pcbm_mapi': 25e-9,    # PCBM/MAPI interface
        'mapi_pedot': 525e-9   # MAPI/PEDOT interface
    }
    
    # Run simulations for each combination
    total_simulations = len(interface_combinations) * len(bulk_combinations)
    completed = 0
    
    for interface_params in interface_combinations:
        for bulk_params in bulk_combinations:
            try:
                # Update layer parameters
                update_layer_parameters('L1_parameters.txt', interface_params, interface_param_names)
                update_layer_parameters('L2_parameters.txt', bulk_params, bulk_param_names)
                update_layer_parameters('L3_parameters.txt', interface_params, interface_param_names)
                
                # Run simulation
                if run_simulation():
                    # Extract and save data
                    interface_data = extract_interface_data('output_Var.dat', interface_positions)
                    
                    # Save data with parameter combination identifier
                    data_file = os.path.join(output_dir, f"sim_{completed:06d}.npz")
                    np.savez(data_file,
                            interface_params=interface_params,
                            bulk_params=bulk_params,
                            interface_data=interface_data)
                
                completed += 1
                print(f"Progress: {completed}/{total_simulations} simulations completed")
                
            except Exception as e:
                print(f"Error in simulation {completed}: {str(e)}")
                continue

if __name__ == "__main__":
    main() 