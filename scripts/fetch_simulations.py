import os
import json
import logging

# Define base paths
SCRIPT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SIM_DIR = os.path.join(PROJECT_ROOT, 'sim', 'simulations')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'fetch')

# Configure logging
os.makedirs(RESULTS_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(RESULTS_DIR, 'fetch.log')),
        logging.StreamHandler()
    ]
)

def extract_and_combine_data(sim_path, combined_csv_path, is_first_simulation, param_values):
    """Extract data from output_Var.dat and append to combined CSV file with layer parameters."""
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
        header_line = lines[0].strip()
        data_lines = lines[1:]

        modified_lines = []
        for line in data_lines:
            line = line.strip()
            if line:
                values = line.split()
                csv_line = ','.join(values)
                param_values_list = [f"{param_values.get(param, 'N/A')}" for param in sorted(param_values.keys())]
                param_str = ','.join(param_values_list)
                modified_line = f"{csv_line},{param_str}\n"
                modified_lines.append(modified_line)

        if not file_exists:
            header_values = header_line.split()
            csv_header = ','.join(header_values)
            param_names = ','.join(sorted(param_values.keys()))
            new_header = f"{csv_header},{param_names}\n"
            with open(combined_csv_path, 'w') as f:
                f.write(new_header)
                f.writelines(modified_lines)
        else:
            with open(combined_csv_path, 'a') as f:
                f.writelines(modified_lines)

        logging.info(f"Combined data from {sim_path} with parameters.")
    except Exception as e:
        logging.error(f"Error combining data from {sim_path}: {str(e)}")

def main():
    combined_csv_path = os.path.join(RESULTS_DIR, 'combined_output.csv')
    sim_folders = [os.path.join(SIM_DIR, d) for d in os.listdir(SIM_DIR) if os.path.isdir(os.path.join(SIM_DIR, d))]
    sim_folders.sort()
    logging.info(f"Found {len(sim_folders)} simulation folders.")

    first = True
    for sim_path in sim_folders:
        param_file = os.path.join(sim_path, 'parameters.json')
        if not os.path.exists(param_file):
            logging.warning(f"No parameters.json found in {sim_path}")
            continue
        try:
            with open(param_file, 'r') as f:
                param_values = json.load(f)
        except Exception as e:
            logging.error(f"Could not read parameters.json in {sim_path}: {str(e)}")
            continue
        extract_and_combine_data(sim_path, combined_csv_path, first, param_values)
        if first:
            first = False
    logging.info(f"Combined output saved to {combined_csv_path}")

if __name__ == "__main__":
    main() 