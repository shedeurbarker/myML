import pandas as pd
import re
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

# Create result directory if it doesn't exist
Path('results/extract').mkdir(parents=True, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('results/extract/interface_data.log'),
        logging.StreamHandler()
    ]
)

# Constants
SIM_DIR = Path('sim/simulations')
var_file = SIM_DIR / 'combined_output.csv'
out_padded = 'results/extract/interface_data.csv'

layer_files = {
    1: Path('sim/L1_parameters.txt'),
    2: Path('sim/L2_parameters.txt'),
    3: Path('sim/L3_parameters.txt')
}

# List of parameters to extract from each layer
all_params = [
    'L', 'eps_r', 'E_c', 'E_v', 'N_c', 'mu_n', 'mu_p',
    'N_t_int', 'C_n_int', 'C_p_int', 'E_t_int'
]

def extract_params(filename: Optional[Path]) -> Dict[str, Any]:
    """
    Extract parameters from a layer parameter file.
    
    Args:
        filename: Path to the parameter file
        
    Returns:
        Dictionary containing extracted parameters
    """
    params = {k: 0.0 for k in all_params}  # Initialize with 0.0 instead of None
    if filename is None or not filename.exists():
        logging.warning(f"Parameter file not found: {filename}")
        return params
        
    try:
        with open(filename) as f:
            for line in f:
                for key in all_params:
                    if re.match(rf'\s*{key}\s*=', line):
                        val = line.split('=')[1].split()[0]
                        try:
                            params[key] = float(val)
                        except ValueError:
                            params[key] = 0.0  # Use 0.0 instead of the string value
                            logging.warning(f"Could not convert {key}={val} to float in {filename}, using 0.0 instead")
    except Exception as e:
        logging.error(f"Error reading {filename}: {str(e)}")
    
    return params

def main():
    try:
        # Verify input file exists
        if not var_file.exists():
            raise FileNotFoundError(f"Input file not found: {var_file}")

        # Read the data with optimized settings
        logging.info(f"Reading data from {var_file}")
        df = pd.read_csv(
            var_file,
            sep=r'\s+',  # Using regex pattern for whitespace instead of delim_whitespace
            on_bad_lines='skip',
            dtype={'lid': 'int32'}  # Optimize memory usage
        )

        # Find interface indices more efficiently
        interface_indices = df.index[df['lid'].diff() != 0].tolist()
        interface_df = df.iloc[interface_indices].copy()

        # Build padded rows using list comprehension for better performance
        logging.info("Processing interface data")
        rows = []
        for _, row in interface_df.iterrows():
            lid = int(row['lid'])
            left_params = extract_params(layer_files.get(lid - 1)) if lid > 1 else {k: 0.0 for k in all_params}
            right_params = extract_params(layer_files.get(lid))
            
            # Use dictionary comprehension for better performance
            left_params = {f'left_{k}': left_params.get(k, 0.0) for k in all_params}
            right_params = {f'right_{k}': right_params.get(k, 0.0) for k in all_params}
            
            combined = {**row.to_dict(), **left_params, **right_params}
            rows.append(combined)

        # Create DataFrame more efficiently
        final_df = pd.DataFrame(rows)
        
        # Fill any remaining NaN values with 0
        final_df = final_df.fillna(0)
        
        # Save with optimized settings
        logging.info(f"Saving data to {out_padded}")
        final_df.to_csv(out_padded, index=False, float_format='%.6g')
        
        logging.info(f"Successfully extracted {len(rows)} interface grid points")
        logging.info(f"Output saved to {out_padded}")
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 