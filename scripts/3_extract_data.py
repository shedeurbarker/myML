import pandas as pd
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
        logging.FileHandler('results/extract/extracted_data.log'),
        logging.StreamHandler()
    ]
)

# Constants
SIM_DIR = Path('results/generate')
var_file = SIM_DIR / 'combined_output.csv'
out_padded = 'results/extract/extracted_data.csv'

# Layer parameter mapping - these are the parameters available in the combined CSV
layer_params = {
    'L': 'L',           # Layer thickness
    'E_c': 'E_c',       # Conduction band energy
    'E_v': 'E_v',       # Valence band energy
    'N_A': 'N_A',       # Acceptor concentration
    'N_D': 'N_D'        # Donor concentration
}

def get_interface_parameters(row: pd.Series, lid: int) -> Dict[str, float]:
    """
    Extract left and right layer parameters for an interface based on layer ID.
    
    Args:
        row: DataFrame row containing layer parameters
        lid: Layer ID (1, 2, or 3)
        
    Returns:
        Dictionary with left and right parameters for the interface
    """
    interface_params = {}
    
    # For interface at layer boundary, left layer is lid-1, right layer is lid
    left_layer = lid - 1
    right_layer = lid
    
    # Extract parameters for left layer (if it exists)
    if left_layer >= 1:
        for param_name, csv_param in layer_params.items():
            csv_col = f'L{left_layer}_{csv_param}'
            if csv_col in row:
                interface_params[f'left_{param_name}'] = row[csv_col]
            else:
                interface_params[f'left_{param_name}'] = 0.0
    else:
        # No left layer (interface at first layer)
        for param_name in layer_params.keys():
            interface_params[f'left_{param_name}'] = 0.0
    
    # Extract parameters for right layer
    for param_name, csv_param in layer_params.items():
        csv_col = f'L{right_layer}_{csv_param}'
        if csv_col in row:
            interface_params[f'right_{param_name}'] = row[csv_col]
        else:
            interface_params[f'right_{param_name}'] = 0.0
    
    return interface_params

def main():
    try:
        # Verify input file exists
        if not var_file.exists():
            raise FileNotFoundError(f"Input file not found: {var_file}")

        # Read the combined CSV data
        logging.info(f"Reading data from {var_file}")
        df = pd.read_csv(var_file, dtype={'lid': 'int32'})

        # Find interface indices (where layer ID changes)
        interface_indices = df.index[df['lid'].diff() != 0].tolist()
        interface_df = df.iloc[interface_indices].copy()

        logging.info(f"Found {len(interface_df)} interface data points")

        # Process interface data
        logging.info("Processing interface data")
        rows = []
        for _, row in interface_df.iterrows():
            lid = int(row['lid'])
            
            # Get interface parameters from the existing layer data in CSV
            interface_params = get_interface_parameters(row, lid)
            
            # Combine original row data with interface parameters
            combined = {**row.to_dict(), **interface_params}
            rows.append(combined)

        # Create DataFrame
        final_df = pd.DataFrame(rows)
        
        # Fill any remaining NaN values with 0
        final_df = final_df.fillna(0)
        
        # Filter out rows where IntSRHp or IntSRHn are exactly zero
        initial_rows = len(final_df)
        final_df = final_df[
            (final_df['IntSRHn'] != 0) |  # Keep any non-zero value
            (final_df['IntSRHp'] != 0)
        ]
        filtered_rows = initial_rows - len(final_df)
        
        # Save the extracted interface data
        logging.info(f"Saving data to {out_padded}")
        final_df.to_csv(out_padded, index=False, float_format='%.6g')
        
        logging.info(f"Successfully extracted {len(final_df)} interface grid points")
        logging.info(f"Filtered out {filtered_rows} rows with zero IntSRHn and IntSRHp values")
        logging.info(f"Output saved to {out_padded}")
        
        # Log some statistics about the extracted data
        if len(final_df) > 0:
            logging.info(f"Interface parameters extracted: {list(layer_params.keys())}")
            logging.info(f"Sample interface data columns: {list(final_df.columns[-10:])}")  # Last 10 columns
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 