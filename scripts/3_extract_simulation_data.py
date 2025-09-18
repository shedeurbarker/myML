"""
===============================================================================
EXTRACT SIMULATION DATA FOR ML - MPP and IntSRHn_mean ONLY
===============================================================================

PURPOSE:
This script extracts device parameters and the two target variables (MPP and IntSRHn_mean) 
from simulation results for ML workflows, based on the definitions in script 1.

WHAT THIS SCRIPT DOES:
1. Reads device parameters from parameters.json
2. Calculates MPP (Maximum Power Point) from the J-V curve in output_JV.dat
3. Extracts IntSRHn_mean (mean electron interfacial recombination rate) from output_Var.dat
4. Writes combined results to a CSV for ML

TARGET VARIABLES (as defined in script 1):
- MPP: Maximum Power Point (W/cm²) - efficiency target
- IntSRHn_mean: Mean electron interfacial recombination rate - recombination target

INPUT FILES:
- output_JV.dat (for MPP calculation)
- output_Var.dat (for IntSRHn_mean extraction)
- parameters.json (for device parameters)

USAGE:
python scripts/3_extract_simulation_data.py

AUTHOR: Anthony Barker
DATE: 2025
"""

import os
import json
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import csv # Added for writing CSV files incrementally

# Set up logging
log_dir = 'results/extract_simulation_data'
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'extraction.log')),
        logging.StreamHandler()
    ]
)

def load_feature_definitions():
    """Load feature definitions from the JSON file."""
    try:
        with open('results/feature_definitions.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error("Feature definitions not found. Run 1_create_feature_names.py first.")
        return None

def extract_jv_curve_data(sim_dir):
    """Extract J-V curve data and calculate MPP (Maximum Power Point) only."""
    jv_file = os.path.join(sim_dir, 'output_JV.dat')
    
    if not os.path.exists(jv_file):
        logging.warning(f"No J-V data found in {sim_dir}")
        return None
    
    try:
        # Read J-V data - skip header row and use first two columns (Vext, Jext)
        jv_data = pd.read_csv(jv_file, sep=r'\s+', header=0, usecols=[0, 1], 
                              names=['V', 'J'], dtype={'V': float, 'J': float})
        
        if len(jv_data) == 0:
            logging.warning(f"Empty J-V data in {sim_dir}")
            return None
        
        # Calculate power: P = V * J
        jv_data['P'] = jv_data['V'] * jv_data['J']
        
        # Find maximum power point (MPP) - this is our only efficiency target
        mpp_idx = jv_data['P'].idxmax()
        mpp_data = jv_data.loc[mpp_idx]
        
        return {
            'MPP': mpp_data['P']
        }
        
    except Exception as e:
        logging.error(f"Error extracting J-V data from {sim_dir}: {str(e)}")
        return None

def extract_recombination_data(sim_dir):
    """Extract IntSRHn_mean (mean electron interfacial recombination rate) only."""
    var_file = os.path.join(sim_dir, 'output_Var.dat')
    
    if not os.path.exists(var_file):
        logging.warning(f"No variable data found in {sim_dir}")
        return None
    
    try:
        # Read variable data (contains recombination rates) - skip header
        var_data = pd.read_csv(var_file, sep=r'\s+', header=0)
        
        if len(var_data) == 0:
            logging.warning(f"Empty variable data in {sim_dir}")
            return None
        
        # Extract only IntSRHn (electron interfacial recombination) - our only recombination target
        if 'IntSRHn' in var_data.columns:
            intsrhn_data = var_data['IntSRHn'].dropna()
            
            if len(intsrhn_data) > 0:
                return {
                    'IntSRHn_mean': intsrhn_data.mean()
                }
        
        logging.warning(f"No IntSRHn data found in {sim_dir}")
        return None
        
    except Exception as e:
        logging.error(f"Error extracting recombination data from {sim_dir}: {str(e)}")
        return None

def extract_device_parameters(sim_dir):
    """Extract device parameters from parameters.json file."""
    param_file = os.path.join(sim_dir, 'parameters.json')
    
    if not os.path.exists(param_file):
        logging.warning(f"No parameters.json found in {sim_dir}")
        return None
    
    try:
        with open(param_file, 'r') as f:
            params = json.load(f)
        
        # Extract only the primary parameters (15 optimization variables)
        feature_defs = load_feature_definitions()
        if feature_defs is None:
            return None
        
        primary_params = feature_defs['primary_parameters']
        extracted_params = {}
        
        for param in primary_params.keys():
            if param in params:
                extracted_params[param] = params[param]
            else:
                logging.warning(f"Parameter {param} not found in {sim_dir}")
                extracted_params[param] = np.nan
        
        return extracted_params
        
    except Exception as e:
        logging.error(f"Error extracting device parameters from {sim_dir}: {str(e)}")
        return None

def process_simulation_directory(sim_dir):
    """Process a single simulation directory and extract all data."""
    logging.info(f"Processing simulation directory: {sim_dir}")
    
    # Extract device parameters
    device_params = extract_device_parameters(sim_dir)
    if device_params is None:
        logging.warning(f"No device parameters found in {sim_dir}")
        return None
    
    # Extract J-V curve data and efficiency metrics
    efficiency_data = extract_jv_curve_data(sim_dir)
    if efficiency_data is None:
        logging.warning(f"No valid J-V data found in {sim_dir} - skipping failed simulation")
        return None  # Skip this simulation entirely
    
    # Extract recombination data
    recombination_data = extract_recombination_data(sim_dir)
    if recombination_data is None:
        recombination_data = {
            'IntSRHn_mean': np.nan
        }
    
    # Combine all data
    combined_data = {**device_params, **efficiency_data, **recombination_data}
    
    # Add simulation metadata
    combined_data['simulation_id'] = os.path.basename(sim_dir)
    combined_data['extraction_timestamp'] = datetime.now().isoformat()
    
    return combined_data

def main():
    """Main function to extract data from all simulation results."""
    logging.info("Starting simulation data extraction...")
    
    # Find all simulation directories
    sim_base_dir = 'sim/simulations'
    if not os.path.exists(sim_base_dir):
        logging.error(f"Simulation directory not found: {sim_base_dir}")
        return
    
    sim_dirs = []
    for item in os.listdir(sim_base_dir):
        item_path = os.path.join(sim_base_dir, item)
        if os.path.isdir(item_path) and item.startswith('sim_'):
            sim_dirs.append(item_path)
    
    logging.info(f"Found {len(sim_dirs)} simulation directories")
    
    # Initialize CSV file with headers
    output_file = os.path.join(log_dir, 'extracted_simulation_data.csv')
    
    # Delete existing files if they exist
    # try:
    #     if os.path.exists(output_file):
    #         os.remove(output_file)
    #         logging.info(f"Deleted existing CSV file: {output_file}")
    # except Exception as e:
    #     logging.warning(f"Could not delete existing CSV file: {e}")
    
    try:
        log_file = os.path.join(log_dir, 'extraction.log')
        if os.path.exists(log_file):
            os.remove(log_file)
            logging.info(f"Deleted existing log file: {log_file}")
    except Exception as e:
        logging.warning(f"Could not delete existing log file: {e}")
    
    # Define column headers based on expected data structure
    headers = [
        # Device parameters (15)
        'L1_L', 'L1_E_c', 'L1_E_v', 'L1_N_D', 'L1_N_A',
        'L2_L', 'L2_E_c', 'L2_E_v', 'L2_N_D', 'L2_N_A', 
        'L3_L', 'L3_E_c', 'L3_E_v', 'L3_N_D', 'L3_N_A',
        # Efficiency metrics (1)
        'MPP',
        # Recombination data (1)
        'IntSRHn_mean',
        # Metadata
        'simulation_id', 'extraction_timestamp'
    ]
    
    # Create CSV file with headers (if not already created)
    write_header = True
    if os.path.exists(output_file):
        with open(output_file, 'r', newline='') as csvfile:
            first_line = csvfile.readline()
            if first_line.strip() == ','.join(headers):
                write_header = False
    if write_header:
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
    
    logging.info(f"Created output file: {output_file} with headers: {headers}")
    
    # Process each simulation directory and write immediately
    successful = 0
    failed = 0
    skipped = 0
    
    for i, sim_dir in enumerate(sim_dirs, 1):
        try:
            logging.info(f"Processing simulation {i}/{len(sim_dirs)}: {sim_dir}")
            
            data = process_simulation_directory(sim_dir)
            if data is not None:
                # Write data immediately to CSV
                with open(output_file, 'a', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=headers)
                    writer.writerow(data)
                
                successful += 1
                logging.info(f"Successfully extracted and wrote data from {sim_dir}")
                
                # Print progress every 10 simulations
                if successful % 10 == 0:
                    logging.info(f"Progress: {successful}/{len(sim_dirs)} completed ({successful/len(sim_dirs)*100:.1f}%)")
            else:
                skipped += 1
                logging.info(f"Skipped failed simulation: {sim_dir}")
        except Exception as e:
            failed += 1
            logging.error(f"Error processing {sim_dir}: {str(e)}")
    
    logging.info(f"Extraction complete!")
    logging.info(f"Successful extractions: {successful}")
    logging.info(f"Skipped failed simulations: {skipped}")
    logging.info(f"Failed extractions: {failed}")
    logging.info(f"Output saved to: {output_file}")
    
    # Print summary statistics
    print("\n=== EXTRACTION SUMMARY ===")
    print(f"Total simulations processed: {len(sim_dirs)}")
    print(f"Successful extractions: {successful}")
    print(f"Skipped failed simulations: {skipped}")
    print(f"Failed extractions: {failed}")
    print(f"Data points extracted: {successful}")
    print(f"Output file: {output_file}")

if __name__ == "__main__":
    main() 