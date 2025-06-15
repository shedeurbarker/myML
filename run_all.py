import subprocess
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def run_script(script_name, description):
    """Run a Python script and log its output."""
    logging.info(f"Starting {description}")
    try:
        result = subprocess.run(
            ['python', script_name],
            capture_output=True,
            text=True,
            check=True
        )
        logging.info(f"Successfully completed {description}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running {description}")
        logging.error(f"Error code: {e.returncode}")
        logging.error(f"Error output: {e.stderr}")
        return False

def main():
    # List of scripts to run in sequence
    scripts = [
        ('sim/generate_simulations.py', 'Generate and run simulations'),
        ('scripts/extract_interface_data.py', 'Extract interface data from simulation results'),
        ('scripts/prepare_ml_data.py', 'Prepare data for machine learning'),
        ('scripts/train_ml_models.py', 'Train machine learning models'),
        ('scripts/visualize_model_comparison.py', 'Visualize model comparison results'),
        ('scripts/predict.py', 'Make predictions using trained models')
    ]
    
    # Run each script in sequence
    successful = 0
    for script, description in scripts:
        if run_script(script, description):
            successful += 1
    
    # Print summary
    logging.info("\nScript Execution Summary:")
    logging.info(f"Total scripts: {len(scripts)}")
    logging.info(f"Successful: {successful}")
    logging.info(f"Failed: {len(scripts) - successful}")
    logging.info(f"Success rate: {(successful/len(scripts))*100:.2f}%")

if __name__ == "__main__":
    main() 