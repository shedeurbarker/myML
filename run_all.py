import subprocess
import sys
import os
from datetime import datetime

def run_script(script_name):
    """Run a Python script and display its output in real-time."""
    print(f"\n{'='*50}")
    print(f"Running {script_name}...")
    print(f"{'='*50}\n")
    
    # Run the script and capture output in real-time
    process = subprocess.Popen(
        [sys.executable, script_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # Print output in real-time
    for line in process.stdout:
        print(line, end='')
    
    # Wait for the process to complete
    process.wait()
    
    # Check if the script ran successfully
    if process.returncode != 0:
        print(f"\nError: {script_name} failed with return code {process.returncode}")
        sys.exit(1)
    
    print(f"\n{script_name} completed successfully!\n")

def main():
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Get current timestamp for log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'logs/run_all_{timestamp}.log'
    
    # Start logging to file
    with open(log_file, 'w') as f:
        f.write(f"Starting run_all.py at {timestamp}\n")
    
    # List of scripts to run in order
    scripts = [
        'scripts/create_feature_names.py',
#        'sim/generate_simulations.py',
        'scripts/extract_data.py',
        'scripts/prepare_data.py',
        'scripts/train_models.py',
        'scripts/create_example_data.py',
        'scripts/predict.py',
        'scripts/visualize_example.py'
    ]
    
    # Run each script
    for script in scripts:
        try:
            run_script(script)
            # Log success to file
            with open(log_file, 'a') as f:
                f.write(f"{script} completed successfully\n")
        except Exception as e:
            # Log error to file
            with open(log_file, 'a') as f:
                f.write(f"Error running {script}: {str(e)}\n")
            raise
    
    print(f"\nAll scripts completed! Log file: {log_file}")

if __name__ == "__main__":
    main() 