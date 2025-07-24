import subprocess
import sys
import os
import shutil
from datetime import datetime

# def cleanup_previous_results():
#     """Delete previous results and simulation folders before starting new run."""
#     print(f"\n{'='*50}")
#     print("Cleaning up previous results...")
#     print(f"{'='*50}\n")
    
    # Folders to delete
    # folders_to_delete = [
    #     'results',
    #     'sim/simulations'
    # ]
    
    # for folder in folders_to_delete:
    #     if os.path.exists(folder):
    #         try:
    #             shutil.rmtree(folder)
    #             print(f"✓ Deleted folder: {folder}")
    #         except Exception as e:
    #             print(f"✗ Error deleting {folder}: {e}")
    #     else:
    #         print(f"- Folder does not exist: {folder}")
    
    # print("\nCleanup completed!\n")

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
    # Clean up previous results before starting
    # cleanup_previous_results()
    
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
        'scripts/1_create_feature_names.py',
        #'scripts/2_generate_simulations.py',
        #'scripts/3_extract_data.py',
        'scripts/4_prepare_data.py',
        'scripts/5_train_models.py',
        'scripts/7_predict.py',
        'scripts/8_visualize_example.py'
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