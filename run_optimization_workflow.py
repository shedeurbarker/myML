import subprocess
import sys
import os
import shutil
from datetime import datetime

def cleanup_previous_results():
    """Delete previous results and simulation folders before starting new run."""
    print(f"\n{'='*50}")
    print("Cleaning up previous results...")
    print(f"{'='*50}\n")
    
    # Folders to delete
    folders_to_delete = [
        'results/generate_enhanced',
        'results/train_optimization_models',
        'results/optimize_efficiency',
        'sim/simulations'
    ]
    
    for folder in folders_to_delete:
        if os.path.exists(folder):
            try:
                shutil.rmtree(folder)
                print(f"✓ Deleted folder: {folder}")
            except Exception as e:
                print(f"✗ Error deleting {folder}: {e}")
        else:
            print(f"- Folder does not exist: {folder}")
    
    print("\nCleanup completed!\n")

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
    """Main function to run the complete optimization workflow."""
    print("🚀 Starting Solar Cell Optimization Workflow")
    print("=" * 60)
    print("This workflow will:")
    print("1. [SKIPPED] Generate enhanced simulations with efficiency metrics")
    print("2. Train multi-target ML models for optimization")
    print("3. Run optimization to find optimal recombination rates")
    print("4. Generate comprehensive reports and visualizations")
    print("=" * 60)
    
    # Ask user if they want to clean up previous results
    cleanup = input("\nDo you want to clean up previous results? (y/n): ").lower().strip()
    if cleanup == 'y':
        cleanup_previous_results()
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Get current timestamp for log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'logs/optimization_workflow_{timestamp}.log'
    
    # Start logging to file
    with open(log_file, 'w') as f:
        f.write(f"Starting optimization workflow at {timestamp}\n")
    
    # List of scripts to run in order
    scripts = [
        #'scripts/2_generate_simulations_enhanced.py',
        'scripts/5_train_optimization_models.py',
        'scripts/6_optimize_efficiency.py'
    ]
    
    # Run each script
    for i, script in enumerate(scripts, 1):
        print(f"\n📋 Step {i}/{len(scripts)}: {script}")
        try:
            run_script(script)
            # Log success to file
            with open(log_file, 'a') as f:
                f.write(f"{script} completed successfully\n")
        except Exception as e:
            # Log error to file
            with open(log_file, 'a') as f:
                f.write(f"Error running {script}: {str(e)}\n")
            print(f"\n❌ Error running {script}: {str(e)}")
            print("Please check the logs for more details.")
            return
    
    # Create summary report
    print("\n" + "="*60)
    print("🎉 OPTIMIZATION WORKFLOW COMPLETE!")
    print("="*60)
    
    # Check for results
    results_files = [
        'results/generate_enhanced/combined_output_with_efficiency.csv',
        'results/train_optimization_models/models/metadata.json',
        'results/optimize_efficiency/reports/optimization_report.json'
    ]
    
    print("\n📊 Results Summary:")
    for file_path in results_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} (not found)")
    
    print(f"\n📁 Output Directories:")
    output_dirs = [
        'results/generate_enhanced/',
        'results/train_optimization_models/',
        'results/optimize_efficiency/'
    ]
    
    for dir_path in output_dirs:
        if os.path.exists(dir_path):
            print(f"📂 {dir_path}")
        else:
            print(f"❌ {dir_path} (not found)")
    
    print(f"\n📋 Next Steps:")
    print("1. Check the optimization report: results/optimize_efficiency/reports/optimization_report.json")
    print("2. View optimization plots: results/optimize_efficiency/plots/")
    print("3. Review training results: results/train_optimization_models/plots/")
    print("4. Analyze enhanced simulation data: results/generate_enhanced/combined_output_with_efficiency.csv")
    
    print(f"\n📝 Log file: {log_file}")
    print("="*60)

if __name__ == "__main__":
    main() 