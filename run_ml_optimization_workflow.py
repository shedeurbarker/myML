"""
ML and Optimization Workflow Script
==================================

This script runs the ML and optimization portion of the solar cell optimization pipeline
(scripts 4-8). It assumes that the data preparation steps (scripts 1-3) have already
been completed.

WORKFLOW:
---------
1. Script 4: Prepare ML data (derived features, data cleaning, train/test splits)
2. Script 5: Train optimization models (efficiency and recombination prediction)
3. Script 6: Run optimization (find optimal device parameters)
4. Script 7: Make predictions (validate models and make new predictions)
5. Script 8: Visualize results (create comprehensive dashboard)

PREREQUISITES:
--------------
- Script 1: Define feature structure (results/feature_definitions.json)
- Script 2: Generate simulations (sim/simulations/)
- Script 3: Extract simulation data (results/extract_simulation_data/)

OUTPUT:
-------
- ML-ready datasets: results/prepare_ml_data/
- Trained models: results/train_optimization_models/models/
- Optimization results: results/optimize_efficiency/
- Predictions: results/predict/
- Visualizations: results/visualize/

USAGE:
------
python run_ml_optimization_workflow.py
"""

import subprocess
import sys
import os
import shutil
from datetime import datetime
import json

def check_prerequisites():
    """Check if required files exist before running the workflow."""
    print("🔍 Checking prerequisites...")
    
    required_files = [
        'results/feature_definitions.json',
        'results/extract_simulation_data/combined_output_with_efficiency.csv'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\nPlease run scripts 1-3 first:")
        print("   python scripts/1_create_feature_names.py")
        print("   python scripts/2_generate_simulations_enhanced.py")
        print("   python scripts/3_extract_simulation_data.py")
        return False
    
    print("✅ All prerequisites satisfied!")
    return True

def cleanup_previous_results():
    """Delete previous ML and optimization results before starting new run."""
    print(f"\n{'='*50}")
    print("Cleaning up previous ML and optimization results...")
    print(f"{'='*50}\n")
    
    # Folders to delete
    folders_to_delete = [
        'results/prepare_ml_data',
        'results/train_optimization_models',
        'results/optimize_efficiency',
        'results/predict',
        'results/visualize'
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

def run_script(script_name, step_number, total_steps):
    """Run a Python script and display its output in real-time."""
    print(f"\n{'='*60}")
    print(f"Step {step_number}/{total_steps}: Running {script_name}")
    print(f"{'='*60}\n")
    
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
        print(f"\n❌ Error: {script_name} failed with return code {process.returncode}")
        return False
    
    print(f"\n✅ {script_name} completed successfully!")
    return True

def create_workflow_summary():
    """Create a summary of the workflow results."""
    print(f"\n{'='*60}")
    print("WORKFLOW SUMMARY")
    print(f"{'='*60}")
    
    # Check what was created
    output_dirs = [
        ('ML Data Preparation', 'results/prepare_ml_data'),
        ('Trained Models', 'results/train_optimization_models/models'),
        ('Optimization Results', 'results/optimize_efficiency'),
        ('Predictions', 'results/predict'),
        ('Visualizations', 'results/visualize')
    ]
    
    for name, path in output_dirs:
        if os.path.exists(path):
            if os.path.isdir(path):
                files = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
                print(f"✅ {name}: {path} ({files} files)")
            else:
                print(f"✅ {name}: {path}")
        else:
            print(f"❌ {name}: {path} (not found)")
    
    # Check for key files
    key_files = [
        ('Feature Definitions', 'results/feature_definitions.json'),
        ('Optimization Report', 'results/optimize_efficiency/reports/optimization_report.json'),
        ('Dashboard', 'results/visualize/comprehensive_dashboard.png')
    ]
    
    print(f"\nKey Files:")
    for name, path in key_files:
        if os.path.exists(path):
            print(f"✅ {name}: {path}")
        else:
            print(f"❌ {name}: {path} (not found)")

def main():
    """Main function to run the ML and optimization workflow."""
    print("🚀 Starting ML and Optimization Workflow")
    print("=" * 60)
    print("This workflow will:")
    print("1. Prepare ML data (derived features, cleaning, splits)")
    print("2. Train optimization models (efficiency & recombination)")
    print("3. Run optimization (find optimal device parameters)")
    print("4. Make predictions (validate and predict)")
    print("5. Visualize results (comprehensive dashboard)")
    print("=" * 60)
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites not met. Exiting.")
        sys.exit(1)
    
    # Ask user if they want to clean up previous results
    cleanup = input("\nDo you want to clean up previous ML and optimization results? (y/n): ").lower().strip()
    if cleanup == 'y':
        cleanup_previous_results()
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Get current timestamp for log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'logs/ml_optimization_workflow_{timestamp}.log'
    
    # Start logging to file
    with open(log_file, 'w') as f:
        f.write(f"Starting ML and optimization workflow at {timestamp}\n")
    
    # List of scripts to run in order
    scripts = [
        'scripts/4_prepare_ml_data.py',
        'scripts/5_train_optimization_models.py',
        'scripts/6_optimize_efficiency.py',
        'scripts/7_predict.py',
        'scripts/8_visualize_example_fixed.py'
    ]
    
    # Run each script
    successful_steps = 0
    for i, script in enumerate(scripts, 1):
        print(f"\n📋 Step {i}/{len(scripts)}: {script}")
        try:
            if run_script(script, i, len(scripts)):
                successful_steps += 1
            else:
                print(f"\n❌ Workflow failed at step {i}. Stopping.")
                break
        except Exception as e:
            print(f"\n❌ Error running {script}: {e}")
            break
    
    # Create workflow summary
    create_workflow_summary()
    
    # Final status
    if successful_steps == len(scripts):
        print(f"\n🎉 SUCCESS: All {len(scripts)} steps completed successfully!")
        print(f"📊 Results available in:")
        print(f"   - ML Data: results/prepare_ml_data/")
        print(f"   - Models: results/train_optimization_models/models/")
        print(f"   - Optimization: results/optimize_efficiency/")
        print(f"   - Predictions: results/predict/")
        print(f"   - Visualizations: results/visualize/")
    else:
        print(f"\n⚠️  PARTIAL SUCCESS: {successful_steps}/{len(scripts)} steps completed.")
        print("Check the logs above for details on what failed.")
    
    print(f"\n📝 Log file: {log_file}")

if __name__ == "__main__":
    main() 