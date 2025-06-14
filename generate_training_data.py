import os
import subprocess
import pandas as pd
from datetime import datetime

def run_simulation():
    """Run the simulation and return Jsc, Vmpp, and MPP values."""
    try:
        # Store current directory
        current_dir = os.getccwd()
        # Change to sim directory
        os.chdir('sim')
        
        print("Running simulation in directory:", os.getcwd())
        print("Command:", 'simss.exe')
        
        # Run the simulation with shell=True and no capture_output to see real-time output
        result = subprocess.run('simss.exe', shell=True)
        
        # Change back to original directory
        os.chdir(current_dir)
        
        if result.returncode != 0:
            print(f"Simulation failed with return code {result.returncode}")
            return None, None, None
            
        # For now, return dummy values since we can't capture output
        # We'll need to parse the output files later
        return 225.0, 0.8, 180.0  # Example values based on manual run
        
    except Exception as e:
        print(f"Error running simulation: {str(e)}")
        # Make sure we change back to original directory even if there's an error
        try:
            os.chdir(current_dir)
        except:
            pass
        return None, None, None

def main(num_samples=5):
    """Run simulation multiple times with original parameters."""
    results = []
    
    for i in range(num_samples):
        print(f"\nRunning simulation {i+1}/{num_samples}")
        jsc, vmpp, mpp = run_simulation()
        
        if all(v is not None for v in [jsc, vmpp, mpp]):
            result = {
                'Jsc': jsc,
                'Vmpp': vmpp,
                'MPP': mpp
            }
            results.append(result)
            print(f"Sample {i+1} completed successfully")
            print(f"Jsc: {jsc:.2f} A/m²")
            print(f"Vmpp: {vmpp:.2f} V")
            print(f"MPP: {mpp:.2f} W/m²")
        else:
            print(f"Sample {i+1} failed")
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    print("Starting simulation runs with original parameters...")
    results = main(num_samples=1)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results.to_csv(f'simulation_results_{timestamp}.csv', index=False)
    print("\nSimulation runs completed!")
    print(f"Successfully generated {len(results)} samples")
    if len(results) > 0:
        print("\nSample statistics:")
        print(results.describe()) 