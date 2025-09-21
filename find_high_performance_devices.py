#!/usr/bin/env python3
"""
Find High-Performance Device Parameters from Training Data

This script analyzes the training data to find device parameter combinations
that yield the highest MPP/PCE values (> 25%).
"""

import pandas as pd
import numpy as np
import json

def find_top_devices():
    """Find the top performing devices from training data."""
    print("=== ANALYZING TRAINING DATA FOR HIGH-PERFORMANCE DEVICES ===")
    
    # Load training data
    X = pd.read_csv('results/4_prepare_ml_data/X_full.csv')
    y = pd.read_csv('results/4_prepare_ml_data/y_efficiency_full.csv')
    
    print(f"Total devices in training data: {len(y)}")
    print(f"MPP statistics:")
    print(f"  Min: {y['MPP'].min():.1f} W/cm²")
    print(f"  Max: {y['MPP'].max():.1f} W/cm²")
    print(f"  Mean: {y['MPP'].mean():.1f} W/cm²")
    print(f"  Median: {y['MPP'].median():.1f} W/cm²")
    print()
    
    # Find devices with MPP > 300 (assuming these could achieve 25%+ PCE)
    high_performance_mask = y['MPP'] > 300
    high_perf_devices = y[high_performance_mask]
    
    print(f"Devices with MPP > 300 W/cm²: {len(high_perf_devices)}")
    if len(high_perf_devices) > 0:
        print(f"Range: {high_perf_devices['MPP'].min():.1f} - {high_perf_devices['MPP'].max():.1f} W/cm²")
        print()
    
    # Get top 5 devices
    top_5_indices = y['MPP'].nlargest(5).index
    
    print("=== TOP 5 HIGHEST PERFORMANCE DEVICES ===")
    top_devices = []
    
    for i, idx in enumerate(top_5_indices):
        mpp = y.loc[idx, 'MPP']
        params = X.loc[idx]
        
        print(f"--- DEVICE {i+1} (MPP: {mpp:.1f} W/cm²) ---")
        
        # Primary parameters
        device_params = {
            "L1_L": float(params["L1_L"]),
            "L1_E_c": float(params["L1_E_c"]),
            "L1_E_v": float(params["L1_E_v"]),
            "L1_N_D": float(params["L1_N_D"]),
            "L1_N_A": float(params["L1_N_A"]),
            "L2_L": float(params["L2_L"]),
            "L2_E_c": float(params["L2_E_c"]),
            "L2_E_v": float(params["L2_E_v"]),
            "L2_N_D": float(params["L2_N_D"]),
            "L2_N_A": float(params["L2_N_A"]),
            "L3_L": float(params["L3_L"]),
            "L3_E_c": float(params["L3_E_c"]),
            "L3_E_v": float(params["L3_E_v"]),
            "L3_N_D": float(params["L3_N_D"]),
            "L3_N_A": float(params["L3_N_A"])
        }
        
        # Display in readable format
        print(f"Thickness: L1={params['L1_L']*1e9:.1f}nm, L2={params['L2_L']*1e9:.1f}nm, L3={params['L3_L']*1e9:.1f}nm")
        print(f"Energy L1: E_c={params['L1_E_c']:.3f}eV, E_v={params['L1_E_v']:.3f}eV")
        print(f"Energy L2: E_c={params['L2_E_c']:.3f}eV, E_v={params['L2_E_v']:.3f}eV")
        print(f"Energy L3: E_c={params['L3_E_c']:.3f}eV, E_v={params['L3_E_v']:.3f}eV")
        print(f"Doping L1: N_D={params['L1_N_D']:.2e}, N_A={params['L1_N_A']:.2e}")
        print(f"Doping L2: N_D={params['L2_N_D']:.2e}, N_A={params['L2_N_A']:.2e}")
        print(f"Doping L3: N_D={params['L3_N_D']:.2e}, N_A={params['L3_N_A']:.2e}")
        print()
        
        top_devices.append({
            'mpp': mpp,
            'parameters': device_params
        })
    
    return top_devices

def create_high_performance_examples(top_devices):
    """Create example device parameter files for high-performance devices."""
    print("=== CREATING HIGH-PERFORMANCE EXAMPLE FILES ===")
    
    for i, device in enumerate(top_devices[:3]):  # Top 3 devices
        filename = f"high_performance_device_{i+1}.json"
        
        device_config = {
            "description": f"High-performance solar cell device parameters (MPP: {device['mpp']:.1f} W/cm²)",
            "device_type": "Optimized Perovskite Solar Cell",
            "expected_performance": {
                "MPP_W_per_cm2": device['mpp'],
                "estimated_PCE_percent": device['mpp']  # Assuming PCE ≈ MPP in simulation units
            },
            "last_updated": "2025-09-21",
            "parameters": device['parameters'],
            "layer_descriptions": {
                "L1": "ETL - Electron Transport Layer",
                "L2": "Active - Perovskite Absorber Layer", 
                "L3": "HTL - Hole Transport Layer"
            },
            "parameter_descriptions": {
                "L_L": "Layer thickness (m)",
                "E_c": "Conduction band energy (eV)",
                "E_v": "Valence band energy (eV)",
                "N_D": "Donor concentration (cm^-3)",
                "N_A": "Acceptor concentration (cm^-3)"
            },
            "physics_notes": {
                "energy_alignment": "ETL E_c >= Active E_c, Active E_v >= HTL E_v",
                "electrode_compatibility": "W_L=4.05eV >= ETL E_c, W_R=5.2eV <= HTL E_v",
                "doping_types": "ETL: n-type, Active: intrinsic, HTL: p-type"
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(device_config, f, indent=4)
        
        print(f"Created: {filename} (Expected MPP: {device['mpp']:.1f} W/cm²)")
    
    print()

if __name__ == "__main__":
    top_devices = find_top_devices()
    create_high_performance_examples(top_devices)
    
    print("=== NEXT STEPS ===")
    print("1. Test these high-performance parameters with Script 7:")
    print("   python scripts/7_predict_experimental_data.py")
    print("   (Replace example_device_parameters.json with high_performance_device_1.json)")
    print()
    print("2. Use Script 8 to optimize further:")
    print("   python scripts/8_optimize_device_parameters.py --method global --maxiter 100")
    print()
    print("3. These parameters should yield PCE > 25% if the training data is accurate!")
