"""
Feature Names Definition for Solar Cell Optimization Pipeline

This script defines all feature names used in the ML-driven optimization pipeline.
It includes the 15 primary device parameters (optimization variables) and any derived features.
"""

import json
import os

# Create results directory
os.makedirs('results', exist_ok=True)

# =============================================================================
# PRIMARY DEVICE PARAMETERS (15 Optimization Variables)
# =============================================================================

# Layer 1 (PCBM - Electron Transport Layer)
L1_PARAMETERS = {
    'L1_L': 'Layer 1 thickness (nm)',
    'L1_E_c': 'Layer 1 conduction band energy (eV)',
    'L1_E_v': 'Layer 1 valence band energy (eV)',
    'L1_N_D': 'Layer 1 donor concentration (m⁻³)',
    'L1_N_A': 'Layer 1 acceptor concentration (m⁻³)'
}

# Layer 2 (MAPI - Active Layer)
L2_PARAMETERS = {
    'L2_L': 'Layer 2 thickness (nm)',
    'L2_E_c': 'Layer 2 conduction band energy (eV)',
    'L2_E_v': 'Layer 2 valence band energy (eV)',
    'L2_N_D': 'Layer 2 donor concentration (m⁻³)',
    'L2_N_A': 'Layer 2 acceptor concentration (m⁻³)'
}

# Layer 3 (PEDOT - Hole Transport Layer)
L3_PARAMETERS = {
    'L3_L': 'Layer 3 thickness (nm)',
    'L3_E_c': 'Layer 3 conduction band energy (eV)',
    'L3_E_v': 'Layer 3 valence band energy (eV)',
    'L3_N_D': 'Layer 3 donor concentration (m⁻³)',
    'L3_N_A': 'Layer 3 acceptor concentration (m⁻³)'
}

# Combine all primary parameters
PRIMARY_PARAMETERS = {**L1_PARAMETERS, **L2_PARAMETERS, **L3_PARAMETERS}

# =============================================================================
# DERIVED FEATURES (Calculated from primary parameters)
# =============================================================================

DERIVED_FEATURES = {
    # Thickness features
    'total_thickness': 'Total device thickness (L1_L + L2_L + L3_L)',
    'thickness_ratio_L2': 'Active layer thickness ratio (L2_L / total_thickness)',
    'thickness_ratio_ETL': 'ETL thickness ratio (L1_L / total_thickness)',
    'thickness_ratio_HTL': 'HTL thickness ratio (L3_L / total_thickness)',
    
    # Energy gap features
    'energy_gap_L1': 'Layer 1 energy gap (L1_E_c - L1_E_v)',
    'energy_gap_L2': 'Layer 2 energy gap (L2_E_c - L2_E_v)',
    'energy_gap_L3': 'Layer 3 energy gap (L3_E_c - L3_E_v)',
    
    # Band alignment features
    'band_offset_L1_L2': 'Band offset between Layer 1 and Layer 2',
    'band_offset_L2_L3': 'Band offset between Layer 2 and Layer 3',
    'conduction_band_offset': 'Conduction band offset across device',
    'valence_band_offset': 'Valence band offset across device',
    
    # Doping features
    'doping_ratio_L1': 'Layer 1 doping ratio (L1_N_D / L1_N_A)',
    'doping_ratio_L2': 'Layer 2 doping ratio (L2_N_D / L2_N_A)',
    'doping_ratio_L3': 'Layer 3 doping ratio (L3_N_D / L3_N_A)',
    'total_donor_concentration': 'Total donor concentration across device',
    'total_acceptor_concentration': 'Total acceptor concentration across device',
    
    # Material property features
    'average_energy_gap': 'Average energy gap across all layers',
    'energy_gap_variance': 'Variance in energy gaps across layers',
    'thickness_variance': 'Variance in layer thicknesses',
    'doping_variance': 'Variance in doping concentrations'
}

# =============================================================================
# TARGET VARIABLES (What we want to predict/optimize)
# =============================================================================

EFFICIENCY_TARGETS = {
    'MPP': 'Maximum Power Point (W/cm²)',
    'Jsc': 'Short-circuit current density (A/cm²)',
    'Voc': 'Open-circuit voltage (V)',
    'FF': 'Fill Factor (dimensionless)',
    'PCE': 'Power Conversion Efficiency (%)'
}

RECOMBINATION_TARGETS = {
    'IntSRHn_mean': 'Mean electron interfacial recombination rate',
    'IntSRHn_std': 'Standard deviation of electron recombination',
    'IntSRHp_mean': 'Mean hole interfacial recombination rate',
    'IntSRHp_std': 'Standard deviation of hole recombination',
    'IntSRH_total': 'Total interfacial recombination rate',
    'IntSRH_ratio': 'Ratio of electron to hole recombination'
}

# Combine all targets
ALL_TARGETS = {**EFFICIENCY_TARGETS, **RECOMBINATION_TARGETS}

# =============================================================================
# FEATURE CATEGORIES
# =============================================================================

FEATURE_CATEGORIES = {
    'primary_parameters': list(PRIMARY_PARAMETERS.keys()),
    'derived_features': list(DERIVED_FEATURES.keys()),
    'efficiency_targets': list(EFFICIENCY_TARGETS.keys()),
    'recombination_targets': list(RECOMBINATION_TARGETS.keys()),
    'all_features': list(PRIMARY_PARAMETERS.keys()) + list(DERIVED_FEATURES.keys()),
    'all_targets': list(ALL_TARGETS.keys())
}

# =============================================================================
# PARAMETER BOUNDS (For optimization constraints)
# =============================================================================

PARAMETER_BOUNDS = {
    # Layer 1 (PCBM - Electron Transport Layer)
    'L1_L': (20, 50),      # nm
    'L1_E_c': (3.7, 4.0),  # eV
    'L1_E_v': (5.7, 5.9),  # eV
    'L1_N_D': (1e20, 1e21), # m⁻³
    'L1_N_A': (1e20, 1e21), # m⁻³
    
    # Layer 2 (MAPI - Active Layer)
    'L2_L': (200, 500),     # nm
    'L2_E_c': (4.4, 4.6),   # eV
    'L2_E_v': (5.6, 5.8),   # eV
    'L2_N_D': (1e20, 1e21), # m⁻³
    'L2_N_A': (1e20, 1e21), # m⁻³
    
    # Layer 3 (PEDOT - Hole Transport Layer)
    'L3_L': (20, 50),       # nm
    'L3_E_c': (3.4, 3.6),   # eV
    'L3_E_v': (5.3, 5.5),   # eV
    'L3_N_D': (1e20, 1e21), # m⁻³
    'L3_N_A': (1e20, 1e21)  # m⁻³
}

# =============================================================================
# SAVE FEATURE DEFINITIONS
# =============================================================================

def save_feature_definitions():
    """Save all feature definitions to JSON files."""
    
    # Create feature definitions dictionary
    feature_definitions = {
        'primary_parameters': PRIMARY_PARAMETERS,
        'derived_features': DERIVED_FEATURES,
        'efficiency_targets': EFFICIENCY_TARGETS,
        'recombination_targets': RECOMBINATION_TARGETS,
        'all_targets': ALL_TARGETS,
        'feature_categories': FEATURE_CATEGORIES,
        'parameter_bounds': PARAMETER_BOUNDS
    }
    
    # Save to JSON file
    with open('results/feature_definitions.json', 'w') as f:
        json.dump(feature_definitions, f, indent=2)
    
    print("Feature definitions saved to: results/feature_definitions.json")
    
    # Also save individual files for easy access
    with open('results/primary_parameters.json', 'w') as f:
        json.dump(PRIMARY_PARAMETERS, f, indent=2)
    
    with open('results/parameter_bounds.json', 'w') as f:
        json.dump(PARAMETER_BOUNDS, f, indent=2)
    
    with open('results/target_variables.json', 'w') as f:
        json.dump(ALL_TARGETS, f, indent=2)
    
    print("Individual parameter files saved to results/")

def print_feature_summary():
    """Print a summary of all features."""
    print("=" * 60)
    print("FEATURE DEFINITIONS FOR SOLAR CELL OPTIMIZATION")
    print("=" * 60)
    
    print(f"\nPRIMARY PARAMETERS ({len(PRIMARY_PARAMETERS)}):")
    print("-" * 40)
    for param, description in PRIMARY_PARAMETERS.items():
        print(f"  {param}: {description}")
    
    print(f"\nDERIVED FEATURES ({len(DERIVED_FEATURES)}):")
    print("-" * 40)
    for feature, description in DERIVED_FEATURES.items():
        print(f"  {feature}: {description}")
    
    print(f"\nEFFICIENCY TARGETS ({len(EFFICIENCY_TARGETS)}):")
    print("-" * 40)
    for target, description in EFFICIENCY_TARGETS.items():
        print(f"  {target}: {description}")
    
    print(f"\nRECOMBINATION TARGETS ({len(RECOMBINATION_TARGETS)}):")
    print("-" * 40)
    for target, description in RECOMBINATION_TARGETS.items():
        print(f"  {target}: {description}")
    
    print(f"\nPARAMETER BOUNDS:")
    print("-" * 40)
    for param, (min_val, max_val) in PARAMETER_BOUNDS.items():
        print(f"  {param}: [{min_val}, {max_val}]")
    
    print("\n" + "=" * 60)
    print(f"Total features: {len(PRIMARY_PARAMETERS) + len(DERIVED_FEATURES)}")
    print(f"Total targets: {len(ALL_TARGETS)}")
    print("=" * 60)

if __name__ == "__main__":
    print_feature_summary()
    save_feature_definitions()
    print("\nFeature definitions created successfully!")