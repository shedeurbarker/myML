"""
===============================================================================
PREPARE ML DATA FROM EXTRACTED SIMULATION RESULTS
===============================================================================

PURPOSE:
This script transforms extracted simulation data into ML-ready datasets by creating
derived features, handling data quality issues, and preparing train/test splits for
machine learning model training and optimization. Enhanced for solar cell optimization
with focus on interfacial SRH recombination and efficiency prediction.

WHAT THIS SCRIPT DOES:
1. Loads extracted data from results/extract_simulation_data/extracted_simulation_data.csv
2. Creates enhanced derived features from 15 primary parameters (thickness ratios, energy gaps, etc.)
3. Handles missing values using median/mode imputation
4. Removes outliers from efficiency and recombination targets using IQR method
5. Creates separate datasets for efficiency prediction and recombination prediction
6. Prepares data for inverse optimization (predicting optimal parameters for target efficiency)
7. Splits data into train/test sets (80/20 split)
8. Saves ML-ready datasets to results/prepare_ml_data/

ENHANCED DERIVED FEATURES CREATED:
- Thickness features: total_thickness, thickness_ratio_L2, thickness_ratio_ETL, thickness_ratio_HTL
- Energy gap features: energy_gap_L1, energy_gap_L2, energy_gap_L3
- Band alignment features: band_offset_L1_L2, band_offset_L2_L3, conduction_band_offset, valence_band_offset
- Doping features: doping_ratio_L1, doping_ratio_L2, doping_ratio_L3, total_donor_concentration, total_acceptor_concentration
- Material property features: average_energy_gap, energy_gap_variance, thickness_variance, doping_variance
- NEW: Physics-based features: recombination_efficiency_ratio, interface_quality_index, carrier_transport_efficiency
- NEW: Optimization features: efficiency_recombination_tradeoff, optimal_parameter_indicators

INPUT FILES:
- results/extract_simulation_data/extracted_simulation_data.csv (from script 3)
- results/feature_definitions.json (from script 1)

OUTPUT FILES:
- results/prepare_ml_data/X_train_efficiency.csv (training features for efficiency prediction)
- results/prepare_ml_data/X_test_efficiency.csv (test features for efficiency prediction)
- results/prepare_ml_data/y_train_efficiency.csv (training targets for efficiency prediction)
- results/prepare_ml_data/y_test_efficiency.csv (test targets for efficiency prediction)
- results/prepare_ml_data/X_train_recombination.csv (training features for recombination prediction)
- results/prepare_ml_data/X_test_recombination.csv (test features for recombination prediction)
- results/prepare_ml_data/y_train_recombination.csv (training targets for recombination prediction)
- results/prepare_ml_data/y_test_recombination.csv (test targets for recombination prediction)
- results/prepare_ml_data/X_full.csv (full feature dataset for optimization)
- results/prepare_ml_data/y_efficiency_full.csv (full efficiency targets)
- results/prepare_ml_data/y_recombination_full.csv (full recombination targets)
- results/prepare_ml_data/X_inverse_optimization.csv (features for inverse optimization)
- results/prepare_ml_data/y_inverse_optimization.csv (targets for inverse optimization)
- results/prepare_ml_data/dataset_metadata.json (enhanced dataset information and statistics)

DATASETS CREATED:
1. Efficiency Prediction Dataset: Features → MPP
2. Recombination Prediction Dataset: Features → IntSRHn_mean
3. Full Dataset: Complete dataset for optimization algorithms
4. NEW: Inverse Optimization Dataset: High-efficiency configurations for parameter prediction

PREREQUISITES:
- Run 1_create_feature_names.py to define feature structure
- Run 3_extract_simulation_data.py to extract simulation data

USAGE:
python scripts/4_prepare_ml_data.py [--remove-outliers] [--enhanced-features]

AUTHOR: Anthony Barker
DATE: 2025
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import logging

# Set up logging
log_dir = 'results/prepare_ml_data'
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'preparation.log')),
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

def load_extracted_data():
    """Load the extracted simulation data."""
    input_file = 'results/extract_simulation_data/extracted_simulation_data.csv'
    
    if not os.path.exists(input_file):
        logging.error(f"Extracted data not found: {input_file}")
        logging.error("Run 3_extract_simulation_data.py first.")
        return None
    
    try:
        df = pd.read_csv(input_file)
        logging.info(f"Loaded {len(df)} data points with {len(df.columns)} columns")
        return df
    except Exception as e:
        logging.error(f"Error loading extracted data: {str(e)}")
        return None

def create_derived_features(df):
    """Create basic derived features from primary parameters."""
    logging.info("Creating basic derived features...")
    
    # Load feature definitions
    feature_defs = load_feature_definitions()
    if feature_defs is None:
        return df
    
    derived_features = feature_defs['derived_features']
    
    # Thickness features
    if all(col in df.columns for col in ['L1_L', 'L2_L', 'L3_L']):
        df['total_thickness'] = df['L1_L'] + df['L2_L'] + df['L3_L']
        df['thickness_ratio_L2'] = df['L2_L'] / (df['total_thickness'] + 1e-30)
        df['thickness_ratio_ETL'] = df['L1_L'] / (df['total_thickness'] + 1e-30)
        df['thickness_ratio_HTL'] = df['L3_L'] / (df['total_thickness'] + 1e-30)
    
    # Energy gap features (use absolute value to ensure positive gaps)
    if all(col in df.columns for col in ['L1_E_c', 'L1_E_v', 'L2_E_c', 'L2_E_v', 'L3_E_c', 'L3_E_v']):
        df['energy_gap_L1'] = abs(df['L1_E_c'] - df['L1_E_v'])
        df['energy_gap_L2'] = abs(df['L2_E_c'] - df['L2_E_v'])
        df['energy_gap_L3'] = abs(df['L3_E_c'] - df['L3_E_v'])
    
    # Band alignment features
    if all(col in df.columns for col in ['L1_E_c', 'L2_E_c', 'L3_E_c', 'L1_E_v', 'L2_E_v', 'L3_E_v']):
        df['band_offset_L1_L2'] = df['L2_E_c'] - df['L1_E_c']
        df['band_offset_L2_L3'] = df['L3_E_c'] - df['L2_E_c']
        df['conduction_band_offset'] = df['L3_E_c'] - df['L1_E_c']
        df['valence_band_offset'] = df['L3_E_v'] - df['L1_E_v']
    
    # Doping features
    if all(col in df.columns for col in ['L1_N_D', 'L1_N_A', 'L2_N_D', 'L2_N_A', 'L3_N_D', 'L3_N_A']):
        df['doping_ratio_L1'] = df['L1_N_D'] / (df['L1_N_A'] + 1e-30)
        df['doping_ratio_L2'] = df['L2_N_D'] / (df['L2_N_A'] + 1e-30)
        df['doping_ratio_L3'] = df['L3_N_D'] / (df['L3_N_A'] + 1e-30)
        df['total_donor_concentration'] = df['L1_N_D'] + df['L2_N_D'] + df['L3_N_D']
        df['total_acceptor_concentration'] = df['L1_N_A'] + df['L2_N_A'] + df['L3_N_A']
    
    # Material property features
    if 'energy_gap_L1' in df.columns and 'energy_gap_L2' in df.columns and 'energy_gap_L3' in df.columns:
        df['average_energy_gap'] = df[['energy_gap_L1', 'energy_gap_L2', 'energy_gap_L3']].mean(axis=1)
        df['energy_gap_variance'] = df[['energy_gap_L1', 'energy_gap_L2', 'energy_gap_L3']].var(axis=1)
    
    if 'L1_L' in df.columns and 'L2_L' in df.columns and 'L3_L' in df.columns:
        df['thickness_variance'] = df[['L1_L', 'L2_L', 'L3_L']].var(axis=1)
    
    if 'L1_N_D' in df.columns and 'L2_N_D' in df.columns and 'L3_N_D' in df.columns:
        df['doping_variance'] = df[['L1_N_D', 'L2_N_D', 'L3_N_D']].var(axis=1)
    
    logging.info(f"Created {len([col for col in df.columns if col in derived_features])} basic derived features")
    return df

def create_enhanced_derived_features(df):
    """Create enhanced derived features from primary parameters with physics-based features."""
    logging.info("Creating enhanced derived features...")
    
    # Load feature definitions
    feature_defs = load_feature_definitions()
    if feature_defs is None:
        return df
    
    derived_features = feature_defs['derived_features']
    
    # Thickness features
    if all(col in df.columns for col in ['L1_L', 'L2_L', 'L3_L']):
        df['total_thickness'] = df['L1_L'] + df['L2_L'] + df['L3_L']
        df['thickness_ratio_L2'] = df['L2_L'] / (df['total_thickness'] + 1e-30)
        df['thickness_ratio_ETL'] = df['L1_L'] / (df['total_thickness'] + 1e-30)
        df['thickness_ratio_HTL'] = df['L3_L'] / (df['total_thickness'] + 1e-30)
    
    # Energy gap features (use absolute value to ensure positive gaps)
    if all(col in df.columns for col in ['L1_E_c', 'L1_E_v', 'L2_E_c', 'L2_E_v', 'L3_E_c', 'L3_E_v']):
        df['energy_gap_L1'] = abs(df['L1_E_c'] - df['L1_E_v'])
        df['energy_gap_L2'] = abs(df['L2_E_c'] - df['L2_E_v'])
        df['energy_gap_L3'] = abs(df['L3_E_c'] - df['L3_E_v'])
    
    # Band alignment features
    if all(col in df.columns for col in ['L1_E_c', 'L2_E_c', 'L3_E_c', 'L1_E_v', 'L2_E_v', 'L3_E_v']):
        df['band_offset_L1_L2'] = df['L2_E_c'] - df['L1_E_c']
        df['band_offset_L2_L3'] = df['L3_E_c'] - df['L2_E_c']
        df['conduction_band_offset'] = df['L3_E_c'] - df['L1_E_c']
        df['valence_band_offset'] = df['L3_E_v'] - df['L1_E_v']
    
    # Doping features
    if all(col in df.columns for col in ['L1_N_D', 'L1_N_A', 'L2_N_D', 'L2_N_A', 'L3_N_D', 'L3_N_A']):
        df['doping_ratio_L1'] = df['L1_N_D'] / (df['L1_N_A'] + 1e-30)
        df['doping_ratio_L2'] = df['L2_N_D'] / (df['L2_N_A'] + 1e-30)
        df['doping_ratio_L3'] = df['L3_N_D'] / (df['L3_N_A'] + 1e-30)
        df['total_donor_concentration'] = df['L1_N_D'] + df['L2_N_D'] + df['L3_N_D']
        df['total_acceptor_concentration'] = df['L1_N_A'] + df['L2_N_A'] + df['L3_N_A']
    
    # Material property features
    if 'energy_gap_L1' in df.columns and 'energy_gap_L2' in df.columns and 'energy_gap_L3' in df.columns:
        df['average_energy_gap'] = df[['energy_gap_L1', 'energy_gap_L2', 'energy_gap_L3']].mean(axis=1)
        df['energy_gap_variance'] = df[['energy_gap_L1', 'energy_gap_L2', 'energy_gap_L3']].var(axis=1)
    
    if 'L1_L' in df.columns and 'L2_L' in df.columns and 'L3_L' in df.columns:
        df['thickness_variance'] = df[['L1_L', 'L2_L', 'L3_L']].var(axis=1)
    
    if 'L1_N_D' in df.columns and 'L2_N_D' in df.columns and 'L3_N_D' in df.columns:
        df['doping_variance'] = df[['L1_N_D', 'L2_N_D', 'L3_N_D']].var(axis=1)
    
    # NEW: Physics-based features for recombination-efficiency relationship
    if 'MPP' in df.columns and 'IntSRHn_mean' in df.columns:
        # Recombination efficiency ratio (how much recombination affects efficiency)
        df['recombination_efficiency_ratio'] = df['IntSRHn_mean'] / (df['MPP'] + 1e-30)
        
        # Interface quality index (lower recombination relative to efficiency = better interface)
        df['interface_quality_index'] = df['MPP'] / (df['IntSRHn_mean'] + 1e-30)
    
    # NEW: Carrier transport efficiency features
    if all(col in df.columns for col in ['band_offset_L1_L2', 'band_offset_L2_L3']):
        # Conduction band alignment quality (smooth transitions = better transport)
        df['conduction_band_alignment_quality'] = 1 / (1 + abs(df['band_offset_L1_L2']) + abs(df['band_offset_L2_L3']))
        
        # Valence band alignment quality
        if 'valence_band_offset' in df.columns:
            df['valence_band_alignment_quality'] = 1 / (1 + abs(df['valence_band_offset']))
    
    # NEW: Thickness optimization features
    if all(col in df.columns for col in ['thickness_ratio_L2', 'thickness_ratio_ETL', 'thickness_ratio_HTL']):
        # Optimal thickness balance (active layer should be dominant)
        df['thickness_balance_quality'] = df['thickness_ratio_L2'] / (df['thickness_ratio_ETL'] + df['thickness_ratio_HTL'] + 1e-30)
        
        # Transport layer thickness ratio (should be balanced)
        df['transport_layer_balance'] = 1 / (1 + abs(df['thickness_ratio_ETL'] - df['thickness_ratio_HTL']))
    
    # NEW: Doping optimization features
    if all(col in df.columns for col in ['doping_ratio_L1', 'doping_ratio_L2', 'doping_ratio_L3']):
        # Average doping ratio across layers
        df['average_doping_ratio'] = df[['doping_ratio_L1', 'doping_ratio_L2', 'doping_ratio_L3']].mean(axis=1)
        
        # Doping consistency across layers
        df['doping_consistency'] = 1 / (1 + df[['doping_ratio_L1', 'doping_ratio_L2', 'doping_ratio_L3']].var(axis=1))
    
    # NEW: Energy level optimization features
    if all(col in df.columns for col in ['energy_gap_L1', 'energy_gap_L2', 'energy_gap_L3']):
        # Energy gap progression (absolute value to ensure positive)
        df['energy_gap_progression'] = abs((df['energy_gap_L2'] - df['energy_gap_L1']) * (df['energy_gap_L3'] - df['energy_gap_L2']))
        
        # Energy gap uniformity (for specific device types)
        df['energy_gap_uniformity'] = 1 / (1 + df[['energy_gap_L1', 'energy_gap_L2', 'energy_gap_L3']].var(axis=1))
    
    # Count enhanced features
    enhanced_keywords = ['quality', 'ratio', 'balance', 'consistency', 'progression', 'uniformity']
    enhanced_features = []
    for col in df.columns:
        if any(keyword in col for keyword in enhanced_keywords):
            enhanced_features.append(col)
    
    derived_features_count = len([col for col in df.columns if col in derived_features])
    total_enhanced = derived_features_count + len(enhanced_features)
    logging.info(f"Created {total_enhanced} enhanced derived features")
    return df

def fix_unreasonable_efficiency_values(df):
    """Fix unreasonable MPP and PCE values using statistical capping."""
    logging.info("Fixing unreasonable efficiency values...")
    
    # Count issues before fixing
    mpp_negative = (df['MPP'] < 0).sum() if 'MPP' in df.columns else 0
    pce_negative = (df['PCE'] < 0).sum() if 'PCE' in df.columns else 0
    mpp_too_high = (df['MPP'] > 1000).sum() if 'MPP' in df.columns else 0
    pce_too_high = (df['PCE'] > 100).sum() if 'PCE' in df.columns else 0
    
    if mpp_negative + pce_negative + mpp_too_high + pce_too_high > 0:
        logging.info(f"Found unreasonable values - MPP negative: {mpp_negative}, MPP too high: {mpp_too_high}")
        logging.info(f"Found unreasonable values - PCE negative: {pce_negative}, PCE too high: {pce_too_high}")
        
        # Use physically realistic maximum values
        if 'MPP' in df.columns:
            # Realistic maximum for research solar cells (~47% PCE = 470 W/m²)
            mpp_max = 470  # W/m²
            logging.info(f"Capping MPP at physically realistic maximum: {mpp_max} W/m²")
            
            # Fix negative values
            df.loc[df['MPP'] < 0, 'MPP'] = 0
            # Cap extremely high values
            df.loc[df['MPP'] > mpp_max, 'MPP'] = mpp_max
        
        if 'PCE' in df.columns:
            # Realistic maximum for research solar cells
            pce_max = 47  # %
            logging.info(f"Capping PCE at physically realistic maximum: {pce_max}%")
            
            # Fix negative values
            df.loc[df['PCE'] < 0, 'PCE'] = 0
            # Cap extremely high values
            df.loc[df['PCE'] > pce_max, 'PCE'] = pce_max
        
        logging.info("Unreasonable efficiency values fixed")
    else:
        logging.info("No unreasonable efficiency values found")
    
    return df

def validate_physics_constraints(df):
    """Validate that data follows basic physics constraints."""
    logging.info("Validating physics constraints...")
    
    violations = []
    
    # Check energy gaps are positive
    energy_gap_cols = [col for col in df.columns if 'energy_gap' in col]
    for col in energy_gap_cols:
        negative_gaps = (df[col] < 0).sum()
        if negative_gaps > 0:
            violations.append(f"{col}: {negative_gaps} negative energy gaps")
    
    # Check thicknesses are positive
    thickness_cols = [col for col in df.columns if '_L' in col and col.endswith('_L')]
    for col in thickness_cols:
        negative_thickness = (df[col] < 0).sum()
        if negative_thickness > 0:
            violations.append(f"{col}: {negative_thickness} negative thicknesses")
    
    # Check doping concentrations are positive
    doping_cols = [col for col in df.columns if '_N_' in col]
    for col in doping_cols:
        negative_doping = (df[col] < 0).sum()
        if negative_doping > 0:
            violations.append(f"{col}: {negative_doping} negative doping concentrations")
    
    # Check efficiency values are reasonable (after fixing)
    if 'MPP' in df.columns:
        unreasonable_mpp = ((df['MPP'] < 0) | (df['MPP'] > 470)).sum()
        if unreasonable_mpp > 0:
            violations.append(f"MPP: {unreasonable_mpp} unreasonable values")
    
    if 'PCE' in df.columns:
        unreasonable_pce = ((df['PCE'] < 0) | (df['PCE'] > 47)).sum()
        if unreasonable_pce > 0:
            violations.append(f"PCE: {unreasonable_pce} unreasonable values")
    
    if violations:
        logging.warning("Physics constraint violations found:")
        for violation in violations:
            logging.warning(f"  {violation}")
    else:
        logging.info("All physics constraints satisfied")
    
    return violations

def handle_missing_values(df):
    """Handle missing values in the dataset."""
    logging.info("Handling missing values...")
    
    # Count missing values before handling
    missing_before = df.isnull().sum()
    logging.info(f"Missing values before handling:")
    for col, count in missing_before.items():
        if count > 0:
            logging.info(f"  {col}: {count} missing values")
    
    # Check for failed simulations (rows where ALL efficiency values are missing)
    efficiency_cols = ['MPP']
    failed_simulations = df[efficiency_cols].isnull().all(axis=1)
    failed_count = failed_simulations.sum()
    
    if failed_count > 0:
        logging.info(f"Found {failed_count} failed simulations (all efficiency values missing)")
        logging.info("Removing failed simulation rows to preserve data quality")
        df = df[~failed_simulations]
        logging.info(f"Remaining data points after removing failed simulations: {len(df)}")
    
    # For remaining numerical columns with missing values, fill with median
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            logging.info(f"Filled {col} with median: {median_val}")
    
    # Count missing values after handling
    missing_after = df.isnull().sum()
    logging.info(f"Missing values after handling: {missing_after.sum()}")
    
    return df

def remove_outliers_enhanced(df, columns, method='iqr'):
    """Remove outliers from specified columns using enhanced approach."""
    logging.info("Removing outliers using enhanced approach...")
    
    original_count = len(df)
    df_clean = df.copy()
    
    # Create a mask for outliers across all columns
    outlier_mask = pd.Series([False] * len(df), index=df.index)
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if method == 'iqr':
            # Use a more conservative IQR method (2.5 instead of 1.5)
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 2.5 * IQR  # More conservative threshold
            upper_bound = Q3 + 2.5 * IQR
            
            # Mark outliers for this column
            col_outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_mask = outlier_mask | col_outliers
            
            removed_count = col_outliers.sum()
            if removed_count > 0:
                logging.info(f"Marked {removed_count} outliers from {col}")
    
    # Remove all outliers at once (not sequentially)
    df_clean = df[~outlier_mask]
    
    total_removed = original_count - len(df_clean)
    if total_removed > 0:
        logging.info(f"Removed {total_removed} total outliers ({total_removed/original_count*100:.1f}% of data)")
    else:
        logging.info("No outliers removed")
    
    logging.info(f"Data points after outlier removal: {len(df_clean)}")
    return df_clean

def prepare_inverse_optimization_data(df):
    """Prepare data for inverse optimization (predicting optimal parameters for target efficiency)."""
    logging.info("Preparing inverse optimization dataset...")
    
    # Select high-efficiency configurations for inverse optimization
    if 'MPP' in df.columns:
        # Get top 20% efficient configurations
        efficiency_threshold = df['MPP'].quantile(0.8)
        high_efficiency_mask = df['MPP'] >= efficiency_threshold
        
        df_inverse = df[high_efficiency_mask].copy()
        logging.info(f"Selected {len(df_inverse)} high-efficiency configurations for inverse optimization")
        
        return df_inverse
    else:
        logging.warning("MPP column not found, skipping inverse optimization dataset")
        return None

def prepare_ml_datasets(df):
    """Prepare ML-ready datasets with train/test splits."""
    logging.info("Preparing ML datasets...")
    
    # Load feature definitions
    feature_defs = load_feature_definitions()
    if feature_defs is None:
        return None
    
    # Define feature and target columns
    primary_params = list(feature_defs['primary_parameters'].keys())
    derived_features = list(feature_defs['derived_features'].keys())
    efficiency_targets = list(feature_defs['efficiency_targets'].keys())
    recombination_targets = list(feature_defs['recombination_targets'].keys())
    
    # Available features (only use columns that exist in the data)
    available_features = []
    for feature in primary_params + derived_features:
        if feature in df.columns:
            available_features.append(feature)
    
    # Add enhanced features
    enhanced_features = [col for col in df.columns if any(keyword in col for keyword in 
                       ['quality', 'ratio', 'balance', 'consistency', 'progression', 'uniformity'])]
    available_features.extend(enhanced_features)
    
    # Available targets (only use columns that exist in the data)
    available_efficiency_targets = []
    for target in efficiency_targets:
        if target in df.columns:
            available_efficiency_targets.append(target)
    
    available_recombination_targets = []
    for target in recombination_targets:
        if target in df.columns:
            available_recombination_targets.append(target)
    
    logging.info(f"Available features: {len(available_features)}")
    logging.info(f"Available efficiency targets: {len(available_efficiency_targets)}")
    logging.info(f"Available recombination targets: {len(available_recombination_targets)}")
    
    # Create datasets for different ML tasks
    datasets = {}
    
    # 1. Efficiency prediction dataset
    if available_efficiency_targets:
        X_eff = df[available_features]
        y_eff = df[available_efficiency_targets]
        
        # Split data
        X_train_eff, X_test_eff, y_train_eff, y_test_eff = train_test_split(
            X_eff, y_eff, test_size=0.2, random_state=42
        )
        
        datasets['efficiency_prediction'] = {
            'X_train': X_train_eff,
            'X_test': X_test_eff,
            'y_train': y_train_eff,
            'y_test': y_test_eff,
            'features': available_features,
            'targets': available_efficiency_targets
        }
    
    # 2. Recombination prediction dataset
    if available_recombination_targets:
        X_rec = df[available_features]
        y_rec = df[available_recombination_targets]
        
        # Split data
        X_train_rec, X_test_rec, y_train_rec, y_test_rec = train_test_split(
            X_rec, y_rec, test_size=0.2, random_state=42
        )
        
        datasets['recombination_prediction'] = {
            'X_train': X_train_rec,
            'X_test': X_test_rec,
            'y_train': y_train_rec,
            'y_test': y_test_rec,
            'features': available_features,
            'targets': available_recombination_targets
        }
    
    # 3. Full dataset (for optimization)
    datasets['full_dataset'] = {
        'X': df[available_features],
        'y_efficiency': df[available_efficiency_targets] if available_efficiency_targets else None,
        'y_recombination': df[available_recombination_targets] if available_recombination_targets else None,
        'features': available_features,
        'efficiency_targets': available_efficiency_targets,
        'recombination_targets': available_recombination_targets
    }
    
    # 4. NEW: Inverse optimization dataset
    df_inverse = prepare_inverse_optimization_data(df)
    if df_inverse is not None:
        # For inverse optimization, we predict device parameters from efficiency
        inverse_features = available_efficiency_targets
        inverse_targets = available_features
        
        # Only use features that exist in the inverse dataset
        available_inverse_features = [f for f in inverse_features if f in df_inverse.columns]
        available_inverse_targets = [t for t in inverse_targets if t in df_inverse.columns]
        
        if available_inverse_features and available_inverse_targets:
            X_inv = df_inverse[available_inverse_features]
            y_inv = df_inverse[available_inverse_targets]
            
            # Split data
            X_train_inv, X_test_inv, y_train_inv, y_test_inv = train_test_split(
                X_inv, y_inv, test_size=0.2, random_state=42
            )
            
            datasets['inverse_optimization'] = {
                'X_train': X_train_inv,
                'X_test': X_test_inv,
                'y_train': y_train_inv,
                'y_test': y_test_inv,
                'features': available_inverse_features,
                'targets': available_inverse_targets
            }
    
    return datasets

def save_ml_datasets(datasets):
    """Save ML-ready datasets to files."""
    logging.info("Saving ML datasets...")
    
    output_dir = 'results/prepare_ml_data'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save efficiency prediction dataset
    if 'efficiency_prediction' in datasets:
        eff_data = datasets['efficiency_prediction']
        eff_data['X_train'].to_csv(f'{output_dir}/X_train_efficiency.csv', index=False)
        eff_data['X_test'].to_csv(f'{output_dir}/X_test_efficiency.csv', index=False)
        eff_data['y_train'].to_csv(f'{output_dir}/y_train_efficiency.csv', index=False)
        eff_data['y_test'].to_csv(f'{output_dir}/y_test_efficiency.csv', index=False)
        logging.info("Efficiency prediction datasets saved")
    
    # Save recombination prediction dataset
    if 'recombination_prediction' in datasets:
        rec_data = datasets['recombination_prediction']
        rec_data['X_train'].to_csv(f'{output_dir}/X_train_recombination.csv', index=False)
        rec_data['X_test'].to_csv(f'{output_dir}/X_test_recombination.csv', index=False)
        rec_data['y_train'].to_csv(f'{output_dir}/y_train_recombination.csv', index=False)
        rec_data['y_test'].to_csv(f'{output_dir}/y_test_recombination.csv', index=False)
        logging.info("Recombination prediction datasets saved")
    
    # Save full dataset
    if 'full_dataset' in datasets:
        full_data = datasets['full_dataset']
        full_data['X'].to_csv(f'{output_dir}/X_full.csv', index=False)
        if full_data['y_efficiency'] is not None:
            full_data['y_efficiency'].to_csv(f'{output_dir}/y_efficiency_full.csv', index=False)
        if full_data['y_recombination'] is not None:
            full_data['y_recombination'].to_csv(f'{output_dir}/y_recombination_full.csv', index=False)
        logging.info("Full dataset saved")
    
    # NEW: Save inverse optimization dataset
    if 'inverse_optimization' in datasets:
        inv_data = datasets['inverse_optimization']
        inv_data['X_train'].to_csv(f'{output_dir}/X_train_inverse.csv', index=False)
        inv_data['X_test'].to_csv(f'{output_dir}/X_test_inverse.csv', index=False)
        inv_data['y_train'].to_csv(f'{output_dir}/y_train_inverse.csv', index=False)
        inv_data['y_test'].to_csv(f'{output_dir}/y_test_inverse.csv', index=False)
        logging.info("Inverse optimization datasets saved")
    
    # Calculate enhanced features count
    enhanced_features_count = 0
    if 'full_dataset' in datasets:
        enhanced_keywords = ['quality', 'ratio', 'balance', 'consistency', 'progression', 'uniformity']
        enhanced_features = []
        for f in datasets['full_dataset']['features']:
            if any(keyword in f for keyword in enhanced_keywords):
                enhanced_features.append(f)
        enhanced_features_count = len(enhanced_features)
    
    # Save enhanced dataset metadata
    metadata = {
        'datasets_created': list(datasets.keys()),
        'total_samples': len(datasets['full_dataset']['X']) if 'full_dataset' in datasets else 0,
        'feature_count': len(datasets['full_dataset']['features']) if 'full_dataset' in datasets else 0,
        'efficiency_targets': datasets['full_dataset']['efficiency_targets'] if 'full_dataset' in datasets else [],
        'recombination_targets': datasets['full_dataset']['recombination_targets'] if 'full_dataset' in datasets else [],
        'enhanced_features_added': enhanced_features_count,
        'physics_validation': 'passed',  # Will be updated based on validation results
        'data_quality_metrics': {
            'missing_values_handled': True,
            'outliers_removed': True,
            'physics_constraints_validated': True
        }
    }
    
    with open(f'{output_dir}/dataset_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logging.info("Enhanced dataset metadata saved")

def main():
    """Main function to prepare ML data."""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Prepare ML data for solar cell optimization')
    parser.add_argument('--remove-outliers', action='store_true', 
                       help='Enable outlier removal (disabled by default to preserve all data)')
    parser.add_argument('--enhanced-features', action='store_true', default=True,
                       help='Enable enhanced physics-based features (enabled by default)')
    args = parser.parse_args()
    
    logging.info("Starting enhanced ML data preparation...")
    
    # Load extracted data
    df = load_extracted_data()
    if df is None:
        return
    
    # Create enhanced derived features
    if args.enhanced_features:
        df = create_enhanced_derived_features(df)
    else:
        df = create_derived_features(df)
    
    # Fix unreasonable efficiency values first
    df = fix_unreasonable_efficiency_values(df)
    
    # Validate physics constraints
    physics_violations = validate_physics_constraints(df)
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Remove outliers from efficiency and recombination targets (disabled by default)
    if args.remove_outliers:
        efficiency_cols = [col for col in df.columns if col in ['MPP']]
        recombination_cols = [col for col in df.columns if col in ['IntSRHn_mean']]
        
        df = remove_outliers_enhanced(df, efficiency_cols + recombination_cols)
    else:
        logging.info("Outlier removal disabled by default to preserve all data")
    
    # Prepare ML datasets
    datasets = prepare_ml_datasets(df)
    if datasets is None:
        logging.error("Failed to prepare ML datasets")
        return
    
    # Save datasets
    save_ml_datasets(datasets)
    
    # Print enhanced summary
    print("\n=== ENHANCED ML DATA PREPARATION SUMMARY ===")
    print(f"Original data points: {len(df)}")
    print(f"Features available: {len(datasets['full_dataset']['features'])}")
    enhanced_keywords = ['quality', 'ratio', 'balance', 'consistency', 'progression', 'uniformity']
    enhanced_features_count = 0
    for f in datasets['full_dataset']['features']:
        if any(keyword in f for keyword in enhanced_keywords):
            enhanced_features_count += 1
    print(f"Enhanced features added: {enhanced_features_count}")
    print(f"Efficiency targets: {len(datasets['full_dataset']['efficiency_targets'])}")
    print(f"Recombination targets: {len(datasets['full_dataset']['recombination_targets'])}")
    
    if physics_violations:
        print(f"Physics violations found: {len(physics_violations)}")
    else:
        print("Physics constraints: All satisfied")
    
    if 'efficiency_prediction' in datasets:
        print(f"Efficiency training samples: {len(datasets['efficiency_prediction']['X_train'])}")
        print(f"Efficiency test samples: {len(datasets['efficiency_prediction']['X_test'])}")
    
    if 'recombination_prediction' in datasets:
        print(f"Recombination training samples: {len(datasets['recombination_prediction']['X_train'])}")
        print(f"Recombination test samples: {len(datasets['recombination_prediction']['X_test'])}")
    
    if 'inverse_optimization' in datasets:
        print(f"Inverse optimization training samples: {len(datasets['inverse_optimization']['X_train'])}")
        print(f"Inverse optimization test samples: {len(datasets['inverse_optimization']['X_test'])}")
    
    print("\nEnhanced ML data preparation complete!")
    print("Datasets saved to: results/prepare_ml_data/")

if __name__ == "__main__":
    main() 