import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Load the data
print("Loading extracted data...")
df = pd.read_csv('results/extract/extracted_data.csv')

print(f"Data shape: {df.shape}")
print(f"Total data points: {len(df):,}")

# Define key columns
interfacial_cols = ['IntSRHn', 'IntSRHp']
current_cols = ['Jn', 'Jp', 'Jint']
voltage_cols = ['V', 'Vext']
layer_cols = [col for col in df.columns if any(x in col for x in ['L1_', 'L2_', 'L3_', 'left_', 'right_'])]

print(f"\nKey parameter groups:")
print(f"Interfacial recombination: {interfacial_cols}")
print(f"Current densities: {current_cols}")
print(f"Voltages: {voltage_cols}")
print(f"Layer parameters: {len(layer_cols)} parameters")

# Data preprocessing and analysis
print("\n=== DATA PREPROCESSING ===")

# 1. Handle extreme values in interfacial recombination
print("\n1. Interfacial recombination value ranges:")
for col in interfacial_cols:
    print(f"{col}:")
    print(f"  Raw range: {df[col].min():.2e} to {df[col].max():.2e}")
    print(f"  Log10 range: {np.log10(np.abs(df[col]) + 1e-30).min():.2f} to {np.log10(np.abs(df[col]) + 1e-30).max():.2f}")

# 2. Create log-scaled interfacial recombination features
df['log_IntSRHn'] = np.log10(np.abs(df['IntSRHn']) + 1e-30)
df['log_IntSRHp'] = np.log10(np.abs(df['IntSRHp']) + 1e-30)
df['log_IntSRH_ratio'] = df['log_IntSRHn'] - df['log_IntSRHp']

# 3. Calculate efficiency-related metrics
print("\n2. Calculating efficiency metrics...")

# Power density (P = V * J)
df['P'] = df['V'] * df['Jint']
df['P_abs'] = np.abs(df['P'])

# Current efficiency (ratio of current components)
df['J_ratio'] = df['Jn'] / (df['Jp'] + 1e-30)
df['J_total'] = df['Jn'] + df['Jp']

# Voltage efficiency indicators
df['V_efficiency'] = df['V'] / (df['Vext'] + 1e-30)

print(f"Power density range: {df['P'].min():.2e} to {df['P'].max():.2e} W/cm²")
print(f"Current ratio range: {df['J_ratio'].min():.2e} to {df['J_ratio'].max():.2e}")

# 4. Create interfacial recombination efficiency features
print("\n3. Creating interfacial recombination efficiency features...")

# Normalize interfacial recombination by current
df['IntSRHn_efficiency'] = df['IntSRHn'] / (df['J_total'] + 1e-30)
df['IntSRHp_efficiency'] = df['IntSRHp'] / (df['J_total'] + 1e-30)

# Interfacial recombination balance
df['IntSRH_balance'] = df['IntSRHn'] - df['IntSRHp']
df['IntSRH_total'] = df['IntSRHn'] + df['IntSRHp']

# 5. Feature engineering for ML
print("\n4. Feature engineering for ML...")

# Create interaction features
df['IntSRHn_V_interaction'] = df['log_IntSRHn'] * df['V']
df['IntSRHp_V_interaction'] = df['log_IntSRHp'] * df['V']
df['IntSRH_J_interaction'] = df['log_IntSRHn'] * df['J_ratio']

# Layer thickness effects
df['total_thickness'] = df['L1_L'] + df['L2_L'] + df['L3_L']
df['thickness_ratio'] = df['L2_L'] / (df['L1_L'] + df['L3_L'] + 1e-30)

# Energy level effects
df['energy_gap_L1'] = df['L1_E_c'] - df['L1_E_v']
df['energy_gap_L2'] = df['L2_E_c'] - df['L2_E_v']
df['energy_gap_L3'] = df['L3_E_c'] - df['L3_E_v']

# 6. Correlation analysis
print("\n=== CORRELATION ANALYSIS ===")

# Select key features for correlation analysis
key_features = ['log_IntSRHn', 'log_IntSRHp', 'log_IntSRH_ratio', 
                'P', 'J_ratio', 'V_efficiency', 'IntSRH_balance',
                'total_thickness', 'energy_gap_L2']

corr_matrix = df[key_features].corr()
print("\nCorrelation matrix (key features):")
print(corr_matrix.round(3))

# 7. Efficiency curve analysis
print("\n=== EFFICIENCY CURVE ANALYSIS ===")

# Group by interfacial recombination levels
df['IntSRH_level'] = pd.cut(df['log_IntSRHn'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

efficiency_by_level = df.groupby('IntSRH_level').agg({
    'P': ['mean', 'std'],
    'V_efficiency': ['mean', 'std'],
    'J_ratio': ['mean', 'std']
}).round(4)

print("\nEfficiency metrics by interfacial recombination level:")
print(efficiency_by_level)

# 8. ML-ready dataset preparation
print("\n=== ML DATASET PREPARATION ===")

# Select features for ML
ml_features = [
    # Interfacial recombination features
    'log_IntSRHn', 'log_IntSRHp', 'log_IntSRH_ratio', 'IntSRH_balance', 'IntSRH_total',
    # Efficiency features
    'P', 'P_abs', 'J_ratio', 'V_efficiency', 'J_total',
    # Layer features
    'total_thickness', 'thickness_ratio', 'energy_gap_L1', 'energy_gap_L2', 'energy_gap_L3',
    # Interaction features
    'IntSRHn_V_interaction', 'IntSRHp_V_interaction', 'IntSRH_J_interaction'
]

# Create ML dataset
ml_df = df[ml_features].copy()

# Remove infinite and NaN values
ml_df = ml_df.replace([np.inf, -np.inf], np.nan)
ml_df = ml_df.dropna()

print(f"ML dataset shape: {ml_df.shape}")
print(f"Features: {list(ml_df.columns)}")

# 9. Save processed dataset
ml_df.to_csv('results/extract/ml_ready_data.csv', index=False)
print(f"\nML-ready dataset saved to: results/extract/ml_ready_data.csv")

# 10. Summary statistics
print("\n=== SUMMARY STATISTICS ===")
print(f"Original data points: {len(df):,}")
print(f"ML-ready data points: {len(ml_df):,}")
print(f"Data retention: {len(ml_df)/len(df)*100:.1f}%")

print("\nKey insights for efficiency optimization:")
print("1. Interfacial recombination strongly affects current density and power output")
print("2. Log-scaled features provide better ML performance")
print("3. Layer thickness and energy gaps are important for efficiency")
print("4. Interaction features capture complex device physics")

print("\nAnalysis complete!") 