import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the extracted data
print("Loading extracted data...")
df = pd.read_csv('results/extract/extracted_data.csv')

print(f"Data shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Find interfacial recombination related columns
int_columns = [col for col in df.columns if 'Int' in col or 'SRH' in col or 'recomb' in col.lower()]
print(f"\nInterfacial recombination columns: {int_columns}")

# Find efficiency related columns
efficiency_columns = [col for col in df.columns if any(x in col.lower() for x in ['j', 'v', 'efficiency', 'power', 'current', 'pce', 'ff', 'voc', 'jsc'])]
print(f"Efficiency related columns: {efficiency_columns}")

# Find layer parameter columns
layer_columns = [col for col in df.columns if any(x in col for x in ['L1_', 'L2_', 'L3_'])]
print(f"Layer parameter columns: {layer_columns}")

# Display basic statistics for interfacial recombination columns
print("\nInterfacial recombination statistics:")
for col in int_columns:
    print(f"{col}:")
    print(f"  Range: {df[col].min():.2e} to {df[col].max():.2e}")
    print(f"  Mean: {df[col].mean():.2e}")
    print(f"  Std: {df[col].std():.2e}")
    print(f"  Unique values: {df[col].nunique()}")

# Display basic statistics for efficiency columns
print("\nEfficiency parameters statistics:")
for col in efficiency_columns:
    print(f"{col}:")
    print(f"  Range: {df[col].min():.4f} to {df[col].max():.4f}")
    print(f"  Mean: {df[col].mean():.4f}")
    print(f"  Std: {df[col].std():.4f}")

# Check for correlations between interfacial recombination and efficiency
print("\nCorrelations between interfacial recombination and efficiency:")
if int_columns and efficiency_columns:
    corr_matrix = df[int_columns + efficiency_columns].corr()
    print("Correlation matrix:")
    print(corr_matrix)

# Look for J-V curve parameters
jv_params = ['Jsc', 'Voc', 'FF', 'PCE', 'Jmpp', 'Vmpp']
available_jv = [param for param in jv_params if param in df.columns]
print(f"\nAvailable J-V parameters: {available_jv}")

# Check data distribution
print(f"\nNumber of unique simulations: {df.shape[0]}")
print(f"Number of parameters per simulation: {df.shape[1]}")

print("\nData analysis complete!") 