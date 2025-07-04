import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load a sample of the data to understand structure
print("Loading data sample...")
df_sample = pd.read_csv('results/fetch/combined_output.csv', nrows=1000)

print(f"Data shape: {df_sample.shape}")
print(f"Columns: {list(df_sample.columns)}")

# Find interfacial recombination related columns
int_columns = [col for col in df_sample.columns if 'Int' in col or 'SRH' in col]
print(f"\nInterfacial recombination columns: {int_columns}")

# Find efficiency related columns
efficiency_columns = [col for col in df_sample.columns if any(x in col.lower() for x in ['j', 'v', 'efficiency', 'power', 'current'])]
print(f"Efficiency related columns: {efficiency_columns}")

# Find layer parameter columns
layer_columns = [col for col in df_sample.columns if any(x in col for x in ['L1_', 'L2_', 'L3_'])]
print(f"Layer parameter columns: {layer_columns}")

# Check for unique values in interfacial recombination columns
print("\nUnique values in interfacial recombination columns:")
for col in int_columns:
    unique_vals = df_sample[col].unique()
    print(f"{col}: {len(unique_vals)} unique values, range: {unique_vals.min():.2e} to {unique_vals.max():.2e}")

# Check for unique values in layer parameters
print("\nUnique values in layer parameters:")
for col in layer_columns:
    unique_vals = df_sample[col].unique()
    print(f"{col}: {len(unique_vals)} unique values, range: {unique_vals.min():.2e} to {unique_vals.max():.2e}")

# Look for J-V curve data
jv_columns = [col for col in df_sample.columns if col in ['J', 'V', 'Jn', 'Jp', 'Jint']]
print(f"\nJ-V related columns: {jv_columns}")

# Check if we have voltage and current data
if 'V' in df_sample.columns and 'J' in df_sample.columns:
    print(f"\nVoltage range: {df_sample['V'].min():.3f} to {df_sample['V'].max():.3f} V")
    print(f"Current density range: {df_sample['J'].min():.2e} to {df_sample['J'].max():.2e} A/cm²")
    
    # Calculate power density
    df_sample['P'] = df_sample['V'] * df_sample['J']
    print(f"Power density range: {df_sample['P'].min():.2e} to {df_sample['P'].max():.2e} W/cm²")

print("\nData analysis complete!") 