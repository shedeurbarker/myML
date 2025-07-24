import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

print("=== SIMPLE VISUALIZATION OF EXTRACTED DATA ===")

# Create results directory
os.makedirs('results/visualize', exist_ok=True)

# Load the extracted data
print("Loading extracted data...")
df = pd.read_csv('results/extract/extracted_data.csv')
print(f"Data shape: {df.shape}")

# Calculate power output
df['Power'] = df['V'] * df['Jint']
df['Power_abs'] = np.abs(df['Power'])

# Create visualizations
print("Creating visualizations...")

# 1. Power vs Interfacial Recombination
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Interfacial Recombination Analysis', fontsize=16, fontweight='bold')

# Plot 1: Power vs Electron Recombination
ax1 = axes[0, 0]
sample_df = df.sample(n=1000, random_state=42)  # Sample for clarity
ax1.scatter(sample_df['IntSRHn'], sample_df['Power_abs'], alpha=0.6, s=10)
ax1.set_xlabel('Electron Recombination (IntSRHn)')
ax1.set_ylabel('Power Output (W/cm²)')
ax1.set_title('Power vs Electron Recombination')
ax1.set_yscale('log')
ax1.set_xscale('log')

# Plot 2: Power vs Hole Recombination
ax2 = axes[0, 1]
ax2.scatter(sample_df['IntSRHp'], sample_df['Power_abs'], alpha=0.6, s=10, color='orange')
ax2.set_xlabel('Hole Recombination (IntSRHp)')
ax2.set_ylabel('Power Output (W/cm²)')
ax2.set_title('Power vs Hole Recombination')
ax2.set_yscale('log')
ax2.set_xscale('log')

# Plot 3: Recombination Balance
ax3 = axes[1, 0]
scatter = ax3.scatter(sample_df['IntSRHn'], sample_df['IntSRHp'], 
                     c=sample_df['Power_abs'], cmap='viridis', alpha=0.6, s=10)
ax3.set_xlabel('Electron Recombination (IntSRHn)')
ax3.set_ylabel('Hole Recombination (IntSRHp)')
ax3.set_title('Recombination Balance\n(colored by power)')
ax3.set_xscale('log')
ax3.set_yscale('log')
plt.colorbar(scatter, ax=ax3, label='Power (W/cm²)')

# Plot 4: Power Distribution
ax4 = axes[1, 1]
ax4.hist(df['Power_abs'], bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
ax4.set_xlabel('Power Output (W/cm²)')
ax4.set_ylabel('Frequency')
ax4.set_title('Power Output Distribution')
ax4.set_xscale('log')

plt.tight_layout()
plt.savefig('results/visualize/interfacial_recombination_analysis.png', dpi=300, bbox_inches='tight')
print("Main analysis plot saved to: results/visualize/interfacial_recombination_analysis.png")

# 2. Create efficiency analysis
print("Creating efficiency analysis...")

# Group by recombination levels
df['Recombination_Level'] = pd.cut(df['IntSRHn'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

# Calculate statistics by level
efficiency_stats = df.groupby('Recombination_Level').agg({
    'Power_abs': ['mean', 'std', 'count']
}).round(4)

fig2, axes2 = plt.subplots(1, 2, figsize=(15, 6))
fig2.suptitle('Efficiency Analysis by Recombination Level', fontsize=16, fontweight='bold')

# Plot 1: Average power by recombination level
ax1 = axes2[0]
efficiency_stats[('Power_abs', 'mean')].plot(kind='bar', ax=ax1, color='skyblue')
ax1.set_xlabel('Recombination Level')
ax1.set_ylabel('Average Power (W/cm²)')
ax1.set_title('Average Power by Recombination Level')
ax1.set_yscale('log')
ax1.tick_params(axis='x', rotation=45)

# Add count labels
for i, (level, count) in enumerate(efficiency_stats[('Power_abs', 'count')].items()):
    ax1.text(i, efficiency_stats[('Power_abs', 'mean')][level], f'n={count:,}', 
             ha='center', va='bottom', fontsize=8)

# Plot 2: Power vs Voltage
ax2 = axes2[1]
ax2.scatter(sample_df['V'], sample_df['Power_abs'], alpha=0.6, s=10, c='red')
ax2.set_xlabel('Voltage (V)')
ax2.set_ylabel('Power Output (W/cm²)')
ax2.set_title('Power vs Voltage')
ax2.set_yscale('log')

plt.tight_layout()
plt.savefig('results/visualize/efficiency_analysis.png', dpi=300, bbox_inches='tight')
print("Efficiency analysis plot saved to: results/visualize/efficiency_analysis.png")

# 3. Create summary statistics
print("\n=== SUMMARY STATISTICS ===")
print(f"Total data points: {len(df):,}")
print(f"Power range: {df['Power_abs'].min():.2e} to {df['Power_abs'].max():.2e} W/cm²")
print(f"Average power: {df['Power_abs'].mean():.2e} W/cm²")
print(f"Median power: {df['Power_abs'].median():.2e} W/cm²")

# Find optimal ranges
optimal_threshold = df['Power_abs'].quantile(0.9)
optimal_data = df[df['Power_abs'] > optimal_threshold]

print(f"\nOptimal power threshold (top 10%): {optimal_threshold:.2e} W/cm²")
print(f"Optimal cases: {len(optimal_data):,}")

if len(optimal_data) > 0:
    print(f"\nOptimal recombination ranges (top 10% efficiency):")
    print(f"IntSRHn range: {optimal_data['IntSRHn'].min():.2e} to {optimal_data['IntSRHn'].max():.2e}")
    print(f"IntSRHp range: {optimal_data['IntSRHp'].min():.2e} to {optimal_data['IntSRHp'].max():.2e}")

print("\nVisualization complete! Check the plots in results/visualize/") 