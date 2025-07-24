import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Load the ML-ready data
print("Loading ML-ready data...")
df = pd.read_csv('results/extract/extracted_data.csv')

print(f"Data shape: {df.shape}")

# Create figure with subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Efficiency Analysis with Respect to Interfacial Recombination', fontsize=16, fontweight='bold')

# 1. Power vs Interfacial Recombination
print("Creating power vs interfacial recombination plot...")
ax1 = axes[0, 0]
scatter1 = ax1.scatter(df['log_IntSRHn'], df['P_abs'], 
                      c=df['log_IntSRHp'], cmap='viridis', alpha=0.6, s=1)
ax1.set_xlabel('Log10(IntSRHn)')
ax1.set_ylabel('Power Density (W/cm²)')
ax1.set_title('Power vs Electron Interfacial Recombination')
ax1.set_yscale('log')
plt.colorbar(scatter1, ax=ax1, label='Log10(IntSRHp)')

# 2. Current Ratio vs Interfacial Recombination
print("Creating current ratio vs interfacial recombination plot...")
ax2 = axes[0, 1]
scatter2 = ax2.scatter(df['log_IntSRHn'], df['J_ratio'], 
                      c=df['log_IntSRHp'], cmap='plasma', alpha=0.6, s=1)
ax2.set_xlabel('Log10(IntSRHn)')
ax2.set_ylabel('Current Ratio (Jn/Jp)')
ax2.set_title('Current Ratio vs Interfacial Recombination')
ax2.set_yscale('symlog')
plt.colorbar(scatter2, ax=ax2, label='Log10(IntSRHp)')

# 3. Voltage Efficiency vs Interfacial Recombination
print("Creating voltage efficiency vs interfacial recombination plot...")
ax3 = axes[0, 2]
scatter3 = ax3.scatter(df['log_IntSRHn'], df['V_efficiency'], 
                      c=df['log_IntSRHp'], cmap='coolwarm', alpha=0.6, s=1)
ax3.set_xlabel('Log10(IntSRHn)')
ax3.set_ylabel('Voltage Efficiency (V/Vext)')
ax3.set_title('Voltage Efficiency vs Interfacial Recombination')
plt.colorbar(scatter3, ax=ax3, label='Log10(IntSRHp)')

# 4. Interfacial Recombination Balance
print("Creating interfacial recombination balance plot...")
ax4 = axes[1, 0]
scatter4 = ax4.scatter(df['log_IntSRHn'], df['log_IntSRHp'], 
                      c=df['P_abs'], cmap='hot', alpha=0.6, s=1)
ax4.set_xlabel('Log10(IntSRHn)')
ax4.set_ylabel('Log10(IntSRHp)')
ax4.set_title('Interfacial Recombination Balance\n(colored by power)')
plt.colorbar(scatter4, ax=ax4, label='Power Density (W/cm²)')

# 5. Energy Gap vs Interfacial Recombination
print("Creating energy gap vs interfacial recombination plot...")
ax5 = axes[1, 1]
scatter5 = ax5.scatter(df['energy_gap_L2'], df['log_IntSRHn'], 
                      c=df['P_abs'], cmap='spring', alpha=0.6, s=1)
ax5.set_xlabel('Energy Gap L2 (eV)')
ax5.set_ylabel('Log10(IntSRHn)')
ax5.set_title('Energy Gap vs Interfacial Recombination\n(colored by power)')
plt.colorbar(scatter5, ax=ax5, label='Power Density (W/cm²)')

# 6. Thickness vs Interfacial Recombination
print("Creating thickness vs interfacial recombination plot...")
ax6 = axes[1, 2]
scatter6 = ax6.scatter(df['total_thickness'], df['log_IntSRHn'], 
                      c=df['P_abs'], cmap='winter', alpha=0.6, s=1)
ax6.set_xlabel('Total Thickness (m)')
ax6.set_ylabel('Log10(IntSRHn)')
ax6.set_title('Device Thickness vs Interfacial Recombination\n(colored by power)')
plt.colorbar(scatter6, ax=ax6, label='Power Density (W/cm²)')

plt.tight_layout()
plt.savefig('results/extract/efficiency_curves_analysis.png', dpi=300, bbox_inches='tight')
print("Efficiency curves plot saved to: results/extract/efficiency_curves_analysis.png")

# Create additional analysis plots
fig2, axes2 = plt.subplots(2, 2, figsize=(15, 12))
fig2.suptitle('Detailed Efficiency Analysis', fontsize=16, fontweight='bold')

# 7. Efficiency distribution by recombination level
print("Creating efficiency distribution plots...")
ax7 = axes2[0, 0]
df['IntSRH_level'] = pd.cut(df['log_IntSRHn'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
efficiency_stats = df.groupby('IntSRH_level')['P_abs'].agg(['mean', 'std', 'count']).reset_index()
bars = ax7.bar(efficiency_stats['IntSRH_level'], efficiency_stats['mean'], 
               yerr=efficiency_stats['std'], capsize=5, alpha=0.7)
ax7.set_xlabel('Interfacial Recombination Level')
ax7.set_ylabel('Average Power Density (W/cm²)')
ax7.set_title('Average Power by Recombination Level')
ax7.set_yscale('log')
ax7.tick_params(axis='x', rotation=45)

# Add count labels on bars
for bar, count in zip(bars, efficiency_stats['count']):
    height = bar.get_height()
    ax7.text(bar.get_x() + bar.get_width()/2., height,
             f'n={count:,}', ha='center', va='bottom', fontsize=8)

# 8. Correlation heatmap
print("Creating correlation heatmap...")
ax8 = axes2[0, 1]
key_features = ['log_IntSRHn', 'log_IntSRHp', 'P_abs', 'J_ratio', 'V_efficiency', 
                'total_thickness', 'energy_gap_L2']
corr_matrix = df[key_features].corr()
im = ax8.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax8.set_xticks(range(len(key_features)))
ax8.set_yticks(range(len(key_features)))
ax8.set_xticklabels(key_features, rotation=45, ha='right')
ax8.set_yticklabels(key_features)
ax8.set_title('Feature Correlation Matrix')

# Add correlation values
for i in range(len(key_features)):
    for j in range(len(key_features)):
        text = ax8.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                       ha="center", va="center", color="black", fontsize=8)

plt.colorbar(im, ax=ax8, label='Correlation Coefficient')

# 9. Power density distribution
print("Creating power density distribution plot...")
ax9 = axes2[1, 0]
ax9.hist(df['P_abs'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
ax9.set_xlabel('Power Density (W/cm²)')
ax9.set_ylabel('Frequency')
ax9.set_title('Power Density Distribution')
ax9.set_xscale('log')

# 10. Optimal recombination zones
print("Creating optimal recombination zones plot...")
ax10 = axes2[1, 1]
# Find high-efficiency regions (top 10% power)
high_efficiency = df[df['P_abs'] > df['P_abs'].quantile(0.9)]
scatter10 = ax10.scatter(df['log_IntSRHn'], df['log_IntSRHp'], 
                        c='lightgray', alpha=0.3, s=1, label='All data')
scatter10_high = ax10.scatter(high_efficiency['log_IntSRHn'], high_efficiency['log_IntSRHp'], 
                             c='red', alpha=0.7, s=2, label='Top 10% efficiency')
ax10.set_xlabel('Log10(IntSRHn)')
ax10.set_ylabel('Log10(IntSRHp)')
ax10.set_title('Optimal Interfacial Recombination Zones')
ax10.legend()

plt.tight_layout()
plt.savefig('results/extract/detailed_efficiency_analysis.png', dpi=300, bbox_inches='tight')
print("Detailed analysis plot saved to: results/extract/detailed_efficiency_analysis.png")

# Print summary statistics
print("\n=== EFFICIENCY OPTIMIZATION INSIGHTS ===")
print(f"Total data points analyzed: {len(df):,}")
print(f"Power density range: {df['P_abs'].min():.2e} to {df['P_abs'].max():.2e} W/cm²")
print(f"Optimal power threshold (top 10%): {df['P_abs'].quantile(0.9):.2e} W/cm²")

# Find optimal recombination ranges
optimal_data = df[df['P_abs'] > df['P_abs'].quantile(0.9)]
print(f"\nOptimal interfacial recombination ranges (top 10% efficiency):")
print(f"IntSRHn range: {optimal_data['log_IntSRHn'].min():.2f} to {optimal_data['log_IntSRHn'].max():.2f}")
print(f"IntSRHp range: {optimal_data['log_IntSRHp'].min():.2f} to {optimal_data['log_IntSRHp'].max():.2f}")

print("\nVisualization complete!") 