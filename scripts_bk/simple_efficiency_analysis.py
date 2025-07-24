import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("=== SIMPLE EFFICIENCY ANALYSIS ===")
print("Understanding how interfacial recombination affects device efficiency\n")

# Load the data
print("Loading data...")
df = pd.read_csv('results/extract/extracted_data.csv')
print(f"Total data points: {len(df):,}\n")

# 1. What is interfacial recombination?
print("=== WHAT IS INTERFACIAL RECOMBINATION? ===")
print("Interfacial recombination happens when electrons and holes meet at layer boundaries")
print("and cancel each other out, reducing the current that can be used for power.\n")

print("We have two types:")
print("- IntSRHn: Electron recombination (how many electrons are lost)")
print("- IntSRHp: Hole recombination (how many holes are lost)\n")

# 2. How does it affect power output?
print("=== HOW DOES IT AFFECT POWER OUTPUT? ===")

# Calculate power (V × J)
df['Power'] = df['V'] * df['Jint']
df['Power_abs'] = np.abs(df['Power'])

print(f"Power output range: {df['Power_abs'].min():.2e} to {df['Power_abs'].max():.2e} W/cm²")

# Find best and worst cases
best_power = df['Power_abs'].max()
worst_power = df['Power_abs'].min()

best_case = df[df['Power_abs'] == best_power].iloc[0]
worst_case = df[df['Power_abs'] == worst_power].iloc[0]

print(f"\nBest case power: {best_power:.2e} W/cm²")
print(f"  - Electron recombination: {best_case['IntSRHn']:.2e}")
print(f"  - Hole recombination: {best_case['IntSRHp']:.2e}")

print(f"\nWorst case power: {worst_power:.2e} W/cm²")
print(f"  - Electron recombination: {worst_case['IntSRHn']:.2e}")
print(f"  - Hole recombination: {worst_case['IntSRHp']:.2e}\n")

# 3. Simple relationship analysis
print("=== SIMPLE RELATIONSHIP ANALYSIS ===")

# Group by recombination levels
df['Recombination_Level'] = pd.cut(df['IntSRHn'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

print("Average power output by recombination level:")
power_by_level = df.groupby('Recombination_Level')['Power_abs'].mean()
for level, power in power_by_level.items():
    print(f"  {level}: {power:.2e} W/cm²")

# 4. Create simple visualizations
print("\n=== CREATING SIMPLE PLOTS ===")

# Create a simple figure
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Simple Efficiency Analysis', fontsize=16, fontweight='bold')

# Plot 1: Power vs Recombination (simple scatter)
ax1 = axes[0, 0]
# Take a sample to avoid overcrowding
sample_df = df.sample(n=1000, random_state=42)
ax1.scatter(sample_df['IntSRHn'], sample_df['Power_abs'], alpha=0.6, s=10)
ax1.set_xlabel('Electron Recombination (IntSRHn)')
ax1.set_ylabel('Power Output (W/cm²)')
ax1.set_title('Power vs Electron Recombination')
ax1.set_yscale('log')
ax1.set_xscale('log')

# Plot 2: Power vs Recombination (averaged)
ax2 = axes[0, 1]
power_by_level.plot(kind='bar', ax=ax2, color='skyblue')
ax2.set_xlabel('Recombination Level')
ax2.set_ylabel('Average Power (W/cm²)')
ax2.set_title('Average Power by Recombination Level')
ax2.set_yscale('log')
ax2.tick_params(axis='x', rotation=45)

# Plot 3: Recombination balance
ax3 = axes[1, 0]
ax3.scatter(sample_df['IntSRHn'], sample_df['IntSRHp'], 
           c=sample_df['Power_abs'], cmap='viridis', alpha=0.6, s=10)
ax3.set_xlabel('Electron Recombination (IntSRHn)')
ax3.set_ylabel('Hole Recombination (IntSRHp)')
ax3.set_title('Recombination Balance\n(colored by power)')
ax3.set_xscale('log')
ax3.set_yscale('log')
plt.colorbar(ax3.collections[0], ax=ax3, label='Power (W/cm²)')

# Plot 4: Power distribution
ax4 = axes[1, 1]
ax4.hist(df['Power_abs'], bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
ax4.set_xlabel('Power Output (W/cm²)')
ax4.set_ylabel('Number of Cases')
ax4.set_title('Power Output Distribution')
ax4.set_xscale('log')

plt.tight_layout()
plt.savefig('results/extract/simple_efficiency_analysis.png', dpi=300, bbox_inches='tight')
print("Simple analysis plot saved to: results/extract/simple_efficiency_analysis.png")

# 5. Key insights
print("\n=== KEY INSIGHTS ===")
print("1. Too much recombination = Low power (electrons/holes cancel out)")
print("2. Too little recombination = Sometimes low power (not enough current flow)")
print("3. Optimal recombination = Sweet spot for maximum power")
print("4. Balance between electron and hole recombination is important")

# 6. Practical recommendations
print("\n=== PRACTICAL RECOMMENDATIONS ===")
print("For better device efficiency:")
print("- Don't just minimize recombination - find the optimal level")
print("- Balance electron and hole recombination")
print("- Consider how recombination changes with voltage")
print("- Focus on the interaction between recombination and current flow")

# 7. Simple summary
print("\n=== SIMPLE SUMMARY ===")
print(f"Total simulations analyzed: {len(df):,}")
print(f"Power range: {df['Power_abs'].min():.2e} to {df['Power_abs'].max():.2e} W/cm²")
print(f"Best power: {best_power:.2e} W/cm²")
print(f"Worst power: {worst_power:.2e} W/cm²")
print(f"Power improvement potential: {best_power/worst_power:.1f}x")

print("\nAnalysis complete! Check the plot for visual understanding.") 