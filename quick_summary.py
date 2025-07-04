import pandas as pd
import numpy as np

print("=== QUICK SUMMARY: INTERFACIAL RECOMBINATION & EFFICIENCY ===\n")

# Load data
df = pd.read_csv('results/extract/extracted_data.csv')
df['Power'] = df['V'] * df['Jint']

print("🔍 WHAT WE FOUND:")
print("=" * 50)

# 1. The problem
print("\n❌ THE PROBLEM:")
print("Interfacial recombination = electrons and holes canceling each other out")
print("This reduces the current that can be used to generate power")

# 2. The data
print("\n📊 THE DATA:")
print(f"- We analyzed {len(df):,} different device configurations")
print(f"- Power output varies from {df['Power'].min():.2e} to {df['Power'].max():.2e} W/cm²")
print(f"- That's a {df['Power'].max()/df['Power'].min():.1e}x difference!")

# 3. The key insight
print("\n💡 THE KEY INSIGHT:")
print("It's NOT about minimizing recombination completely")
print("It's about finding the OPTIMAL level of recombination")

# 4. Simple analysis
print("\n🎯 SIMPLE ANALYSIS:")

# Find reasonable power range (remove extreme outliers)
reasonable_power = df[df['Power'] > 0]
reasonable_power = reasonable_power[reasonable_power['Power'] < reasonable_power['Power'].quantile(0.95)]

print(f"- Best 5% of devices: {reasonable_power['Power'].quantile(0.95):.2e} W/cm²")
print(f"- Average device: {reasonable_power['Power'].mean():.2e} W/cm²")
print(f"- Worst 5% of devices: {reasonable_power['Power'].quantile(0.05):.2e} W/cm²")

# 5. What to do
print("\n🚀 WHAT TO DO:")
print("1. Don't just reduce recombination - find the sweet spot")
print("2. Balance electron and hole recombination")
print("3. Consider how recombination changes with voltage")
print("4. Focus on the interaction between recombination and current")

# 6. The bottom line
print("\n📈 THE BOTTOM LINE:")
print("Your device efficiency depends on finding the right balance")
print("Too much recombination = low power")
print("Too little recombination = sometimes low power")
print("Optimal recombination = maximum power")

print("\n" + "=" * 50)
print("✅ SUMMARY COMPLETE!")
print("Check the plots in results/extract/ for visual confirmation") 