import matplotlib.pyplot as plt
import os

# Try both possible paths for the JV data file
possible_files = ['sim/output_JV.dat', 'output_JV.dat']
filename = None
for f in possible_files:
    if os.path.exists(f):
        filename = f
        break
if filename is None:
    raise FileNotFoundError('output_JV.dat not found in sim/ or project root.')

voltages = []
currents = []
with open(filename, 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        if not line or not (line[0].isdigit() or line[0] == '-'):  # skip headers and blanks
            continue
        parts = line.split()
        voltages.append(float(parts[0]))
        currents.append(float(parts[1]))

plt.figure(figsize=(8,6))
plt.plot(voltages, currents, marker='o', color='blue', label='JV Curve')
plt.xlabel('Voltage (V)', fontsize=14)
plt.ylabel('Current Density (A/m²)', fontsize=14)
plt.title('Simulated JV Curve', fontsize=16, fontweight='bold')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('simulated_jv_curve.png', dpi=300)
plt.show() 