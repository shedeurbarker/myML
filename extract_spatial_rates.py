import numpy as np
import matplotlib.pyplot as plt

# Path to the output_Var.dat file
var_file_path = 'sim/output_Var.dat'

# Read the data, skipping the header
data = np.loadtxt(var_file_path, skiprows=1)

# Extract columns
x = data[:, 0]  # Position
IntSRHn = data[:, 6]  # Interface SRH recombination rate for electrons

# Identify the active/ETL interface position (e.g., at grid point 100)
active_etl_interface_index = 100  # Adjust this index based on your device structure

# Define the layer between active and ETL (e.g., 50 grid points)
layer_size = 50
start_index = active_etl_interface_index
end_index = min(len(x), active_etl_interface_index + layer_size)

# Extract the electron recombination rate for the layer between active and ETL
layer_x = x[start_index:end_index]
layer_IntSRHn = IntSRHn[start_index:end_index]

# Plot the electron recombination rate for the layer between active and ETL
plt.figure(figsize=(10, 6))
plt.plot(layer_x, layer_IntSRHn, 'o-', label='IntSRHn (electrons)')
plt.xlabel('Position (m)')
plt.ylabel('Recombination Rate (m⁻³ s⁻¹)')
plt.title('Electron Recombination Rate in the Layer Between Active and ETL')
plt.legend()
plt.grid(True)
plt.savefig('electron_recombination_active_etl_layer.png')
plt.show()