import numpy as np # type: ignore

# Updated feature names to match actual columns in extracted_data.csv
feature_names = [
    # Basic physics parameters (available in extracted data)
    'p',         # hole concentration
    'n',         # electron concentration
    'mun',       # electron mobility
    'mup',       # hole mobility
    'Ec',        # electron energy
    'Ev',        # hole energy
    'NA',        # acceptor concentration
    'ND',        # donor concentration
    #'V',         # voltage
    'Vext',      # external voltage
    
    # Interface parameters (available in extracted data)
    'left_L',    # left layer thickness
    'left_E_c',  # left conduction band energy
    'left_E_v',  # left valence band energy
    'left_N_A',  # left acceptor concentration
    'left_N_D',  # left donor concentration
    
    'right_L',   # right layer thickness
    'right_E_c', # right conduction band energy
    'right_E_v', # right valence band energy
    'right_N_A', # right acceptor concentration
    'right_N_D', # right donor concentration
]

# Save the feature names to a .npy file
np.save('scripts/feature_names.npy', feature_names)

print("Updated feature names saved to scripts/feature_names.npy")
print(f"Number of features: {len(feature_names)}")
print("Features:", feature_names)