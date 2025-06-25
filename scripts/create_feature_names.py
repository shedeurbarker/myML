import numpy as np

# feature_names = [
#     # simulation parameters
#     'p', 
#     'n',
#     'cation',
#     'anion',
#     'phip', 
#     'phin', 
#     'mun',
#     'mup',
#     'Ec', 
#     'Ev',
#     'Jp', 
#     'Jn',
#     # interface parameters
#     'Gfree', 
#     'G_ehp', 
#     'Jint', 
#     'Vext', 
#     'Evac', 
#     'V', 
#     'Rdir',
#     'NA',
#     # device parameters
#     'left_mu_n',
#     'right_mu_n',
#     'left_eps_r',
#     'right_eps_r',
#     'left_mu_p', 
#     'right_mu_p',
#     'left_L', 
#     'right_L',
#     'left_E_c', 
#     'right_E_c',
#     'left_E_v', 
#     'right_E_v',
#     'left_N_t_int', 
#     'right_N_t_int',
#     'left_N_c', 
#     'right_N_c'
# ]


feature_names = [
    # simulation parameters
    'p',         # hole concentration
    'n',         # electron concentration
    'mun',       # electron mobility
    'mup',       # hole mobility
    'Ec',        # electron energy
    'Ev',        # hole energy
    'NA',        # acceptor concentration
    'ND',        # donor concentration
    
    # left side
    'left_mu_n', # electron mobility
    'left_mu_p', # hole mobility
    'left_L',    # layer thickness
    'left_E_c',  # conduction band energy
    'left_E_v',  # valence band energy
    'left_N_t_int', # left interface trap density
    'left_N_c', # DOS at the left electrode
    
    # right side
    'right_mu_n', # electron mobility
    'right_mu_p', # hole mobility
    'right_L',    # layer thickness
    'right_E_c',  # conduction band energy
    'right_E_v',  # valence band energy
    'right_N_t_int', # right interface trap density
    'right_N_c', # DOS at the right electrode
]

# define the meaning of the feature names
feature_meanings = {
    'p': 'hole concentration',
    'n': 'electron concentration',
#    'cation': 'cation concentration',
#    'anion': 'anion concentration',
#    'phip': 'hole potential',
#    'phin': 'electron potential',
    'mun': 'electron mobility',
    'mup': 'hole mobility',
    'Ec': 'electron energy',
    'Ev': 'hole energy',
#    'Jp': 'hole current density',
#    'Jn': 'electron current density',
#    'Gfree': 'free energy',
#    'G_ehp': 'ehp energy',
#    'Jint': 'interface current density',
    'Vext': 'external voltage',
#    'Evac': 'vacuum energy',
    'V': 'voltage',
#    'Rdir': 'recombination rate',
    'NA': 'acceptor concentration',
    'left_mu_n': 'left electron mobility',
    'right_mu_n': 'right electron mobility',
#    'left_eps_r': 'left dielectric constant',
#    'right_eps_r': 'right dielectric constant',
    'left_mu_p': 'left hole mobility',
    'right_mu_p': 'right hole mobility',
    'left_L': 'left layer thickness',
    'right_L': 'right layer thickness',
    'left_E_c': 'left conduction band energy',
    'right_E_c': 'right conduction band energy',
    'left_E_v': 'left valence band energy',
    'right_E_v': 'right valence band energy'
}

# Save the feature names to a .npy file
np.save('scripts/feature_names.npy', feature_names)

print("Feature names saved to scripts/feature_names.npy")