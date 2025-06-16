import numpy as np

feature_names = [
    # simulation parameters
    'p', 'n',
    'cation', 'anion',
    'phip', 'phin', 
    'mun', 'mup',
    'Ec', 'Ev',
    'Jp', 'Jn',
    # interface parameters
    'Gfree', 'G_ehp', 
    'Jint', 'Vext', 
    'Evac', 'V', 
    'Rdir', 'NA',
    # device parameters
    'left_mu_n', 'right_mu_n',
    'left_eps_r', 'right_eps_r',
    'left_mu_p', 'right_mu_p',
    'left_L', 'right_L',
    'left_E_c', 'right_E_c',
    'left_E_v', 'right_E_v',
    'left_N_t_int', 'right_N_t_int',
    'left_N_c', 'right_N_c'
]

# Save the feature names to a .npy file
np.save('scripts/feature_names.npy', feature_names)

print("Feature names saved to scripts/feature_names.npy")