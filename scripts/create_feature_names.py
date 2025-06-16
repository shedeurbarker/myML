import numpy as np

# Define the features to be used
# feature_names = [
#     'x', 'V', 'Evac', 'Ec', 'Ev', 'phin', 'phip', 'n', 'p', 'ND', 'NA',
#     'anion', 'cation', 'ntb', 'nti', 'mun', 'mup', 'G_ehp', 'Gfree', 'Rdir',
#     'Jn', 'Jp', 'Jint', 'lid', 'Vext', 'left_L', 'left_eps_r', 'left_E_c',
#     'left_E_v', 'left_N_c', 'left_mu_n', 'left_mu_p', 'left_N_t_int',
#     'left_C_n_int', 'left_C_p_int', 'left_E_t_int', 'right_L', 'right_eps_r',
#     'right_E_c', 'right_E_v', 'right_N_c', 'right_mu_n', 'right_mu_p',
#     'right_N_t_int', 'right_C_n_int', 'right_C_p_int', 'right_E_t_int'
# ]

feature_names = [
    'p', 'n',
    'cation', 'anion',
    'phip', 'phin', 
    'mun', 'mup',
    'Ec', 'Ev',
    'Jp', 'Jn',
    'Gfree', 'G_ehp', 'Jint', 'Vext', 'Evac', 'V', 'Rdir', 'NA',
    'left_mu_n', 'right_mu_n',
    'left_eps_r', 'right_eps_r',
    'left_mu_p', 'right_mu_p',
    'left_L', 'right_L',
    'left_E_c', 'right_E_c',
    'left_E_v', 'right_E_v',
    'left_N_t_int', 'right_N_t_int',
    'left_N_c', 'right_N_c'
]

# feature_names = [
#     'p', 'n',
#     'cation', 'anion',
#     'phip', 'phin', 
#     'mun', 'mup',
#     'Ec', 'Ev',
#     'Jp', 'Jn',
#     'Gfree', 'G_ehp', 'Jint', 'Vext', 'Evac', 'V', 'Rdir', 'NA',
#     'left_mu_n', 'right_mu_n',
#     'left_eps_r', 'right_eps_r',
#     'left_mu_p', 'right_mu_p',
#     'left_L', 'right_L',
#     'left_E_c', 'right_E_c',
#     'left_E_v', 'right_E_v',
#     'left_N_t_int', 'right_N_t_int',
#     'left_N_c', 'right_N_c'
# ]




# Save the feature names to a .npy file
np.save('scripts/feature_names.npy', feature_names)

print("Feature names saved to scripts/feature_names.npy")

# example_data = pd.DataFrame({
#         'x': [3.97265e-08],
#         'V': [-0.61865],
#         'Evac': [1.21865],
#         'Ec': [-3.28135],
#         'Ev': [-4.48135],
#         'phin': [-3.30357],
#         'phip': [-4.1167],
#         'n': [9.18056e+23],
#         'p': [1.297e+18],
#         'ND': [1e+21],
#         'NA': [3.16e+20],
#         'anion': [2.37984e+22],
#         'cation': [1.41364e+19],
#         'ntb': [9.99081e+20],
#         'nti': [0],
#         'mun': [0.0001],
#         'mup': [0.0001],
#         'G_ehp': [1.27696e+28],
#         'Gfree': [1.27696e+28],
#         'Rdir': [1.19422e+25],
#         'Jn': [-18.3825],
#         'Jp': [3.27136],
#         'Jint': [-15.1111],
#         'lid': [2],
#         'Vext': [-0.05],
#         'left_L': [2.5e-08],
#         'left_eps_r': [5],
#         'left_E_c': [4],
#         'left_E_v': [5.9],
#         'left_N_c': [5e+26],
#         'left_mu_n': [1e-06],
#         'left_mu_p': [1e-09],
#         'left_N_t_int': [4e+12],
#         'left_C_n_int': [2e-14],
#         'left_C_p_int': [2e-14],
#         'left_E_t_int': [4.7],
#         'right_L': [5e-07],
#         'right_eps_r': [24],
#         'right_E_c': [3.9],
#         'right_E_v': [5.53],
#         'right_N_c': [2.2e+24],
#         'right_mu_n': [0.0001],
#         'right_mu_p': [0.0001],
#         'right_N_t_int': [1e+12],
#         'right_C_n_int': [2e-14],
#         'right_C_p_int': [2e-14],
#         'right_E_t_int': [4.7]
#     })