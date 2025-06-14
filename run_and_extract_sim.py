import subprocess
import os

# Paths
possible_exes = [os.path.join('sim', 'simss.exe'), 'simss.exe']
sim_exe = None
sim_dir = None
for exe in possible_exes:
    if os.path.exists(exe):
        sim_exe = exe
        sim_dir = os.path.dirname(exe) if os.path.dirname(exe) else '.'
        break
if sim_exe is None:
    raise FileNotFoundError('simss.exe not found in sim/ or project root.')

jv_file = os.path.join(sim_dir, 'output_JV.dat')
pars_file = os.path.join(sim_dir, 'output_scPars.dat')
summary_file = os.path.join(sim_dir, 'summary_results.txt')

# Run the simulation
print('Running simulation...')
proc = subprocess.run([os.path.basename(sim_exe)], cwd=sim_dir)
if proc.returncode != 0:
    raise RuntimeError('Simulation failed!')
print('Simulation finished.')

# Extract JV curve (V, J)
jv_data = []
if os.path.exists(jv_file):
    with open(jv_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or not (line[0].isdigit() or line[0] == '-'):
                continue
            parts = line.split()
            v = float(parts[0])
            j = float(parts[1])
            jv_data.append((v, j))
else:
    print('Warning: JV file not found.')

# Extract solar cell parameters (Jsc, Voc, FF, MPP, etc.)
scpars = {}
if os.path.exists(pars_file):
    with open(pars_file, 'r') as f:
        header = f.readline()
        values = f.readline()
        keys = header.strip().split()
        vals = values.strip().split()
        for k, v in zip(keys, vals):
            scpars[k] = v
else:
    print('Warning: output_scPars.dat not found.')

# Save summary
with open(summary_file, 'w') as f:
    f.write('Simulated Solar Cell Results\n')
    f.write('============================\n')
    if scpars:
        for k, v in scpars.items():
            f.write(f'{k}: {v}\n')
    f.write('\nJV Curve (V, J):\n')
    for v, j in jv_data:
        f.write(f'{v:.4f}\t{j:.4f}\n')
print(f'Results saved to {summary_file}') 