"""
Runs libEnsemble to call the ytopt ask/tell interface in a generator function,
and the ytopt findRunTime interface in a simulator function.

Execute locally via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python run_ytopt_xsbench.py
   python run_ytopt_xsbench.py --nworkers 3 --comms local

The number of concurrent evaluations of the objective function will be 4-1=3.
"""

import os
import glob
import numpy as np
import itertools
import subprocess

import multiprocessing
multiprocessing.set_start_method('fork', force=True)

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
from libensemble.tools import parse_args, save_libE_output, add_unique_random_streams

from ytopt_asktell import persistent_ytopt  # Generator function, communicates with ytopt optimizer
from ytopt_obj import init_obj  # Simulator function, calls Plopper

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from ConfigSpace import ConfigurationSpace, EqualsCondition
from ytopt.search.optimizer import Optimizer

# Parse comms, default options from commandline
nworkers, is_manager, libE_specs, user_args_in = parse_args()
num_sim_workers = nworkers - 1  # Subtracting one because one worker will be the generator

assert len(user_args_in), "learner, etc. not specified, e.g. --learner RF"
#if not len(user_args_in):
#    import warnings
#    warnings.warn("LIBE DEBUG SETTINGS FOR LEARNER/MAX-EVALS USED")
#    user_args_in = ['--learner=RF', '--max-evals=3']
user_args = {}
for entry in user_args_in:
    if entry.startswith('--'):
        if '=' not in entry:
            key = entry.strip('--')
            value = user_args_in[user_args_in.index(entry)+1]
        else:
            split = entry.split('=')
            key = split[0].strip('--')
            value = split[1]

    user_args[key] = value

req_settings = ['learner','max-evals']
assert all([opt in user_args for opt in req_settings]), \
    "Required settings missing. Specify each setting in " + str(req_settings)


# Variables that will be sed-edited to control scaling
APP_SCALE = 64
NODE_SCALE = 4
cs = CS.ConfigurationSpace(seed=1234)
# arg1  precision
p0 = CSH.CategoricalHyperparameter(name='p0', choices=["double", "float"], default_value="float")
# arg2  3D array dimension size
p1 = CSH.Constant(name='p1', value=APP_SCALE)
#p1 = CSH.OrdinalHyperparameter(name='p1', sequence=[64,128,256,512,1024], default_value=128)
# arg3  reorder
p2 = CSH.CategoricalHyperparameter(name='p2', choices=["-no-reorder", "-reorder"," "], default_value=" ")
# arg4 alltoall
p3 = CSH.CategoricalHyperparameter(name='p3', choices=["-a2a", "-a2av", " "], default_value=" ")
# arg5 p2p
p4 = CSH.CategoricalHyperparameter(name='p4', choices=["-p2p", "-p2p_pl"," "], default_value=" ")
# arg6 reshape logic
p5 = CSH.CategoricalHyperparameter(name='p5', choices=["-pencils", "-slabs"," "], default_value=" ")
# arg7
p6 = CSH.CategoricalHyperparameter(name='p6', choices=["-r2c_dir 0", "-r2c_dir 1","-r2c_dir 2", " "], default_value=" ")
# arg8
p7 = CSH.UniformFloatHyperparameter(name='p7', lower=0, upper=1)
# arg9
p8 = CSH.UniformFloatHyperparameter(name='p8', lower=0, upper=1)
# number of threads is hardware-dependent


# Cross-architecture is out-of-scope for now so we determine this for the current platform and leave it at that
proc = subprocess.run(['nproc'], capture_output=True)
if proc.returncode == 0:
    max_cpus = int(proc.stdout.decode('utf-8').strip())
else:
    proc = subprocess.run(['lscpu'], capture_output=True)
    for line in proc.stdout.decode('utf-8'):
        if 'CPU(s):' in line:
            max_cpus = int(line.rstrip().rsplit(' ',1)[1])
            break
print(f"Detected {max_cpus} CPUs on this machine")

gpu_enabled = True
if gpu_enabled:
    proc = subprocess.run('nvidia-smi -L'.split(' '), capture_output=True)
    if proc.returncode != 0:
        raise ValueError("No GPUs Detected, but in GPU mode")
    gpus = len(proc.stdout.decode('utf-8').strip().split('\n'))
    print(f"Detected {gpus} GPUs on this machine")
    # Change exe.pl's PPN value (var named N_NODES) based on the system
    proc = subprocess.run(['sed', '-i',f's/N_NODES = [0-9]*/N_NODES = {gpus}/', 'exe.pl'], capture_output=True)
    if proc.returncode != 0:
        print(proc.stdout.decode('utf-8'))
        print(proc.stderr.decode('utf-8'))
        raise ValueError("sed Substitution for 'PPN' in 'exe.pl' Failed")
    else:
        print(f"Node identified to have access to {max_cpus} cpus and {gpus} gpus")
    PPN = gpus
else:
    PPN = 2
print(f"Set PPN to {PPN}"+"\n")


NODE_COUNT = NODE_SCALE // PPN
print(f"APP_SCALE (AKA Problem Size X, X, X) = {APP_SCALE} x3")
print(f"NODE_SCALE (AKA System Size X * Y = Z) = {NODE_COUNT} * {PPN} = {NODE_SCALE}")
# Don't exceed #threads across total ranks
max_depth = max_cpus // PPN
sequence = [2**_ for _ in range(1,10) if (2**_) <= max_depth]
if len(sequence) >= 2:
    intermediates = []
    prevpow = sequence[1]
    for rawpow in sequence[2:]:
        if rawpow+prevpow >= max_depth:
            break
        intermediates.append(rawpow+prevpow)
        prevpow = rawpow
    sequence = sorted(intermediates + sequence)
# Ensure max_depth is always in the list
if np.log2(max_depth)-int(np.log2(max_depth)) > 0:
    sequence = sorted(sequence+[max_depth])
print(f"Given {NODE_COUNT} nodes * {PPN} processes-per-node (={NODE_SCALE}) and {max_cpus} CPUS on this node...")
print(f"Selectable depths are: {sequence}"+"\n")
# arg10 number threads per MPI process
p9 = CSH.OrdinalHyperparameter(name='p9', sequence=sequence, default_value=max_depth)

cs.add_hyperparameters([p0, p1, p2, p3, p4, p5, p6, p7, p8, p9])

ytoptimizer = Optimizer(
    num_workers=num_sim_workers,
    space=cs,
    learner=user_args['learner'],
    liar_strategy='cl_max',
    acq_func='gp_hedge',
    set_KAPPA=1.96,
    set_SEED=2345,
    set_NI=10,
)

MACHINE_IDENTIFIER = "x3005c0s37b1n0"
print(f"Identifying machine as {MACHINE_IDENTIFIER}"+"\n")
MACHINE_INFO = {
    'identifier': MACHINE_IDENTIFIER,
    'mpi_ranks': NODE_SCALE,
    'ppn': PPN,
    'gpu_enabled': gpu_enabled,
    'libE_workers': num_sim_workers,
    'app_timeout': 30,
}

# Declare the sim_f to be optimized, and the input/outputs
sim_specs = {
    'sim_f': init_obj,
    'in': [f'p{_}' for _ in range(10)],
    'out': [('FLOPS', float, (1,)),
            ('elapsed_sec', float, (1,)),
            ('machine_identifier','<U30', (1,)),
            ('mpi_ranks', int, (1,)),
            ('ppn', int, (1,)),
            ('gpu_enabled', bool, (1,)),
            ('libE_id', int, (1,)),
            ('libE_workers', int, (1,)),],
    'user': {
        'machine_info': MACHINE_INFO,
    }
}

# Declare the gen_f that will generate points for the sim_f, and the various input/outputs
gen_specs = {
    'gen_f': persistent_ytopt,
    'out': [('p0', "<U24", (1,)),
            ('p1', int, (1,)),
            ('p2', "<U24", (1,)),
            ('p3', "<U24", (1,)),
            ('p4', "<U24", (1,)),
            ('p5', "<U24", (1,)),
            ('p6', "<U24", (1,)),
            ('p7', float, (1,)),
            ('p8', float, (1,)),
            ('p9', int, (1,))],
    'persis_in': sim_specs['in'] +\
                 ['FLOPS'] +\
                 ['elapsed_sec'] +\
                 ['machine_identifier'] +\
                 ['mpi_ranks'] +\
                 ['ppn'] +\
                 ['gpu_enabled'] +\
                 ['libE_id'] +\
                 ['libE_workers'],
    'user': {
        'ytoptimizer': ytoptimizer,  # provide optimizer to generator function
        'num_sim_workers': num_sim_workers,
        'machine_info': MACHINE_INFO,
    },
}

alloc_specs = {
    'alloc_f': alloc_f,
    'user': {'async_return': True},
}

# Specify when to exit. More options: https://libensemble.readthedocs.io/en/main/data_structures/exit_criteria.html
exit_criteria = {'sim_max': int(user_args['max-evals'])}

# Added as a workaround to issue that's been resolved on develop
persis_info = add_unique_random_streams({}, nworkers + 1)

# Set options so workers operate in unique directories
here = os.getcwd() + '/'
libE_specs['use_worker_dirs'] = True
libE_specs['sim_dirs_make'] = False  # Otherwise directories separated by each sim call
# Copy or symlink needed files into unique directories
libE_specs['sim_dir_symlink_files'] = [here + f for f in ['speed3d.sh', 'plopper.py', 'set_affinity_gpu_polaris.sh']]
ENSEMBLE_DIR_PATH = "Scaling_2n_4g_4w_2e_adef709e"
libE_specs['ensemble_dir_path'] = f'./ensemble_{ENSEMBLE_DIR_PATH}'
print(f"This ensemble operates as: {libE_specs['ensemble_dir_path']}"+"\n")

if __name__ == '__main__':
    # Perform the libE run
    H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info,
                                alloc_specs=alloc_specs, libE_specs=libE_specs)
    # Save History array to file
    if is_manager:
        # We may have missed the final evaluation in the results file
        print("\nlibEnsemble has completed evaluations.")
        import pandas as pd
        unfinished = H[~H["sim_ended"]][gen_specs['persis_in']]
        unfinished_log = pd.DataFrame(dict((k, unfinished[k].flatten()) for k in gen_specs['persis_in']))
        final_output = f"{libE_specs['ensemble_dir_path']}/unfinished_results.csv"
        if len(unfinished_log) == 0:
            print("All simulations finished.")
        else:
            unfinished_log.to_csv(final_output, index=False)
            print(f"{len(unfinished_log)} unfinished results logged to {final_output}")

        finished = H[H["sim_ended"]][gen_specs['persis_in']]
        full_log = pd.DataFrame(dict((k, finished[k].flatten()) for k in gen_specs['persis_in']))
        final_output = f"{libE_specs['ensemble_dir_path']}/results.csv"
        full_log.to_csv(final_output, index=False)
        print(f"All finished results logged to {final_output}")

