"""
Runs libEnsemble to call the GC_TLA interface in a generator function,
and the GC_TLA interface in a simulator function.

Execute locally via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python run_gctla.py
   python run_gctla.py --nworkers 3 --comms local

The number of concurrent evaluations of the objective function will be 4-1=3.
"""

import os
import glob
import numpy as np
NUMPY_SEED = 1
np.random.seed(NUMPY_SEED)
import pandas as pd
import itertools
import subprocess

import multiprocessing
multiprocessing.set_start_method('fork', force=True)

# Import libEnsemble items for this test
from libensemble.libE import libE
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
from libensemble.tools import parse_args, save_libE_output, add_unique_random_streams
from libensemble import logger
logger.set_level("DEBUG") # Ensure logs are worth reading

from GC_TLA.libE_asktell import persistent_model  # Generator function, communicates with GC model
from GC_TLA.libE_obj import init_obj  # Simulator function, calls Problem

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from ConfigSpace import ConfigurationSpace, EqualsCondition
from sdv.single_table import GaussianCopulaSynthesizer as GaussianCopula
from sdv.metadata import SingleTableMetadata
from sdv.sampling.tabular import Condition
from sdv.constraints import ScalarRange

# Parse comms, default options from commandline
nworkers, is_manager, libE_specs, user_args_in = parse_args()
num_sim_workers = nworkers - 1  # Subtracting one because one worker will be the generator

# Refer to run_ytopt version if user args are needed for run_gctla
user_args = {}
#user_args.update({'max-evals': 10})

# Memo-ize whenever an argument can start and capstone end with length of the list
start_arg_idxs = [_ for _, e in enumerate(user_args_in) if e.startswith('--')]+[len(user_args_in)]
for meta_idx, idx in enumerate(start_arg_idxs[:-1]):
    entry = user_args_in[idx]
    if '=' in entry:
        split = entry.split('=')
        key = split[0].lstrip('--')
        value = split[1]
    else:
        # If = is not used, may have a list of arguments that follow
        key = entry.lstrip('--')
        until_index = start_arg_idxs[meta_idx+1] # Until start of next argument (or end of the list)
        value = user_args_in[idx+1:until_index]
        # One-element lists should just be their value (as if using the '=' operator)
        if len(value) == 1:
            value = value[0]
    user_args[key] = value

req_settings = ['max-evals', 'input', 'constraint-sys', 'constraint-app']
assert all([opt in user_args for opt in req_settings]), \
        f"Required settings missing: {set(req_settings).difference(set(user_args.keys()))}."+"\n"+\
        f"Specify each setting in {req_settings}"

# Type-fixing for args
int_args = ['max-evals', 'constraint-sys', 'constraint-app']
for arg in int_args:
    user_args[arg] = int(user_args[arg])

# Variables that will be sed-edited to control scaling
APP_SCALE = 1024
MPI_RANKS = 64
# SEEDING
CONFIGSPACE_SEED = 1234
YTOPT_SEED = 2345

# Set options so workers operate in unique directories
here = os.getcwd() + '/'
libE_specs['use_worker_dirs'] = True
libE_specs['sim_dirs_make'] = False  # Otherwise directories separated by each sim call
# Copy or symlink needed files into unique directories
libE_specs['sim_dir_symlink_files'] = [here + f for f in ['speed3d.sh', 'speed3d_no_gpu_aware.sh', 'gpu_cleanup.sh', 'plopper.py', 'set_affinity_gpu_polaris.sh']]
ENSEMBLE_DIR_PATH = ""
libE_specs['ensemble_dir_path'] = f'./ensemble_{ENSEMBLE_DIR_PATH}'
#if you need to manually specify resource information, ie:
#    libE_specs['resource_info'] = {'cores_on_node': (64,256), 'gpus_on_node': 0}
print(f"This ensemble operates as: {libE_specs['ensemble_dir_path']}"+"\n")

# Load model
cs = CS.ConfigurationSpace(seed=CONFIGSPACE_SEED)
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
p9 = CSH.UniformFloatHyperparameter(name='p9', lower=0, upper=1)

# Cross-architecture is out-of-scope for now so we determine this for the current platform and leave it at that
cpu_override = None
gpu_enabled = False
cpu_ranks_per_node = 1

c0 = CSH.Constant('c0', value='cufft' if gpu_enabled else 'fftw')

if cpu_override is None:
    proc = subprocess.run(['nproc'], capture_output=True)
    if proc.returncode == 0:
        threads_per_node = int(proc.stdout.decode('utf-8').strip())
    else:
        proc = subprocess.run(['lscpu'], capture_output=True)
        for line in proc.stdout.decode('utf-8'):
            if 'CPU(s):' in line:
                threads_per_node = int(line.rstrip().rsplit(' ',1)[1])
                break
    print(f"Detected {threads_per_node} CPU threads on this machine")
else:
    threads_per_node = cpu_override
    print(f"Override indicates {threads_per_node} CPU threads on this machine")
if cpu_ranks_per_node is None:
    cpu_ranks_per_node = threads_per_node
if gpu_enabled:
    proc = subprocess.run('nvidia-smi -L'.split(' '), capture_output=True)
    if proc.returncode != 0:
        raise ValueError("No GPUs Detected, but in GPU mode")
    gpus = len(proc.stdout.decode('utf-8').strip().split('\n'))
    print(f"Detected {gpus} GPUs on this machine")
    ranks_per_node = gpus
else:
    ranks_per_node = cpu_ranks_per_node
    print(f"CPU mode; force {ranks_per_node} ranks per node")
print(f"Set ranks_per_node to {ranks_per_node}"+"\n")

NODE_COUNT = max(MPI_RANKS // ranks_per_node,1)
print(f"APP_SCALE (AKA Problem Size X, X, X) = {APP_SCALE} x3")
print(f"MPI_RANKS (AKA System Size X * Y = Z) = {NODE_COUNT} * {ranks_per_node} = {MPI_RANKS}")
# Don't exceed #threads across total ranks
max_depth = threads_per_node // ranks_per_node
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
if max_depth not in sequence:
    sequence = sorted(sequence+[max_depth])
print(f"Depths are based on {threads_per_node} threads on each node, shared across {ranks_per_node} MPI ranks on each node")
print(f"Selectable depths are: {sequence}"+"\n")
# arg10 number threads per MPI process
#p9 = CSH.OrdinalHyperparameter(name='p9', sequence=sequence, default_value=max_depth)

cs.add_hyperparameters([p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, c0])

MACHINE_IDENTIFIER = "tbd"
print(f"Identifying machine as {MACHINE_IDENTIFIER}"+"\n")
MACHINE_INFO = {
    'identifier': MACHINE_IDENTIFIER,
    'mpi_ranks': MPI_RANKS,
    'threads_per_node': threads_per_node,
    'ranks_per_node': ranks_per_node,
    'gpu_enabled': gpu_enabled,
    'libE_workers': num_sim_workers,
    'app_timeout': 300,
    'sequence': sequence,
}

# Load data
data = pd.concat([pd.read_csv(_) for _ in user_args['input']])
data_trimmed = data[['c0',]+[f'p{_}' for _ in range(10)]+['mpi_ranks']]
# Create model
conditions = [Condition({'mpi_ranks': user_args['constraint-sys'],
                         'p1': user_args['constraint-app']},
                        num_rows=100)] #max(100, user_args['max-evals']))]
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data_trimmed)
constraints = [{'constraint_class': 'ScalarRange', # App scale limit
                    'constraint_parameters': {
                        'column_name': 'p1',
                        'low_value': 64,
                        'high_value': 2048,
                        'strict_boundaries': False,},
                    },
               {'constraint_class': 'ScalarRange', # System scale limit
                    'constraint_parameters': {
                        'column_name': 'mpi_ranks',
                        'low_value': 1,
                        'high_value': 16384,
                        'strict_boundaries': False,},
                    },
              ]
model = GaussianCopula(metadata, enforce_min_max_values=False)
model.add_constraints(constraints=constraints)
model.fit(data_trimmed)

# Fetch problem instance and set its space based on alterations
import gc_tla_problem
app_scale_name = gc_tla_problem.lookup_ival[(NODE_COUNT, APP_SCALE)]
problem = getattr(gc_tla_problem, app_scale_name) #f"{app_scale_name}_{NODE_COUNT}")
problem.plopper.set_architecture_info(threads_per_node = ranks_per_node,
                                      gpus = ranks_per_node if gpu_enabled else 0,
                                      nodes = NODE_COUNT,
                                      mpi_ranks = MPI_RANKS,
                                      machine_identifier = MACHINE_IDENTIFIER,
                                      )
problem.set_space(cs)
from copy import deepcopy
# We do this to ensure each problem can be referenced with a separate plopper that separately
# points to its own tmp_file directory much like the ytopt version, but this doesn't require
# me to re-initialize the object for every single iteration
problems = dict()
for simulatorID in range(2, 2+num_sim_workers):
    dir_adjusted_problem = deepcopy(problem)
    dir_adjusted_problem.plopper.outputdir = "tmp_files"
    print(dir_adjusted_problem.plopper.outputdir)
    problems[simulatorID] = dir_adjusted_problem

# Declare the sim_f to be optimized, and the input/outputs
sim_specs = {
    'sim_f': init_obj,
    'in': [f'p{_}' for _ in range(10)] + ['c0'],
    'out': [('FLOPS', float, (1,)),
            ('elapsed_sec', float, (1,)),
            ('machine_identifier','<U30', (1,)),
            ('mpi_ranks', int, (1,)),
            ('threads_per_node', int, (1,)),
            ('ranks_per_node', int, (1,)),
            ('gpu_enabled', bool, (1,)),
            ('libE_id', int, (1,)),
            ('libE_workers', int, (1,)),],
    'user': {
        'machine_info': MACHINE_INFO,
        'problem': problems,
    }
}

# Declare the gen_f that will generate points for the sim_f, and the various input/outputs
gen_specs = {
    'gen_f': persistent_model,
    'out': [
            # MUST MATCH ORDER OF THE CONFIGSPACE HYPERPARAMETERS EXACTLY
            ('c0', "<U24", (1,)),
            ('p0', "<U24", (1,)),
            ('p1', int, (1,)),
            ('p2', "<U24", (1,)),
            ('p3', "<U24", (1,)),
            ('p4', "<U24", (1,)),
            ('p5', "<U24", (1,)),
            ('p6', "<U24", (1,)),
            ('p7', float, (1,)),
            ('p8', float, (1,)),
            ('p9', float, (1,)),
            ],
    'persis_in': sim_specs['in'] +\
                 ['FLOPS'] +\
                 ['elapsed_sec'] +\
                 ['machine_identifier'] +\
                 ['mpi_ranks'] +\
                 ['threads_per_node'] +\
                 ['ranks_per_node'] +\
                 ['gpu_enabled'] +\
                 ['libE_id'] +\
                 ['libE_workers'],
    'user': {
        'machine_info': MACHINE_INFO,
        'model': model, # provide generation object
        # Arguments to generation
        'conditions': conditions,
        'num_sim_workers': num_sim_workers,
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

def manager_save(H, gen_specs, libE_specs):
    import pandas as pd
    unfinished = H[~H["sim_ended"]][gen_specs['persis_in']]
    finished = H[H["sim_ended"]][gen_specs['persis_in']]
    unfinished_log = pd.DataFrame(dict((k, unfinished[k].flatten()) for k in gen_specs['persis_in']))
    full_log = pd.DataFrame(dict((k, finished[k].flatten()) for k in gen_specs['persis_in']))

    output = f"{libE_specs['ensemble_dir_path']}/unfinished_results.csv"
    if len(unfinished_log) == 0:
        print("All simulations finished.")
    else:
        unfinished_log.to_csv(output, index=False)
        print(f"{len(unfinished_log)} unfinished results logged to {output}")

    output = f"{libE_specs['ensemble_dir_path']}/manager_results.csv"
    full_log.to_csv(output, index=False)
    print(f"All manager-finished results logged to {output}")

if __name__ == '__main__':
    # Perform the libE run
    H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info,
                                alloc_specs=alloc_specs, libE_specs=libE_specs)
    # Save History array to file
    if is_manager:
        # We may have missed the final evaluation in the results file
        print("\nlibEnsemble has completed evaluations.")
        with open(f"{libE_specs['ensemble_dir_path']}/full_H_array.npz",'wb') as np_save_H:
            np.save(np_save_H, H)
        manager_save(H, gen_specs, libE_specs)

