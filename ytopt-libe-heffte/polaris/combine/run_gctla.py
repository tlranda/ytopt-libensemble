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
from copy import deepcopy
import warnings

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
# Tuple is arg_name, required, cast_type
arg_casts = [('max-evals', True, int),
             ('constraint-sys', True, int),
             ('constraint-app', True, int),
             ('auto-budget', False, bool),
             ('initial-quantile', False, float),
             ('min-quantile', False, float),
             ('budget-confidence', False, float),
             ('quantile-reduction', False, float),
             ('ideal-proportion', False, float),
             ('ideal-attrition', False, float),
             ('determine-budget-only', False, bool),]
for (arg_name, required, cast_type) in arg_casts:
    if required or arg_name in user_args:
        user_args[arg_name] = cast_type(user_args[arg_name])

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

# For efficiency's sake, the condition makes batches of 100 samples at a time
conditions = [Condition({'mpi_ranks': user_args['constraint-sys'],
                         'p1': user_args['constraint-app']},
                        num_rows=100)]
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

# Fetch the floatcasting function for GC fitting
topology_cache = problem.plopper.topology_cache
floatcast_fn = problem.plopper.floatcast
uncasted_space_size = problem.input_space_size
# Arguments to control autobudgeting
ideal_proportion = user_args['ideal-proportion'] if 'ideal-proportion' in user_args else 0.05
assert 0 < ideal_proportion <= 1, "Ideal proportion must be > 0 and <= 1"
ideal_attrition = user_args['ideal-attrition'] if 'ideal-attrition' in user_args else 0.05
assert 0 <= ideal_attrition < 1, "Ideal attrition must be >= 0 and < 1"
budget_confidence = user_args['budget-confidence'] if 'budget-confidence' in user_args else 0.95
assert 0 <= budget_confidence < 1, "Budget confidence must be >= 0 and < 1"
quantile_reduction = user_args['quantile-reduction'] if 'quantile-reduction' in user_args else 0.1
assert 0 < quantile_reduction <= 1, "Quantile reduction must be > 0 and <= 1"
min_quantile = user_args['min-quantile'] if 'min-quantile' in user_args else 0.15
assert min_quantile >= 0, "Minimum quantile must be >= 0"
# Determine qualified space size
possible_configurations = uncasted_space_size * len(sequence) * (len(topology_cache[user_args['constraint-sys']]) ** 2)
# Ensure budgeting saturation
mass_condition = deepcopy(conditions)
mass_condition[0].num_rows = possible_configurations
# Load data
data = pd.concat([pd.read_csv(_) for _ in user_args['input']])
data_trimmed = data[['c0',]+[f'p{_}' for _ in range(10)]+['mpi_ranks', 'FLOPS']]
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data_trimmed.drop(columns=['FLOPS']))
data_quantile = user_args['initial-quantile'] if 'initial-quantile' in user_args else 1.0
accepted_model = None
# Mathematics to control auto-budgeting
try:
    from math import comb
except ImportError:
    from math import factorial
    def comb(n,k):
        return factorial(n) / (factorial(k) * factorial(n-k))
def hypergeo(i,p,t,k):
    return (comb(i,t)*comb((p-i),(k-t))) / comb(p,k)

while True:
    # Get subset for data quantile
    fittable = data_trimmed[data_trimmed['FLOPS'] < data_trimmed['FLOPS'].quantile(data_quantile)]
    fittable = fittable.drop(columns=["FLOPS"])
    warnings.simplefilter('ignore')
    model = GaussianCopula(metadata, enforce_min_max_values=False)
    model.add_constraints(constraints=constraints)
    model.fit(fittable)
    warnings.simplefilter('default')

    # Change max-evals if auto-budgeted
    if 'auto-budget' not in user_args or not user_args['auto-budget']:
        accepted_model = model
        suggested_budget = user_args['max-evals']
        break
    mass_sample = model.sample_from_conditions(mass_condition)
    # Downcast floats to interpolated values
    space_sample = floatcast_fn(mass_sample, MACHINE_INFO)
    # Original Population and available population for sampling
    pop = possible_configurations
    subpop = len(space_sample.drop_duplicates())
    # Ideal Population and expected surviving ideal proportion after sampling
    ideal = int(pop * ideal_proportion)
    subideal = max(1, ideal - int((pop - subpop) * ideal_attrition))
    if subideal > subpop:
        print(f"Autotuning budget indeterminate at quantile {data_quantile}")
        suggested_budget = user_args['max-evals']
    else:
        suggested_budget = 0
        while suggested_budget < subideal:
            suggested_budget += 1
            confidence = sum([hypergeo(subideal,subpop,_,suggested_budget) for _ in range(1,suggested_budget+1)])
            if confidence >= budget_confidence:
                break
        if confidence >= budget_confidence:
            print(f"Autotuning budget {suggested_budget} determined at quantile {data_quantile}")
            accepted_model = model
            # Accepted model can be cached
            if suggested_budget < user_args['max-evals']:
                break
            else:
                print(f"Continue pruning to undercut max-evals {user_args['max-evals']}")
        else:
            print(f"Autotuning budget at quantile {data_quantile} failed to satisfy confidence {budget_confidence}; max confidence: {confidence}")
    data_quantile -= quantile_reduction
    if data_quantile <= min_quantile:
        suggested_budget = user_args['max-evals']
        break
del comb, hypergeo, pop, subpop, ideal, subideal
# Set max-evals if it reduces the budget
if suggested_budget < user_args['max-evals']:
    print(f"Auto-budgeting reduces max-evals: {user_args['max-evals']} --> budget: {suggested_budget}")
elif suggested_budget == user_args['max-evals']:
    print(f"Using max-evals: {user_args['max-evals']}")
else:
    print(f"Auto-budgeting would exceed max-evals: {user_args['max-evals']} <-- budget: {suggested_budget}")
    print("!!Max-evals NOT adjusted to suggested budget!!")
user_args['max-evals'] = min(suggested_budget, user_args['max-evals'])
if accepted_model is not None:
    model = accepted_model

if 'determine-budget-only' in user_args and user_args['determine-budget-only']:
    exit()

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

