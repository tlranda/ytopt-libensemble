"""
Runs libEnsemble to call the ytopt ask/tell interface in a generator function,
and the ytopt findRunTime interface in a simulator function.

Execute locally via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python run_ytopt.py
   python run_ytopt.py --nworkers 3 --comms local

The number of concurrent evaluations of the objective function will be 4-1=3.
"""

import os
import pathlib
import glob
import numpy as np
NUMPY_SEED = 1
np.random.seed(NUMPY_SEED)
import itertools
import subprocess

import multiprocessing
multiprocessing.set_start_method('fork', force=True)

# Import libEnsemble items for this test
try:
    from libensemble.specs import SimSpecs, GenSpecs, LibeSpecs, AllocSpecs, ExitCriteria
    from libensemble import Ensemble
    legacy_mode = False
except ImportError:
    from libensemble.libE import libE
    legacy_mode = True
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
from libensemble.tools import parse_args, save_libE_output, add_unique_random_streams
from libensemble import logger
logger.set_level("DEBUG") # Ensure logs are worth reading

from ytopt_asktell import persistent_ytopt  # Generator function, communicates with ytopt optimizer
from ytopt_obj import init_obj  # Simulator function, calls Plopper

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from ConfigSpace import ConfigurationSpace, EqualsCondition
from ytopt.search.optimizer import Optimizer

import pandas as pd

# Parse comms, default options from commandline
nworkers, is_manager, libE_specs, user_args_in = parse_args()
num_sim_workers = nworkers - 1  # Subtracting one because one worker will be the generator

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

req_settings = ['max-evals', 'learner']
assert all([opt in user_args for opt in req_settings]), \
        f"Required settings missing: {set(req_settings).difference(set(user_args.keys()))}."+"\n"+\
        f"Specify each setting in {req_settings}"
# Type-fixing for args
int_args = ['max-evals', ]
for arg in int_args:
    user_args[arg] = int(user_args[arg])

# Set options so workers operate in unique directories
PLOPPER_TARGET = "roibin_TEMPLATE.json"
here = pathlib.Path('.')
libE_specs['use_worker_dirs'] = True
libE_specs['sim_dirs_make'] = False
libE_specs['sim_dir_symlink_files'] = [here.joinpath('template_jsons').joinpath(f) for f in [PLOPPER_TARGET]]
ENSEMBLE_DIR_PATH = ""
libE_specs['ensemble_dir_path'] = f"./ensemble_{ENSEMBLE_DIR_PATH}"
#if you need to manually specify resource information, ie:
#    libE_specs['resource_info'] = {'cores_on_node': (64,256), 'gpus_on_node': 0}
print(f"This ensemble operates as: {libE_specs['ensemble_dir_path']}"+"\n")

# SEEDING
CONFIGSPACE_SEED = 1234
YTOPT_SEED = 2345

# Cross-architecture is out-of-scope for now so we determine this for the current platform and leave it at that
MPI_RANKS = 64
cpu_override = None
cpu_ranks_per_node = None
gpu_enabled = False

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

# Define space
cs = CS.ConfigurationSpace(seed=CONFIGSPACE_SEED)
c0 = CSH.Constant(name='c0', value=PLOPPER_TARGET)
p0 = CSH.UniformIntegerHyperparameter(name='p0', lower=1, upper=60) # MPI Threads
p1 = CSH.UniformIntegerHyperparameter(name='p1', lower=1, upper=4) # Roibin Threads
p2 = CSH.UniformIntegerHyperparameter(name='p2', lower=1, upper=4) # Binning Threads
cs.add_hyperparameters([c0,p0,p1,p2])
if 'blosc' in PLOPPER_TARGET:
    p3 = CSH.UniformIntegerHyperparameter(name='p3', lower=1, upper=4) # Blosc internal threads
    cs.add_hyperparameters([p3])


ytoptimizer = Optimizer(
    num_workers = num_sim_workers,
    space = cs,
    learner = user_args['learner'],
    liar_strategy = 'cl_max',
    acq_func = 'gp_hedge',
    set_KAPPA = 1.96,
    set_SEED = YTOPT_SEED,
    set_NI = 10,
)

# We may be resuming a previous iteration. LibEnsemble won't let the directory be resumed, so
# results will have to be merged AFTER the fact (probably by libEwrapper.py). To support this
# behavior, we only interact with the Optimizer here for run_ytopt.py, so only the Optimizer
# needs to be lied to in order to simulate the past evaluations
if 'resume' in user_args:
    resume_from = [_ for _ in user_args['resume']]
    print(f"Resuming from records indicated in files: {resume_from}")
    previous_records = pd.concat([pd.read_csv(_) for _ in resume_from])
    print(f"Loaded {len(previous_records)} previous evaluations")

    # Form the fake records using optimizer's preferred lie
    lie = ytoptimizer._get_lie()
    param_cols = cs.get_hyperparameter_names()
    result_col = 'FOM'

    keylist, resultlist = [], []
    for idx, row in previous_records.iterrows():
        # I believe this is sufficient for LibPressio -- however in a conditional search space with
        # sometimes-deactivated parameters you may need to be more careful.
        # ytoptimizer.make_key() will help but only if the records indicate nan-values appropriately
        key = ytoptimizer.make_key(row[param_cols].to_list())
        if key not in ytoptimizer.evals:
            # Stage the result of asking for the key
            ytoptimizer.evals[key] = lie
            # Prepare lie material and the actual results to tell back
            keylist.append(key)
            keydict = dict((k,v) for (k,v) in zip(param_cols, key))
            result = row[result_col]
            resultlist.append(tuple([keydict, result]))
    n_prepared = len(keylist)
    print(f"Prepared {n_prepared} prior evaluations")
    # Now that side affects are in place, commit the actual lies
    # We also guarantee underlying optimizers are forced to fit by setting NI / _n_initial_points = 0
    # This means that future ask()'s will not be random and will be based on a model fitted to available data
    ytoptimizer.NI = ytoptimizer._optimizer._n_initial_points = 0
    ytoptimizer.counter += n_prepared
    ytoptimizer._optimizer.tell(keylist, [lie] * n_prepared)
    # Update the lies and trigger underlying optimizer to refit
    ytoptimizer.tell(resultlist)
    old_max_evals = user_args['max-evals']
    user_args['max-evals'] -= n_prepared
    # When resuming, we never want to actually use ask_initial() so have that function point to ask()
    def wrap_initial(n_points=1):
        points = ytoptimizer.ask(n_points=n_points)
        return list(points)[0]
    ytoptimizer.ask_initial = wrap_initial
    print(f"Optimizer updated and ready to resume -- max-evals reduced {old_max_evals} --> {user_args['max-evals']}")

MACHINE_IDENTIFIER = "tbd"
print(f"Identifying machine as {MACHINE_IDENTIFIER}"+"\n")
MACHINE_INFO = {
    'identifier': MACHINE_IDENTIFIER,
    'mpi_ranks': MPI_RANKS,
    'threads_per_node': threads_per_node,
    'ranks_per_node': ranks_per_node,
    'gpu_enabled': gpu_enabled,
    'libE_workers': num_sim_workers,
    'sequence': sequence,
    'app_timeout': 300,
    'ensemble_dir_path': libE_specs['ensemble_dir_path'],
}


# LibEnsemble structures
sim_specs = {
    'sim_f': init_obj,
    'in': cs.get_hyperparameter_names(),
    'out': [('FOM', float, (1,)),
            ('elapsed_sec', float, (1,)),
            ('machine_identifier', '<U30', (1,)),
            ('mpi_ranks', int, (1,)),
            ('threads_per_node', int, (1,)),
            ('ranks_per_node', int, (1,)),
            ('gpu_enabled', bool, (1,)),
            ('libE_id', int, (1,)),
            ('libE_workers', int, (1,)),],
    'user': {
        'machine_info': MACHINE_INFO,
    },
}
print("Tunable parameters:", cs.get_hyperparameter_names())
gen_specs = {
    'gen_f': persistent_ytopt,
    'out': [(name, '<U50', (1,)) for name in cs.get_hyperparameter_names() if name.startswith('c')] +\
           [(name, int, (1,)) for name in cs.get_hyperparameter_names() if name.startswith('p')],
    'persis_in': sim_specs['in'] +\
                 ['FOM', 'elapsed_sec', 'machine_identifier', 'mpi_ranks', 'threads_per_node',
                  'ranks_per_node', 'gpu_enabled', 'libE_id', 'libE_workers'],
    'user': {
        'machine_info': MACHINE_INFO,
        'ytoptimizer': ytoptimizer,
        'num_sim_workers': num_sim_workers,
    },
}
alloc_specs = {
    'alloc_f': alloc_f,
    'user': {'async_return': True},
}
exit_criteria = {'sim_max': int(user_args['max-evals'])}
persis_info = add_unique_random_streams({}, nworkers+1)

def manager_save(H, gen_specs, libE_specs):
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
    if legacy_mode:
        # Perform the libE run
        H, persis_info, flag = libE(sim_specs, gen_specs, exit_criteria, persis_info,
                                    alloc_specs=alloc_specs, libE_specs=libE_specs)
    else:
        # We can separate experiment creation from running, which can allow an exit trap
        # to capture more results during shutdown
        experiment = Ensemble(sim_specs=sim_specs, gen_specs=gen_specs, alloc_specs=alloc_specs,
                              exit_criteria=exit_criteria, persis_info=persis_info,
                              libE_specs=libE_specs)
        H, persis_info, flag = experiment.run()

    # Save History array to file
    if is_manager:
        # We may have missed the final evaluation in the results file
        print("\nlibEnsemble has completed evaluations.")
        with open(f"{libE_specs['ensemble_dir_path']}/full_H_array.npz",'wb') as np_save_H:
            np.save(np_save_H, H)
        manager_save(H, gen_specs, libE_specs)

