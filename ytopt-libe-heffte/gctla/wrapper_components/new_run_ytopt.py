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
import multiprocessing
multiprocessing.set_start_method('fork', force=True)

import numpy as np
NUMPY_SEED = 1
np.random.seed(NUMPY_SEED)
import pandas as pd

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

from GC_TLA.implementations.heFFTe.heFFTe_problem import heFFTeArchitecture, heFFTe_instance_factory

from wrapper_components.ytopt_asktell import persistent_ytopt  # Generator function, communicates with ytopt optimizer
from wrapper_components.ytopt_obj import init_obj  # Simulator function, calls Plopper
from ytopt.search.optimizer import Optimizer

# SEEDING
CONFIGSPACE_SEED = 1234
YTOPT_SEED = 2345

# Variables that will be sed-edited to control scaling
APP_SCALE = None
APP_SCALE_X = APP_SCALE
APP_SCALE_Y = APP_SCALE
APP_SCALE_Z = APP_SCALE
MPI_RANKS = None
MACHINE_IDENTIFIER = None

def single_listwrap(in_str):
    if type(in_str) is not list:
        return [in_str]
    else:
        return in_str

def name_path(in_str):
    return pathlib.Path(pathlib.Path(in_str).name)

def parse_custom_user_args_in(user_args_in):
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

    # Type-fixing for args: (name, REQUIRED, cast_type, default)
    argument_casts = [('max-evals', True, int, None),
                      ('learner', True, str, None),
                      ('resume', False, single_listwrap, None),
                      ('node-list-file', False, name_path, None),
                      ]
    req_settings = [tup[0] for tup in argument_casts if tup[1]]
    missing = set(req_settings).difference(set(user_args.keys()))
    assert len(missing) == 0, \
            f"Required settings missing: {missing}."+"\n"+\
            f"Specify each setting in {req_settings}"
    for (arg_name, required, cast_type, default) in argument_casts:
        if arg_name in user_args:
            user_args[arg_name] = cast_type(user_args[arg_name])
        else:
            user_args[arg_name] = default
    return user_args

def update_gen_specs(gen_specs, num_sim_workers, problem, MACHINE_INFO, user_args):
    ytoptimizer = Optimizer(
        num_workers = num_sim_workers,
        space = problem.tunable_params,
        learner = user_args['learner'],
        liar_strategy='cl_max',
        acq_func='gp_hedge',
        set_KAPPA=1.96,
        set_SEED=YTOPT_SEED,
        set_NI=10,
        )

    # We may be resuming a previous iteration. LibEnsemble won't let the directory be resumed, so
    # results will have to be merged AFTER the fact (probably by libEwrapper.py). To support this
    # behavior, we only interact with the Optimizer here for run_ytopt.py, so only the Optimizer
    # needs to be lied to in order to simulate the past evaluations
    if user_args['resume'] is not None:
        resume_from = [_ for _ in user_args['resume']]
        print(f"Resuming from records indicated in files: {resume_from}")
        previous_records = pd.concat([pd.read_csv(_) for _ in resume_from])
        print(f"Loaded {len(previous_records)} previous evaluations")

        # Form the fake records using optimizer's preferred lie
        lie = ytoptimizer._get_lie()
        param_cols = problem.tunable_params.get_hyperparameter_names()
        result_col = 'FLOPS'

        keylist, resultlist = [], []
        for idx, row in previous_records.iterrows():
            # I believe this is sufficient for heFFTe -- however in a conditional search space with
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

    # Set values for gen_specs
    gen_specs['gen_f'] = persistent_ytopt
    gen_specs['user']['ytoptimizer'] = ytoptimizer

def prepare_libE(nworkers, libE_specs, user_args_in):
    num_sim_workers = nworkers - 1  # Subtracting one because one worker will be the generator

    user_args = parse_custom_user_args_in(user_args_in)

    # Building the problem would perform architecture detection for us, but right now the nodes aren't
    # identified by libEwrapper so we have to out-of-order this
    arch = heFFTeArchitecture(machine_identifier=MACHINE_IDENTIFIER,
                              hostfile=user_args['node-list-file'],
                              x=APP_SCALE_X,
                              y=APP_SCALE_Y,
                              z=APP_SCALE_Z)
    print(f"Identifying machine as {arch.machine_identifier}"+"\n")
    instance_name = f"heFFTe_{arch.nodes}_{APP_SCALE_X}_{APP_SCALE_Y}_{APP_SCALE_Z}"
    problem = heFFTe_instance_factory.build(instance_name, architecture=arch)
    # Architecture detected nodes from hostfile; if we don't have enough to perform this job then let's exit NOW
    expected_nodes = num_sim_workers * (MPI_RANKS // arch.ranks_per_node)
    assert arch.nodes >= expected_nodes, "Insufficient nodes to perform this job (need: "+\
           f"(sim_workers={num_sim_workers})x((mpi_ranks={MPI_RANKS})/(ranks_per_node={arch.ranks_per_node})) = {expected_nodes}, "+\
           f"detected: {arch.nodes})"
    # Apply seeding
    problem.tunable_params.seed(CONFIGSPACE_SEED)
    print(str(problem))

    libE_specs['use_worker_dirs'] = True # Workers operate in unique directories
    libE_specs['sim_dirs_make'] = False  # Otherwise directories separated by each sim call

    # Copy or symlink needed files into unique directories
    symlinkable = []
    if arch.gpu_enabled:
        symlinkable.extend([pathlib.Path('wrapper_components').joinpath(f) for f in ['gpu_cleanup.sh', 'set_affinity_gpu_polaris.sh']])
    #if user_args['node-list-file'] is not None:
    #    symlinkable.append(pathlib.Path(user_args['node-list-file']))
    libE_specs['sim_dir_symlink_files'] = symlinkable

    # Set working directory for this ensemble
    ENSEMBLE_DIR_PATH = ""
    libE_specs['ensemble_dir_path'] = pathlib.Path(f'ensemble_{ENSEMBLE_DIR_PATH}')
    print(f"This ensemble operates from: {libE_specs['ensemble_dir_path']}"+"\n")

    MACHINE_INFO = {
        'libE_workers': num_sim_workers,
        'app_timeout': 300,
        'mpi_ranks': MPI_RANKS,
        'identifier': arch.machine_identifier,
        'gpu_enabled': arch.gpu_enabled,
        'threads_per_node': arch.threads_per_node,
        'ranks_per_node': arch.ranks_per_node,
        'sequence': arch.thread_sequence,
        'topologies': arch.mpi_topologies,
    }
    # May have a nodelist to work on rather than the full job's nodelist
    if user_args['node-list-file'] is not None:
        MACHINE_INFO['nodelist'] = user_args['node-list-file']
        with open(MACHINE_INFO['nodelist'],'r') as f:
            avail_nodes = [_.rstrip() for _ in f.readlines()]
    elif 'PBS_NODEFILE' in os.environ:
        with open(os.environ['PBS_NODEFILE'],'r') as f:
            avail_nodes = [_.rstrip() for _ in f.readlines()]
    else:
        avail_nodes = None
    # Prepare the node dictionary once outside of the libensemble directory
    if avail_nodes is None:
        worker_nodefile_dictionary = dict((workerID,None) for workerID in range(2,2+num_sim_workers))
    else:
        worker_nodefile_dictionary = dict()
        used_index = len(avail_nodes) % num_sim_workers
        per_worker = len(avail_nodes) // num_sim_workers
        for workerID in range(2,2+num_sim_workers):
            my_nodes = avail_nodes[used_index : used_index+per_worker]
            used_index += per_worker
            my_nodefile = pathlib.Path(f"worker_{workerID}_nodefile")
            with open(my_nodefile,'w') as f:
                f.write("\n".join(my_nodes))
            # From ensemble directory, relative path will be up 2 levels
            worker_nodefile_dictionary[workerID] = pathlib.Path('..').joinpath('..').joinpath(my_nodefile)

    # Declare the sim_f to be optimized, and the input/outputs
    sim_specs = {
        'sim_f': init_obj,
        'in': [_ for _ in problem.tunable_params],
        'out': [('FLOPS', float, (1,)),
                ('elapsed_sec', float, (1,)),
                ('evaluation_sec', float, (1,)),
                ('machine_identifier','<U30', (1,)),
                ('mpi_ranks', int, (1,)),
                ('threads_per_node', int, (1,)),
                ('ranks_per_node', int, (1,)),
                ('gpu_enabled', bool, (1,)),
                ('libE_id', int, (1,)),
                ('libE_workers', int, (1,)),],
        'user': {
            'machine_info': MACHINE_INFO,
            'problem': problem,
            'nodefile_dict': worker_nodefile_dictionary,
        }
    }

    # Declare the gen_f that will generate points for the sim_f, and the various input/outputs
    gen_spec_out_lookup = {
        'C0': ('C0', "<U24", (1,)),
        'P0': ('P0', "<U24", (1,)),
        'P1X': ('P1X', int, (1,)),
        'P1Y': ('P1Y', int, (1,)),
        'P1Z': ('P1Z', int, (1,)),
        'P2': ('P2', "<U24", (1,)),
        'P3': ('P3', "<U24", (1,)),
        'P4': ('P4', "<U24", (1,)),
        'P5': ('P5', "<U24", (1,)),
        'P6': ('P6', "<U24", (1,)),
        'P7': ('P7', "<U24", (1,)),
        'P8': ('P8', int, (1,)),
    }
    gen_specs = {
        'out': [
                # MUST MATCH ORDER OF THE CONFIGSPACE HYPERPARAMETERS EXACTLY
                gen_spec_out_lookup[param] for param in problem.tunable_params
                ],
        'persis_in': sim_specs['in'] +\
                     ['FLOPS', 'elapsed_sec', 'evaluation_sec'] +\
                     ['machine_identifier'] +\
                     ['mpi_ranks', 'threads_per_node', 'ranks_per_node'] +\
                     ['gpu_enabled'] +\
                     ['libE_id', 'libE_workers'],
        'user': {
            'machine_info': MACHINE_INFO,
            'num_sim_workers': num_sim_workers,
            'ensemble_dir': libE_specs['ensemble_dir_path'],
        },
    }
    update_gen_specs(gen_specs, num_sim_workers, problem, MACHINE_INFO, user_args)
    assert 'gen_f' in gen_specs.keys(), "Must set a generator function in update_gen_specs"

    alloc_specs = {
        'alloc_f': alloc_f,
        'user': {'async_return': True},
    }

    # Specify when to exit. More options: https://libensemble.readthedocs.io/en/main/data_structures/exit_criteria.html
    exit_criteria = {'sim_max': int(user_args['max-evals'])}

    # Added as a workaround to issue that's been resolved on develop
    persis_info = add_unique_random_streams({}, nworkers + 1)

    return sim_specs, gen_specs, alloc_specs, libE_specs, exit_criteria, persis_info

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
    # Parse comms, default options from commandline
    nworkers, is_manager, libE_specs, user_args_in = parse_args()
    sim_specs, gen_specs, alloc_specs, libE_specs, exit_criteria, persis_info = prepare_libE(nworkers, libE_specs, user_args_in)
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

