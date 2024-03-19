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

import copy
import warnings
from wrapper_components.libE_asktell import persistent_model # Generator function, communicates with GC model
from wrapper_components.libE_objective import heFFTe_objective # Simulator function, calls Problem
from GC_TLA.plopper.executor import MetricIDs
from sdv.single_table import GaussianCopulaSynthesizer as GaussianCopula
from sdv.metadata import SingleTableMetadata
from sdv.sampling.tabular import Condition
# Mathematics to control auto-budgeting
try:
    from math import comb
except ImportError:
    from math import factorial
    def comb(n,k):
        return factorial(n) / (factorial(k) * factorial(n-k))
def hypergeo(i,p,t,k):
    return (comb(i,t)*comb((p-i),(k-t))) / comb(p,k)


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

def boolcast(in_str):
    return (type(in_str) is str and in_str in ['True','true','t','on','1','Yes','yes','Y','y']) or (type(in_str) is not str and bool(in_str))

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
                      ('input', True, single_listwrap, None),
                      ('constraint-sys', True, int, None),
                      ('constraint-app-x', True, int, None),
                      ('constraint-app-y', True, int, None),
                      ('constraint-app-z', True, int, None),
                      ('ignore', False, single_listwarp, []),
                      ('auto-budget', False, boolcast, False),
                      ('initial-quantile', False, float, 0.1),
                      ('min-quantile', False, float, 0.15),
                      ('budget-confidence', False, float, 0.95),
                      ('quantile-reduction', False, float, 0.1),
                      ('ideal-proportion', False, float, 0.1),
                      ('ideal-attrition', False, float, 0.05),
                      ('determine-budget-only', False, boolcast, False),
                      ('predictions-only', False, boolcast, False),
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

def build_model(problem, user_args):
    # Load data
    cand_files = [pathlib.Path(_) for _ in user_args['input'] if _ not in user_args['ignore']]
    found = [_ for _ in cand_files if _.exists()]
    if len(found) != len(cand_files):
        missing = set(cand_files).difference(set(found))
        warnings.warn(f"Input file(s) not found: {missing}", UserWarning)
    data = pd.concat([pd.read_csv(_) for _ in found]).reset_index(names=["CSV_Order"])
    # Drop non-SDV cols by only using SDV-OK cols
    training_cols = [_ for _ in problem.tunable_params] + ['mpi_ranks', 'threads_per_node', 'ranks_per_node', 'FLOPS']
    # These columns are needed for consistency, but not for SDV learning
    SDV_NONPREDICT = ['threads_per_node','ranks_per_node','FLOPS']
    # Drop erroneous configurations
    least_infinity = min([problem.executor.infinities[_] for _ in MetricIDs if _ != MetricIDs.OK])
    train_data = data.loc[:, training_cols]
    train_data = train_data[train_data['FLOPS'] < least_infinity]
    # Recontextualize topology data
    topo_split = lambda x: [int(_) for _ in x.split(' ') if _ not in ['-ingrid','-outgrid']]
    np_topologies = np.asarray([np.fromstring(_, dtype=int, sep=' ') for _ in problem.architecture.mpi_topologies])
    for topology_key, grid_type in zip(['P6','P7'], ['-ingrid', '-outgrid']):
        # Grab string as 3-integer topology, downsample with logarithm
        top_as_np_log = np.log2(np.vstack(train_data[topology_key].apply(topo_split)))
        # Get max logarithm for each dimension/row
        log_mpi_ranks = np.stack([np.log2(train_data['mpi_ranks'])]*3, axis=1)
        # Change proportion of logarithm to new base for transferred topology sizes
        projection = 2 ** (top_as_np_log / log_mpi_ranks * np.log2(problem.architecture.mpi_ranks))
        # Use nearest-topology search for best match in new topology space
        distances = np.asarray([((np_topologies - p) ** 2).sum(axis=1) for p in projection])
        best_match = np.argmin(distances, axis=1)
        new_topologies = np_topologies[best_match]
        # Return to string
        str_topologies = [" ".join([grid_type]+[str(_) for _ in topo]) for topo in new_topologies]
        train_data[topology_key] = str_topologies
    # Recontextualize sequences
    if 'P8' in problem.tunable_params:
        dest_seq = np.asarray(problem.architecture.thread_sequence)
        # Group by value
        for ((t,r), subframe) in train_data.groupby(['threads_per_node','ranks_per_node']):
            max_select = t // r
            cur_vals = subframe['P8']
            # Reconstruct available sequence at time of this program's execution
            cur_seq = np.asarray(problem.architecture.make_thread_sequence(t,r))
            # Project values via ratio length in list
            projection = (np.asarray([np.argmax((cur_seq == c)) for c in cur_vals]) / len(cur_seq) * len(dest_seq)).round(0).astype(int)
            train_data.loc[subframe.index, 'P8'] = dest_seq[projection]

    # DATA PREPARED
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(train_data.drop(columns=SDV_NONPREDICT))

    # Condition will make batches of 100 samples at a time
    conditions = [Condition({'mpi_ranks': user_args['constraint-sys'],
                             'P1X': user_args['constraint-app-x'],
                             'P1Y': user_args['constraint-app-y'],
                             'P1Z': user_args['constraint-app-z']},
                            num_rows=100)]

    # Condition for budget calculation
    mass_conditions = copy.deepcopy(conditions)
    mass_conditions[0].num_rows = problem.tuning_space_size

    # Fitting process
    accepted_model = None
    suggested_budget = None
    model = GaussianCopula(metadata, enforce_min_max_values=False)
    model.add_constraints(constraints=problem.constraints)
    while accepted_model is None:
        fittable = train_data[train_data['FLOPS'] <= train_data['FLOPS'].qunatile(user_args['initial-quantile'])]
        fittable = fittable.drop(columns=['FLOPS'])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.fit(fittable)
        # Quick-exit if not auto-budgeting
        if not user_args['auto-budget']:
            accepted_model = model
            suggested_budget = user_args['max-evals']
            continue
        # Check expected budget
        mass_sample = model.sample_from_conditions(mass_condition)
        # HyperGeometric arguments
        total_population = problem.tuning_space_size
        sample_population = len(mass_sample.drop_duplicates())
        ideal_samples = int(total_population * user_args['ideal-proportion'])
        subideal_samples = max(1, ideal_samples - int((total_population-sample_population) * user_args['ideal-attrition']))
        print(f"Population {total_population} | Sampleable {sample_population} | Ideal {ideal_samples} | Ideal with Attrition {subideal_samples}")
        if subideal_samples > sample_population:
            print(f"Autotuning budget indeterminate at quantile {user_args['initial-quantile']}")
            suggested_budget = user_args['max-evals']
        else:
            suggested_budget = 0
            tightest_budget = min(subideal_samples, user_args['max-evals'])
            while suggested_budget < tightest_budget:
                suggested_budget += 1:
                confidence = sum([hypergeo(subideal_samples, sample_population, _, suggested_budget) for _ in range(1, suggested_budget+1)])
                # Do not process higher budget with explicitly greater confidence
                if confidence >= user_ags['budget-confidence']:
                    print(f"Autotuning budget {suggested_budget} accepted at quantile {user_args['data-quantile']} (confidence: {confidence})")
                    accepted_model = model
                    break
            if confidence < user_args['budget-confidence']:
                print(f"Autotuning budget at quantile {data_quantile} failed to satisfy confidence {user_args['budget-confidence']}; max confidence: {confidence}")
        user_args['data-quantile'] -= user_args['quantile-reduction']
        if user_args['data-quantile'] <= user_args['min-quantile']:
            print("No autotuning budgets can be satisfied under given constraints")
            exit()
    if user_args['determine-budget-only']:
        exit()
    return accepted_model, conditions

def remove_generated_duplicates(samples, history, dtypes):
    default_machine_info = {'sequence': sequence}
    # Duplicate checking and selection
    samples.insert(0, 'source', ['sample'] * len(samples))
    if len(history) > 0:
        combined = pd.concat((history, samples)).reset_index(drop=False)
    else:
        combined = samples.reset_index(drop=False)
    match_on = list(set(combined.columns).difference(set(['source'])))
    duplicated = np.where(combined.duplicated(subset=match_on))[0]
    sample_idx = combined.loc[duplicated]['index']
    combined = combined.drop(index=duplicated)
    if len(duplicated) > 0:
        print(f"Dropping {len(duplicated)} duplicates from generation")
    else:
        print("No duplicates to remove")
    # Extract non-duplicated samples and ensure history is ready for future iterations
    samples = samples.drop(index=sample_idx)
    combined['source'] = ['history'] * len(combined)
    if 'index' in combined.columns:
        combined = combined.drop(columns=['index'])
    return samples, combined

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
    if user_args['node-list-file'] is not None:
        symlinkable.append(pathlib.Path(user_args['node-list-file']))
    libE_specs['sim_dir_symlink_files'] = symlinkable

    # Set working directory for this ensemble
    ENSEMBLE_DIR_PATH = ""
    libE_specs['ensemble_dir_path'] = pathlib.Path(f'ensemble_{ENSEMBLE_DIR_PATH}')
    print(f"This ensemble operates from: {libE_specs['ensemble_dir_path']}"+"\n")

    print(f"Identifying machine as {arch.machine_identifier}"+"\n")
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

    # Model Creation with Budget Calculation
    model, conditions = build_model(problem, user_args)

    # Declare the sim_f to be optimized, and the input/outputs
    sim_specs = {
        'sim_f': init_obj,
        'in': [_ for _ in problem.tunable_params],
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
            'problem': problem,
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
        'gen_f': persistent_model,
        'out': [
                # MUST MATCH ORDER OF THE CONFIGSPACE HYPERPARAMETERS EXACTLY
                gen_spec_out_lookup[param] for param in problem.tunable_params
                ],
        'persis_in': sim_specs['in'] +\
                     ['FLOPS', 'elapsed_sec'] +\
                     ['machine_identifier'] +\
                     ['mpi_ranks', 'threads_per_node', 'ranks_per_node'] +\
                     ['gpu_enabled'] +\
                     ['libE_id', 'libE_workers'],
        'user': {
            'machine_info': MACHINE_INFO,
            'model': model, # Provides generations
            'conditions': conditions,
            'remove_duplicates': remove_generated_duplicates,
            'num_sim_workers': num_sim_workers,
            'ensemble_dir': libE_specs['ensemble_dir_path'],
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

    # Sometimes just a dry-run for predictions
    if user_args['predictions-only']:
        raw_predictions = model.sample_from_conditions(conditions)
        cleaned, history = remove_generated_duplicates(raw_predictions, [], gen_specs['out'])
        libE_specs['ensemble_dir_path'].mkdir(parents=True, exist_ok=True)
        cleaned.to_csv(libE_specs['ensemble_dir_path'].joinpath('predicted_results.csv'), index=False)
        exit()

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

