from autotune.space import Space, Integer, Real
from autotune.problem import TuningProblem
from GPTune.gptune import GPTune, BuildSurrogateModel
from GPTune.computer import Computer
from GPTune.data import Categoricalnorm, Data
from GPTune.database import HistoryDB, GetMachineConfiguration
from GPTune.options import Options
from GPTune.model import GPy

import openturns as ot
import numpy as np, pandas as pd
import argparse, sys, os, pathlib, uuid, time, copy, subprocess

import gc_tla_problem
import warnings
# GPTune nearly breaks a TON of numpy and scikit-learn APIs and already relies on deprecated behavior
# Feel free to unsuppress warnings and fix, but be warned, there are a LOT of them
warnings.simplefilter('ignore')
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from ConfigSpace import ConfigurationSpace, EqualsCondition

def build():
    prs = argparse.ArgumentParser()
    prs.add_argument("--inputs", type=str, nargs="+", required=True, help="Input files to learn from")
    prs.add_argument("--ignore", type=str, nargs="*", default=None, help="Files to exclude from --inputs (ie: de-globbing)")
    prs.add_argument("--sys", type=int, required=True, help="System scale to target (mpi_ranks)")
    prs.add_argument("--app", type=int, default=64, help="Default application dimension to target (fft size), default: %(default)s")
    prs.add_argument("--app-x", type=int, default=None, help="FFT-size in X dimension (default to --app value)")
    prs.add_argument("--app-y", type=int, default=None, help="FFT-size in Y dimension (default to --app value)")
    prs.add_argument("--app-z", type=int, default=None, help="FFT-size in Z dimension (default to --app value)")
    prs.add_argument("--log", required=True, help="File to log results to as CSV")
    prs.add_argument("--max-evals", type=int, default=30, help="Number of evaluations per task (default: %(default)s)")
    prs.add_argument("--n-init", type=int, default=-1, help="Number of initial evaluations (blind) (default: Use half of <max-evals>)")
    prs.add_argument("--seed", type=int, default=1234, help="RNG seed (default: %(default)s)")
    prs.add_argument("--preserve-history", action="store_true", help="Prevent existing gptune history files from reuse/clobbering when specified")
    prs.add_argument("--dry", action="store_true", help="Do not run actual TL loop, but do all things prior to it")
    prs.add_argument("--system", choices=["theta", "polaris"], default="polaris", help="Call formatting for which cluster (default: %(default)s)")
    prs.add_argument("--cpu-override", type=int, default=None, help="Number of CPU cores on the system (default: detected at runtime)")
    prs.add_argument("--gpu-override", type=int, default=None, help="Number of GPUs on the system (default: detected at runtime)")
    prs.add_argument("--gpu-enabled", action="store_true", help="Unless this flag is given, GPUs will not be used")
    prs.add_argument("--cpu-ranks-per-node", type=int, default=None, help="Number of CPU ranks to give each node (default: value of cpu-cores on system AKA cpu-override)")
    return prs

def parse(args=None, prs=None):
    if prs is None:
        prs = build()
    if args is None:
        args = prs.parse_args()
    # Set undefined FFT size parameters to the default value
    for argname in [f'app_{d}' for d in 'xyz']:
        if getattr(args, argname) is None:
            setattr(args, argname, args.app)
    if args.ignore is None:
        args.ignore = []
    args.inputs = [_ for _ in args.inputs if _ not in args.ignore]
    return args

# Minimum surface splitting solve is used as the default topology for FFT (picked by heFFTe when in-grid and/or out-grid topology == ' ')
def surface(fft_dims, grid):
    # Volume of FFT assigned to each process
    box_size = (np.asarray(fft_dims) / np.asarray(grid)).astype(int)
    # Sum of exchanged surface areas
    return (box_size * np.roll(box_size, -1)).sum()

def minSurfaceSplit(X, Y, Z, procs):
    fft_dims = (X, Y, Z)
    best_grid = (1, 1, procs)
    best_surface = surface(fft_dims, best_grid)
    best_grid = " ".join([str(_) for _ in best_grid])
    topologies = []
    # Consider other topologies that utlize all ranks
    for i in range(1, procs+1):
        if procs % i == 0:
            remainder = int(procs / float(i))
            for j in range(1, remainder+1):
                candidate_grid = (i, j, int(remainder/j))
                if np.prod(candidate_grid) != procs:
                    continue
                strtopology = " ".join([str(_) for _ in candidate_grid])
                topologies.append(strtopology)
                candidate_surface = surface(fft_dims, candidate_grid)
                if candidate_surface < best_surface:
                    best_surface = candidate_surface
                    best_grid = strtopology
    # Topologies are reversed such that the topology order is X-1-1 to 1-1-X
    # This matches the previous version ordering
    return best_grid, list(reversed(topologies))

def csvs_to_gptune(fnames, tuning_metadata, topologies):
    # Top-level JSON info, func_eval will be filled based on data
    json_dict = {'tuning_problem_name': tuning_metadata['tuning_problem_name'],
                 'tuning_problem_category': None,
                 'surrogate_model': [],
                 'func_eval': [],
                }
    # Template for a function evaluation
    func_template = {'constants': {},
                     'machine_configuration': tuning_metadata['machine_configuration'],
                     'software_configuration': tuning_metadata['software_configuration'],
                     'additional_output': {},
                     'source': 'measure',
                    }
    # Loop safety
    parameters = None
    if type(fnames) is str:
        fnames = [fnames]
    # Only need to compute this once
    maxdepth = np.log2(np.prod(np.asarray(topologies[0].split(' ')).astype(int)))
    np_topologies = np.asarray([_.split(' ') for _ in topologies]).astype(int)
    # Prepare return structures
    sizes = [] # Task parameter combinations
    dicts = [] # GPTune-ified data for the task parameter combination
    for fname in fnames:
        # Make basic copy
        gptune_dict = dict((k,v) for (k,v) in json_dict.items())
        csv = pd.read_csv(fname)
        # Only set parameters once -- they'll be consistent throughout different files
        if parameters is None:
            parameters = [_ for _ in csv.columns if (_.startswith('p') or _.startswith('c')) and _ != 'predicted']
        prev_worker_time = {}
        # Prepare topology conversion
        local_maxdepth = np.log2(np.prod(np.asarray(csv.loc[0,'p7'].split(' ')).astype(int)))
        for index, row in csv.iterrows():
            new_eval = dict((k,v) for (k,v) in func_template.items())
            new_eval['task_parameter'] = {'mpi_ranks': row['mpi_ranks'],
                                          'p1x': row['p1x'],
                                          'p1y': row['p1y'],
                                          'p1z': row['p1z'],}
            # SINGLE update per task size
            if index == 0:
                sizes.append(list(new_eval['task_parameter'].values()))
            new_eval['tuning_parameter'] = dict((col, str(row[col])) for col in parameters)
            # P7-8 may require topology reclassification
            if local_maxdepth != maxdepth:
                for key in ['p7','p8']:
                    keylocal = np.asarray(new_eval['tuning_parameter'][key].split(' ')).astype(int)
                    projected = 2 ** (np.log2(keylocal) / local_maxdepth * maxdepth)
                    distances = ((np_topologies - projected) ** 2).sum(axis=1)
                    best_match = np.argmin(distances)
                    new_eval['tuning_parameter'][key] = topologies[best_match]
            new_eval['evaluation_result'] = {'flops': row['FLOPS']}
            elapsed_time = row['elapsed_sec']
            if row['libE_id'] in prev_worker_time.keys():
                elapsed_time -= prev_worker_time[row['libE_id']]
            prev_worker_time[row['libE_id']] = row['elapsed_sec']
            # This assertion probably does not need to exist
            #assert elapsed_time >= 0
            new_eval['evaluation_detail'] = {'time': {'evaluations': elapsed_time,
                                                      'objective_scheme': 'average'}}
            new_eval['uid'] = uuid.uuid4()
            gptune_dict['func_eval'].append(new_eval)
        dicts.append(gptune_dict)
        print(f"GPTune-ified {fname}")
    return dicts, sizes

def wrap_objective(objective, surrogate_to_size_dict, task_keys, machine_info):
    objective.__self__.returnmode = 'GPTune'
    def new_objective(point: dict):
        # Task identifier is 'isize'
        task = tuple([point[key] for key in task_keys])
        if task in surrogate_to_size_dict.keys():
            # For whatever reason, I have GPTune treating P1 categories as strings so you
            # have to honor that and cast the key-value to string
            for xyz in 'xyz':
                point[f'p1{xyz}'] = str(point[f'p1{xyz}'])
            result = surrogate_to_size_dict[task](point)
            # GPTune expects a time component -- just dupe our FOM in there
            result['time'] = result['flops']
            # NO NEED TO LOG RESULTS
        else:
            # Add machine info here
            point['machine_info'] = machine_info
            # Should auto-log results
            result = objective(point)
            # BUG: GPTune's second configuration is unique despite same seed/input. Attempt static eval
            #print(point)
            #result = [1]
        return result
    return new_objective

def cleanup_history(args, problem_name):
    historyfile = f'gptune.db/{problem_name}.json'
    if os.path.exists(historyfile):
        if args.preserve_history:
            contents = os.listdir('gptune.db')
            next_avail = 0
            while os.path.exists(f'gptune.db/{problem_name}_{next_avail}.json'):
                next_avail += 1
            print(f"--PRESERVE HISTORY-- Move {historyfile} --> gptune.db/{problem_name}_{next_avail}.json")
            import shutil
            # Preserves metadata as best as possible in case that is relevant to user
            shutil.copy2(historyfile, f'gptune.db/{problem_name}_{next_avail}.json')
        os.remove(historyfile)

def seqchoice(obj):
    if hasattr(obj, 'sequence') and obj.sequence is not None:
        return obj.sequence
    elif hasattr(obj, 'choices') and obj.choices is not None:
        return obj.choices
    raise ValueError(f"Object {obj} lacks or has NONE for sequences and choices")


def main(args=None, prs=None):
    args = parse(args, prs)
    ot.RandomGenerator.SetSeed(args.seed)
    np.random.seed(args.seed)

    MPI_RANKS = args.sys
    APP_SCALE_X = args.app_x
    APP_SCALE_Y = args.app_y
    APP_SCALE_Z = args.app_z
    SYSTEM = args.system
    if SYSTEM == "polaris":
        template_string = "mpiexec -n {mpi_ranks} --ppn {ranks_per_node} --depth {depth} --cpu-bind depth --env OMP_NUM_THREADS={depth} sh ./set_affinity_gpu_polaris.sh {interimfile}"
    elif SYSTEM == "theta":
        template_string = "aprun -n {mpi_ranks} -N {ranks_per_node} -cc depth -d {depth} -j {j} -e OMP_NUM_THREADS={depth} sh {interimfile}"
    else:
        raise ValueError(f"System {SYSTEM} doesn't have a template string for GPTune to execute the application with")
    # Set space and architecture info
    CONFIGSPACE_SEED = 1234
    cs = CS.ConfigurationSpace(seed=CONFIGSPACE_SEED)
    # arg1  precision
    p0 = CSH.CategoricalHyperparameter(name='p0', choices=["double", "float"], default_value="float")
    # arg2  3D array dimension size
    p1x = CSH.Constant(name='p1x', value=APP_SCALE_X)
    p1y = CSH.Constant(name='p1y', value=APP_SCALE_Y)
    p1z = CSH.Constant(name='p1z', value=APP_SCALE_Z)
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

    # Cross-architecture is out-of-scope for now so we determine this for the current platform and leave it at that
    # This is where overrides would be set in run_gctla, etc, but they're just arguments for GPTune

    c0 = CSH.Constant('c0', value='cufft' if args.gpu_enabled else 'fftw')

    if args.cpu_override is None:
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
        threads_per_node = args.cpu_override
        print(f"Override indicates {threads_per_node} CPU threads on this machine")
    if args.cpu_ranks_per_node is None:
        cpu_ranks_per_node = threads_per_node
    else:
        cpu_ranks_per_node = args.cpu_ranks_per_node
    if args.gpu_enabled:
        if args.gpu_override is None:
            proc = subprocess.run('nvidia-smi -L'.split(' '), capture_output=True)
            if proc.returncode != 0:
                raise ValueError("No GPUs Detected, but in GPU mode")
            gpus = len(proc.stdout.decode('utf-8').strip().split('\n'))
            print(f"Detected {gpus} GPUs on this machine")
        else:
            gpus = args.gpu_override
            print(f"Override indicates {gpus} GPUs on this machine")
        ranks_per_node = gpus
    else:
        ranks_per_node = cpu_ranks_per_node
        print(f"CPU mode; force {ranks_per_node} ranks per node")
    print(f"Set ranks_per_node to {ranks_per_node}"+"\n")

    NODE_COUNT = max(MPI_RANKS // ranks_per_node,1)
    print(f"APP_SCALE (AKA Problem Size X, Y, Z) = {APP_SCALE_X}, {APP_SCALE_Y}, {APP_SCALE_Z}")
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

    # Minimum surface splitting solve is used as the default topology for FFT (picked by heFFTe when in-grid and/or out-grid topology == ' ')
    default_topology, topologies = minSurfaceSplit(APP_SCALE_X, APP_SCALE_Y, APP_SCALE_Z, MPI_RANKS)

    # arg8
    p7 = CSH.CategoricalHyperparameter(name='p7', choices=topologies, default_value=default_topology)
    # arg9
    p8 = CSH.CategoricalHyperparameter(name='p8', choices=topologies, default_value=default_topology)
    # number of threads is hardware-dependent
    # GPTune needs these to be strings
    p9 = CSH.OrdinalHyperparameter(name='p9', sequence=[str(_) for _ in sequence], default_value=str(max_depth))
    cs.add_hyperparameters([p0, p1x, p1y, p1z, p2, p3, p4, p5, p6, p7, p8, p9, c0])
    MACHINE_IDENTIFIER = SYSTEM

    # Create a problem instance
    NODE_COUNT = MPI_RANKS // ranks_per_node
    scale_name = gc_tla_problem.lookup_ival(NODE_COUNT, APP_SCALE_X, APP_SCALE_Y, APP_SCALE_Z)
    warnings.simplefilter('ignore')
    problem = getattr(gc_tla_problem, scale_name)
    problem.selflog = args.log
    # Possibly clean up the default tmp_files
    if len(os.listdir(problem.plopper.outputdir)) == 0:
        print("Removing empty default output directory")
        os.removedirs(problem.plopper.outputdir)
    outputdir = "gptune_files"
    # Deprecate old files to new name so files don't get jumbled together
    if os.path.exists(outputdir):
        i = 0
        mvdir = lambda: f"{outputdir}_{i}"
        while os.path.exists(mvdir()):
            i += 1
        mvdir = mvdir()
        os.rename(outputdir, mvdir)
        print(f"Moved existing gptune files to {mvdir}")
    os.makedirs(outputdir)
    problem.plopper.outputdir = outputdir
    problem.plopper.returnmode = 'GPTune'
    problem.plopper.set_architecture_info(threads_per_node = ranks_per_node,
                                          gpus = ranks_per_node if args.gpu_enabled else 0,
                                          nodes = NODE_COUNT,
                                          mpi_ranks = MPI_RANKS,
                                          machine_identifier = MACHINE_IDENTIFIER,
                                          formatSTR = template_string,
                                          )
    machine_info = {
        'identifier': MACHINE_IDENTIFIER,
        'mpi_ranks': MPI_RANKS,
        'threads_per_node': ranks_per_node,
        'ranks_per_node': ranks_per_node,
        'gpu_enabled': args.gpu_enabled,
        'libE_workers': 1,
        'app_timeout': 300,
        'sequence': sequence,
    }
    problem.set_space(cs)
    warnings.simplefilter('default')
    # Next build the actual instance for evaluating the target problem
    print(f"Target problem ({args.sys}, {APP_SCALE_X}, {APP_SCALE_Y}, {APP_SCALE_Z}) constructed")

    # *S are passed to GPTune objects directly
    # *_space are used to build surrogate models and MOSTLY share kwargs
    # As such the *S_options define common options when this saves re-specification

    # Steal the parameter names / values from Problem object's input space
    # HOWEVER, the p1 parameter which is a constant must be replaced with the FULL space representation!!!
    Space_Components = []
    PS_options = []
    for param_name, param in problem.input_space.items():
        opts = {'name': param_name}
        if type(param) in [CSH.CategoricalHyperparameter, CSH.OrdinalHyperparameter]:
            opts['transform'] = 'onehot'
            opts['categories'] = seqchoice(param)
            PS_type = Categoricalnorm
        elif type(param) is CSH.Constant:
            opts['transform'] = 'onehot'
            PS_type = Categoricalnorm
            if param_name[:2] == 'p1':
                # Replace constant with full range of options
                opts['categories'] = ('64','128','256','512','1024','1400',)
            elif param_name == 'c0':
                # Replace constant with full range of options
                opts['categories'] = ('cufft',) if args.gpu_enabled else ('fftw',)
            else:
                # Actual constant
                opts['categories'] = (param.value,)
        else:
            opts['low'] = param.lower
            opts['high'] = param.upper
            opts['transform'] = 'identity'
            opts['prior'] = 'uniform'
            if type(param) is CSH.UniformFloatHyperparameter:
                PS_type = Real
            else:
                PS_type = Integer
        PS_options.append(opts)
        Space_Components.append(PS_type(**opts))
    PS = Space(Space_Components)
    parameter_space = []
    # Parameter space requires some alteration due to inconsistencies
    for options, obj in zip(PS_options, PS):
        options['transformer'] = options.pop('transform') # REALLY?! Keyname alteration
        if type(obj) is Real:
            options['type'] = 'real'
        elif type(obj) is Integer:
            options['type'] = 'int'
        else: # Categoricalnorm
            options['type'] = 'categorical' # Bonus key
        # Categories key MAY need to become list instead of tuple
        # options['categories'] = list(options['categories'])
        parameter_space.append(options)

    # Able to steal this entirely from Problem object API
    OS = problem.output_space
    output_space = [{'name': 'flops',
                     'type': 'real',
                     'transformer': 'identity',
                     'lower_bound': float('-Inf'),
                     'upper_bound': float('3.0')}]

    # Steal input space limits from Problem object API
    # Tuple is (NODES, APP)
    # Determine values ONCE
    mpi_min_max = [np.inf,-np.inf]
    x_min_max = [np.inf,-np.inf]
    y_min_max = [np.inf,-np.inf]
    z_min_max = [np.inf,-np.inf]
    for tup in problem.dataset_lookup.keys():
        # Used to be [(NODE_SCALE_INT, APP_SCALE_INT)]
        # Now will be [(NODE_SCALE_INT, APP_X_INT, APP_Y_INT, APP_Z_INT)]
        # While I expect the bounds on xyz to be identical, will check independently JIC
        mpi = tup[0] * ranks_per_node
        x = tup[1]
        y = tup[2]
        z = tup[3]
        if mpi < mpi_min_max[0]:
            mpi_min_max[0] = mpi
        elif mpi > mpi_min_max[1]:
            mpi_min_max[1] = mpi
        if x < x_min_max[0]:
            x_min_max[0] = x
        elif x > x_min_max[1]:
            x_min_max[1] = x
        if y < y_min_max[0]:
            y_min_max[0] = y
        elif y > y_min_max[1]:
            y_min_max[1] = y
        if z < z_min_max[0]:
            z_min_max[0] = z
        elif z > z_min_max[1]:
            z_min_max[1] = z
    input_space = [{'name': 'mpi_ranks',
                    'type': 'int',
                    'transformer': 'normalize',
                    'lower_bound': mpi_min_max[0],
                    'upper_bound': mpi_min_max[1],},
                   {'name': 'p1x',
                    'type': 'int',
                    'transformer': 'normalize',
                    'lower_bound': x_min_max[0],
                    'upper_bound': x_min_max[1],},
                   {'name': 'p1y',
                    'type': 'int',
                    'transformer': 'normalize',
                    'lower_bound': y_min_max[0],
                    'upper_bound': y_min_max[1],},
                   {'name': 'p1z',
                    'type': 'int',
                    'transformer': 'normalize',
                    'lower_bound': z_min_max[0],
                    'upper_bound': z_min_max[1],},
                   ]
    IS = Space([Integer(low=input_space[0]['lower_bound'],
                        high=input_space[0]['upper_bound'],
                        transform='normalize',
                        name='mpi_ranks'),
                Integer(low=input_space[1]['lower_bound'],
                        high=input_space[1]['upper_bound'],
                        transform='normalize',
                        name='p1x'),
                Integer(low=input_space[2]['lower_bound'],
                        high=input_space[2]['upper_bound'],
                        transform='normalize',
                        name='p1y'),
                Integer(low=input_space[3]['lower_bound'],
                        high=input_space[3]['upper_bound'],
                        transform='normalize',
                        name='p1z'),
                ])

    # Meta Dicts are part of building surrogate models for each input, but have a lot of common
    # specification templated here
    base_meta_dict = {'tuning_problem_name': problem.name.split('Problem')[0][:-1].replace('/','__'),
                      'modeler': 'Model_GPy_LCM',
                      'input_space': input_space,
                      'output_space': output_space,
                      'parameter_space': parameter_space,
                      'loadable_machine_configurations': {SYSTEM: {'intel': {'nodes': NODE_COUNT, 'cores': threads_per_node}}},
                      'loadable_software_configurations': {}
                     }
    # Used to have consistent machine definition
    tuning_metadata = {
        "tuning_problem_name": base_meta_dict['tuning_problem_name'],
        "use_crowd_repo": "no",
        "machine_configuration": {
            "machine_name": SYSTEM,
            "intel": { "nodes": NODE_COUNT, "cores": threads_per_node }
        },
        "software_configuration": {},
        "loadable_machine_configurations": base_meta_dict['loadable_machine_configurations'],
        "loadable_software_configurations": base_meta_dict['loadable_software_configurations'],
    }
    # IF there is already a historyDB file, it can mess things up. Clean it up nicely
    cleanup_history(args, base_meta_dict['tuning_problem_name'])

    constraints = {}
    objectives = problem.objective
    # Load prior evaluations in GPTune-ready format
    prior_traces, prior_sizes = csvs_to_gptune(args.inputs, tuning_metadata, topologies)
    # Teach GPTune about these prior evaluations
    surrogate_metadata = dict((k,v) for (k,v) in base_meta_dict.items())
    model_functions = {}
    for size, data in zip(prior_sizes, prior_traces):
        surrogate_metadata['task_parameter'] = [size]
        print(f"Build Surrogate model for size {size}")
        model_functions[tuple(size)] = BuildSurrogateModel(problem_space=surrogate_metadata,
                                                    modeler=surrogate_metadata['modeler'],
                                                    input_task=surrogate_metadata['task_parameter'],
                                                    function_evaluations=data['func_eval'])
    wrapped_objectives = wrap_objective(objectives, model_functions, task_keys=['mpi_ranks','p1x','p1y','p1z'], machine_info=machine_info)
    #func_evals = []
    #for prior_data in prior_traces:
    #    func_evals.extend(prior_data['func_eval'])
    #models, model_functions = gt.GenSurrogateModel([[s] for s in prior_sizes], func_evals)

    gptune_problem = TuningProblem(IS,PS,OS, wrapped_objectives, constraints, None) # None = models (dict of names : func(point_dict) -> list(outputs)

    machine, processor, nodes, cores = GetMachineConfiguration(meta_dict=tuning_metadata)
    print(f"Machine: {machine} | Processor: {processor} | Num_Nodes: {nodes} | Num_Cores: {cores}")

    computer = Computer(nodes=nodes, cores=cores, hosts=None)
    data = Data(gptune_problem)
    historydb = HistoryDB(meta_dict=tuning_metadata)
    options  = Options()
    # These options inherited from Jaehoon's script
    options.update({'model_restarts': 1,
                    'distributed_memory_parallelism': False,
                    'shared_memory_parallelism': False,
                    'objective_evaluation_parallelism': False,
                    'objective_multisample_threads': 1,
                    'objective_multisample_Processes': 1,
                    'objective_nprocmax': 1,
                    'model_processes': 1,
                    'model_class': 'Model_GPy_LCM',
                    'verbose': False, # True
                    'sample_class': 'SampleOpenTURNS',
                   })
    options.validate(computer=computer)
    # Create the GPTune object
    gt = GPTune(gptune_problem, computer=computer, data=data, options=options, historydb=historydb)

    # Set up the actual transfer learning task
    # THIS is what GPTune's HistoryDB says you should do for TLA; same # evals in all problems,
    # but leverage model functions on prior tasks to simulate their results
    transfer_task = [list(problem.problem_class)]
    # The problem class indicates # nodes, not # mpi ranks; we need the latter
    transfer_task[0][0] *= ranks_per_node
    transfer_task.extend([s for s in prior_sizes])
    n_task = len(transfer_task)
    if args.n_init == -1:
        NS1 = max(args.max_evals//2,1)
    else:
        NS1 = args.n_init
    if args.dry:
        exit()
    data, modeler, stats = gt.MLA(Tgiven=transfer_task, NS=args.max_evals, NI=n_task, NS1=NS1)
    print(f"Stats: {stats}")

if __name__ == "__main__":
    warnings.simplefilter('ignore')
    main()

