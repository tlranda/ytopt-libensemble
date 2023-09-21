from autotune.space import Space, Integer, Real
from autotune.problem import TuningProblem
from GPTune.gptune import GPTune, BuildSurrogateModel
from GPTune.computer import Computer
from GPTune.data import Categoricalnorm, Data
from GPTune.database import HistoryDB, GetMachineConfiguration
from GPTune.options import Options
from GPTune.model import GPy

import openturns as ot
import argparse, sys, os, pathlib
import numpy as np, pandas as pd, uuid, time, copy

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
    prs.add_argument("--sys", type=int, required=True, help="System scale to target (mpi_ranks)")
    prs.add_argument("--app", type=int, required=True, help="Application scale to target (fft size)")
    prs.add_argument("--max-evals", type=int, default=30, help="Number of evaluations per task")
    prs.add_argument("--n-init", type=int, default=-1, help="Number of initial evaluations (blind)")
    prs.add_argument("--seed", type=int, default=1234, help="RNG seed")
    prs.add_argument("--preserve-history", action="store_true", help="Prevent existing gptune history files from reuse/clobbering")
    return prs

def parse(args=None, prs=None):
    if prs is None:
        prs = build()
    if args is None:
        args = prs.parse_args()
    return args

def csvs_to_gptune(fnames, tuning_metadata):
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
        for index, row in csv.iterrows():
            new_eval = dict((k,v) for (k,v) in func_template.items())
            new_eval['task_parameter'] = {'mpi_ranks': row['mpi_ranks'], 'p1': row['p1']}
            # SINGLE update per task size
            if index == 0:
                sizes.append(list(new_eval['task_parameter'].values()))
            new_eval['tuning_parameter'] = dict((col, str(row[col])) for col in parameters)
            new_eval['evaluation_result'] = {'flops': row['FLOPS']}
            elapsed_time = row['elapsed_sec']
            if row['libE_id'] in prev_worker_time.keys():
                elapsed_time -= prev_worker_time[row['libE_id']]
            prev_worker_time[row['libE_id']] = row['elapsed_sec']
            assert elapsed_time >= 0
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
            point['p1'] = str(point['p1'])
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
    APP_SCALE = args.app
    #SYSTEM = "Polaris"
    #template_string = "mpiexec -n {mpi_ranks} --ppn {ranks_per_node} --depth {depth} --cpu-bind depth --env OMP_NUM_THREADS={depth} sh ./set_affinity_gpu_polaris.sh {interimfile}"
    SYSTEM = "Theta"
    template_string = "aprun -n {mpi_ranks} -N {ranks_per_node} -cc depth -d {depth} -j {j} -e OMP_NUM_THREADS={depth} sh {interimfile}"
    # Set space and architecture info
    CONFIGSPACE_SEED = 1234
    cs = CS.ConfigurationSpace(seed=CONFIGSPACE_SEED)
    # arg1  precision
    p0 = CSH.CategoricalHyperparameter(name='p0', choices=["double", "float"], default_value="float")
    # arg2  3D array dimension size
    p1 = CSH.Constant(name='p1', value=args.app)
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
    cpu_override = 256
    gpu_enabled = False
    cpu_ranks_per_node = 64

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
    MACHINE_IDENTIFIER = "theta-knl"

    # Create a problem instance
    scale_name = gc_tla_problem.lookup_ival[(args.sys, args.app)]
    warnings.simplefilter('ignore')
    problem = getattr(gc_tla_problem, scale_name)
    problem.plopper.returnmode = 'GPTune'
    problem.plopper.set_architecture_info(threads_per_node = ranks_per_node,
                                          gpus = ranks_per_node if gpu_enabled else 0,
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
        'gpu_enabled': gpu_enabled,
        'libE_workers': 1,
        'app_timeout': 300,
        'sequence': sequence,
    }
    problem.set_space(cs)
    warnings.simplefilter('default')
    # Next build the actual instance for evaluating the target problem
    print(f"Target problem ({args.sys}, {args.app}) constructed")

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
            if param_name == 'p1':
                # Replace constant with full range of options
                opts['categories'] = ('64','128','256','512','1024','1400',)
            elif param_name == 'c0':
                # Replace constant with full range of options
                opts['categories'] = ('cufft',) if gpu_enabled else ('fftw',)
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
    input_space = [{'name': 'mpi_ranks',
                    'type': 'int',
                    'transformer': 'normalize',
                    'lower_bound': int(min([tup[0]*ranks_per_node for tup in problem.dataset_lookup.keys()])),
                    'upper_bound': int(max([tup[0]*ranks_per_node for tup in problem.dataset_lookup.keys()])),},
                   {'name': 'p1',
                    'type': 'int',
                    'transformer': 'normalize',
                    'lower_bound': min([tup[1] for tup in problem.dataset_lookup.keys()]),
                    'upper_bound': max([tup[1] for tup in problem.dataset_lookup.keys()]),},
                   ]
    IS = Space([Integer(low=input_space[0]['lower_bound'],
                        high=input_space[0]['upper_bound'],
                        transform='normalize',
                        name='mpi_ranks'),
                Integer(low=input_space[1]['lower_bound'],
                        high=input_space[1]['upper_bound'],
                        transform='normalize',
                        name='p1'),])

    # Meta Dicts are part of building surrogate models for each input, but have a lot of common
    # specification templated here
    base_meta_dict = {'tuning_problem_name': problem.name.split('Problem')[0][:-1].replace('/','__'),
                      'modeler': 'Model_GPy_LCM',
                      'input_space': input_space,
                      'output_space': output_space,
                      'parameter_space': parameter_space,
                      'loadable_machine_configurations': {'theta': {'intel': {'nodes': NODE_COUNT, 'cores': threads_per_node}}},
                      'loadable_software_configurations': {}
                     }
    # Used to have consistent machine definition
    tuning_metadata = {
        "tuning_problem_name": base_meta_dict['tuning_problem_name'],
        "use_crowd_repo": "no",
        "machine_configuration": {
            "machine_name": "theta",
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
    prior_traces, prior_sizes = csvs_to_gptune(args.inputs, tuning_metadata)
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
    wrapped_objectives = wrap_objective(objectives, model_functions, task_keys=['mpi_ranks','p1'], machine_info=machine_info)
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
    data, modeler, stats = gt.MLA(Tgiven=transfer_task, NS=args.max_evals, NI=n_task, NS1=NS1)
    print(f"Stats: {stats}")

if __name__ == "__main__":
    warnings.simplefilter('ignore')
    main()

