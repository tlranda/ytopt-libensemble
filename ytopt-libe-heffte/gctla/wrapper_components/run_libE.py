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
# Module dependencies from non-default sources
import numpy as np
import pandas as pd
from GC_TLA.problem import Problem
# Import libEnsemble items for this test
try:
    from libensemble.specs import SimSpecs, GenSpecs, LibeSpecs, AllocSpecs, ExitCriteria
    from libensemble import Ensemble
    legacy_import = False
except ImportError:
    from libensemble.libE import libE
    legacy_import = True
from libensemble.alloc_funcs.start_only_persistent import only_persistent_gens as alloc_f
from libensemble.tools import parse_args, save_libE_output, add_unique_random_streams
from libensemble import logger
logger.set_level("DEBUG") # Ensure logs are worth reading

# Argument casting methods you may use for user-args-in
def boolcast(in_str):
    return (type(in_str) is str and in_str in ['True','true','t','on','1','Yes','yes','Y','y']) or (type(in_str) is not str and bool(in_str))
def name_path(in_str):
    return pathlib.Path(pathlib.Path(in_str).name)
def single_listwrap(in_str):
    if type(in_str) is not list:
        return [in_str]
    else:
        return in_str

argcasts_dict = {
    'boolcast': boolcast,
    'name_path': name_path,
    'single_listwrap': single_listwrap,
}

class libE_base:
    # Variables that will be sed-edited as class-configurations
    # SEEDING
    CONFIGSPACE_SEED = 1234
    NUMPY_SEED = 1
    # ARCHITECTURE
    MPI_RANKS = None
    MACHINE_IDENTIFIER = None
    # EXPERIMENTS
    ENSEMBLE_DIR_PATH = ""
    DEFAULT_TIMEOUT = 300

    # Tuning space parameters as a dictionary of:
    # hyperparameter_key : numpy dtype tuple (name, dtype, shape)
    gen_spec_out_lookup = {
        # i.e.: 'C0': ('C0', "<U24", (1,)),
    }


    # Type-fixing for args: (name, REQUIRED, cast_callable, default_value)
    argument_casts = [
        ('max-evals', True, int, None),
        ('node-list-file', False, name_path, None),
    ]
    # This variable is easier to override and will always be combined with the defaults above
    argument_casts_extensions = []

    def __init__(self, n_generators=1, legacy_launch=legacy_import):
        self.legacy_launch = legacy_launch
        # Initial collection from libE library
        self.nworkers, self.is_manager, self.libE_specs, user_args_in = parse_args()
        # Get split of generators / simulators (usually 1 generator, rest simulators)
        self.num_sim_workers = self.nworkers - n_generators
        # Additional arguments / casting from command line
        self.set_user_args(user_args_in)
        # Prepare libE for execution
        self.prepare_libE()

    def set_user_args(self, user_args_in):
        # Ensure the easy override target is included in parsing
        self.argument_casts.extend(self.argument_casts_extensions)
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

        req_settings = [tup[0] for tup in self.argument_casts if tup[1]]
        missing = set(req_settings).difference(set(user_args.keys()))
        assert len(missing) == 0, \
                f"Required settings missing: {missing}."+"\n"+\
                f"Specify each setting in {req_settings}"
        for (arg_name, required, cast_type, default) in self.argument_casts:
            if arg_name in user_args:
                user_args[arg_name] = cast_type(user_args[arg_name])
            else:
                user_args[arg_name] = default
        self.user_args = user_args

    def set_problem(self):
        # Must set the self.problem attribute to a GC_TLA.Problem instance (inheritance OK)
        pass

    def set_symlinkable(self):
        # Copy or symlink needed files into unique directories
        self.symlinkable = []
        if self.user_args['node-list-file'] is not None:
            self.symlinkable.append(pathlib.Path(self.user_args['node-list-file']))

    def set_machine_info(self):
        MACHINE_INFO = {
            'libE_workers': self.num_sim_workers,
            'app_timeout': self.DEFAULT_TIMEOUT,
            'mpi_ranks': self.problem.architecture.mpi_ranks,
            'identifier': self.problem.architecture.machine_identifier,
            'threads_per_node': self.problem.architecture.threads_per_node,
            'ranks_per_node': self.problem.architecture.ranks_per_node,
        }
        # May have a nodelist to work on rather than the full job's nodelist
        if self.user_args['node-list-file'] is not None:
            MACHINE_INFO['nodelist'] = self.user_args['node-list-file']
            with open(MACHINE_INFO['nodelist'],'r') as f:
                avail_nodes = [_.rstrip() for _ in f.readlines()]
        elif 'PBS_NODEFILE' in os.environ:
            with open(os.environ['PBS_NODEFILE'],'r') as f:
                MACHINE_INFO['nodelist'] = os.environ['PBS_NODEFILE']
                avail_nodes = [_.rstrip() for _ in f.readlines()]
        else:
            avail_nodes = None
        # Prepare the node dictionary once outside of the libensemble directory
        if avail_nodes is None:
            worker_nodefile_dictionary = dict((workerID,None) for workerID in range(2,2+self.num_sim_workers))
        else:
            worker_nodefile_dictionary = dict()
            used_index = len(avail_nodes) % self.num_sim_workers
            per_worker = len(avail_nodes) // self.num_sim_workers
            for workerID in range(2,2+self.num_sim_workers):
                my_nodes = avail_nodes[used_index : used_index+per_worker]
                used_index += per_worker
                my_nodefile = pathlib.Path(f"worker_{workerID}_nodefile")
                with open(my_nodefile,'w') as f:
                    f.write("\n".join(my_nodes))
                # From ensemble directory, relative path will be up 2 levels
                worker_nodefile_dictionary[workerID] = pathlib.Path('..').joinpath('..').joinpath(my_nodefile)
        self.MACHINE_INFO = MACHINE_INFO
        self.worker_nodefile_dictionary = worker_nodefile_dictionary

    def update_sim_specs(self):
        # Set values for sim_specs
        # Expected attributes to be available/needed: self.{sim_specs,num_sim_workers,nworkers,problem,MACHINE_INFO,user_args}
        #self.sim_specs['sim_f'] = init_obj
        #self.sim_specs['out'].extend([numpy-dtype tuples of any simulator values to track i.e.: (name, type, shape)])
        #self.sim_specs['user'][WHATEVER_YOU_NEED] = WHATEVER_YOU_WANT

    def update_gen_specs(self):
        # Set values for gen_specs
        # Expected attributes to be available/needed: self.{gen_specs,num_sim_workers,nworkers,problem,MACHINE_INFO,user_args}
        #self.gen_specs['gen_f'] = persistent_ytopt
        #self.gen_specs['persis_in'].extend([names of all sim_spec['out'] extensions])
        #self.gen_specs['user'][WHATEVER_YOU_NEED] = WHATEVER_YOU_WANT

    def set_seeds(self):
        np.random.seed(self.NUMPY_SEED)
        self.problem.tunable_params.seed(self.CONFIGSPACE_SEED)

    def prepare_libE(self):
        # Initial Configuration
        self.set_problem()
        assert hasattr(self,'problem') and isinstance(self.problem, Problem), "Must set problem attribute via set_problem() to a GC_TLA.problem.Problem"
        self.set_symlinkable()
        self.set_machine_info()

        self.libE_specs['use_worker_dirs'] = True # Workers operate in unique directories
        self.libE_specs['sim_dirs_make'] = False  # Otherwise directories separated by each sim call
        self.libE_specs['sim_dir_symlink_files'] = self.symlinkable
        self.libE_specs['ensemble_dir_path'] = pathlib.Path(f'ensemble_{self.ENSEMBLE_DIR_PATH}')
        print(f"This ensemble operates from: {self.libE_specs['ensemble_dir_path']}"+"\n")

        # Declare the sim_f to be optimized, and the input/outputs
        self.sim_specs = {
            'in': [_ for _ in self.problem.tunable_params],
            'out': [
                    ('elapsed_sec', float, (1,)),
                    ('evaluation_sec', float, (1,)),
                    ('machine_identifier','<U30', (1,)),
                    ('mpi_ranks', int, (1,)),
                    ('threads_per_node', int, (1,)),
                    ('ranks_per_node', int, (1,)),
                    ('libE_id', int, (1,)),
                    ('libE_workers', int, (1,)),
                   ],
            'user': {
                'machine_info': self.MACHINE_INFO,
                'problem': self.problem,
                'nodefile_dict': self.worker_nodefile_dictionary,
            }
        }
        self.update_sim_specs()
        assert 'sim_f' in self.sim_specs.keys(), "Must set a simulator function 'sim_f' key in self.sim_specs via self.update_sim_specs()"

        # Declare the gen_f that will generate points for the sim_f, and the various input/outputs
        self.gen_specs = {
            'out': [
                    # MUST MATCH ORDER OF THE CONFIGSPACE HYPERPARAMETERS EXACTLY
                    self.gen_spec_out_lookup[param] for param in self.problem.tunable_params
                   ],
            'persis_in': self.sim_specs['in'] +\
                         ['elapsed_sec', 'evaluation_sec'] +\
                         ['machine_identifier', 'mpi_ranks', 'threads_per_node', 'ranks_per_node'] +\
                         ['libE_id', 'libE_workers'],
            'user': {
                'machine_info': self.MACHINE_INFO,
                'num_sim_workers': self.num_sim_workers,
                'ensemble_dir': self.libE_specs['ensemble_dir_path'],
            },
        }
        self.update_gen_specs()
        assert 'gen_f' in self.gen_specs.keys(), "Must set a generator function 'gen_f' key in self.gen_specs via self.update_gen_specs()"

        self.alloc_specs = {'alloc_f': alloc_f, 'user': {'async_return': True}, }
        # Specify when to exit. More options: https://libensemble.readthedocs.io/en/main/data_structures/exit_criteria.html
        self.exit_criteria = {'sim_max': int(self.user_args['max-evals'])}
        # Added as a workaround to issue that's been resolved on develop
        self.persis_info = add_unique_random_streams({}, self.nworkers + 1)
        # Finally, ensure all seeds are set appropriately
        self.set_seeds()

    def manager_save(self, H):
        unfinished = H[~H["sim_ended"]][self.gen_specs['persis_in']]
        finished = H[H["sim_ended"]][self.gen_specs['persis_in']]
        unfinished_log = pd.DataFrame(dict((k, unfinished[k].flatten()) for k in self.gen_specs['persis_in']))
        full_log = pd.DataFrame(dict((k, finished[k].flatten()) for k in self.gen_specs['persis_in']))

        output = f"{self.libE_specs['ensemble_dir_path']}/unfinished_results.csv"
        if len(unfinished_log) == 0:
            print("All simulations finished.")
        else:
            unfinished_log.to_csv(output, index=False)
            print(f"{len(unfinished_log)} unfinished results logged to {output}")

        output = f"{self.libE_specs['ensemble_dir_path']}/manager_results.csv"
        full_log.to_csv(output, index=False)
        print(f"All {len(full_log)} manager-finished results logged to {output}")

    def run(self):
        if self.legacy_launch:
            # Perform the libE run
            H, persis_info, flag = libE(self.sim_specs, self.gen_specs, self.exit_criteria, self.persis_info,
                                        alloc_specs=self.alloc_specs, libE_specs=self.libE_specs)
        else:
            # We can separate experiment creation from running, which can allow an exit trap
            # to capture more results during shutdown
            experiment = Ensemble(sim_specs=self.sim_specs, gen_specs=self.gen_specs, alloc_specs=self.alloc_specs,
                                  exit_criteria=self.exit_criteria, persis_info=self.persis_info,
                                  libE_specs=self.libE_specs)
            H, persis_info, flag = experiment.run()

        # Save History array to file
        if self.is_manager:
            # We may have missed the final evaluation in the results file
            print("\nlibEnsemble has completed evaluations.")
            with open(f"{self.libE_specs['ensemble_dir_path']}/full_H_array.npz",'wb') as np_save_H:
                np.save(np_save_H, H)
            self.manager_save(H)

if __name__ == '__main__':
    # Parse comms, default options from commandline, then execute
    libE_heFFTe().run()

