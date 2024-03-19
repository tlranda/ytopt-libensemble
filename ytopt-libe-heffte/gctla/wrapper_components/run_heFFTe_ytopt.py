"""
Runs libEnsemble to call the ytopt ask/tell interface in a generator function,
and the ytopt findRunTime interface in a simulator function.

Execute locally via one of the following commands (e.g. 3 workers):
   mpiexec -np 4 python run_ytopt.py
   python run_ytopt.py --nworkers 3 --comms local

The number of concurrent evaluations of the objective function will be 4-1=3.
"""

import pathlib
# Module dependencies from non-default sources
import pandas as pd

from wrapper_components.run_libE import libE_base, argcasts_dict

from GC_TLA.implementations.heFFTe.heFFTe_problem import heFFTeArchitecture, heFFTe_instance_factory

from wrapper_components.ytopt_asktell import persistent_ytopt # Generator function, communicates with ytopt optimizer
from wrapper_components.libe_objective import heFFTe_objective # Simulator function, calls Plopper
from ytopt.search.optimizer import Optimizer

class libE_heFFTe(libE_base):
    # Variables that will be sed-edited as class-configurations
    # SEEDING
    CONFIGSPACE_SEED = 1234
    NUMPY_SEED = 1
    YTOPT_SEED = 2345
    # ARCHITECTURE
    MPI_RANKS = None
    MACHINE_IDENTIFIER = None
    APP_SCALE = None
    APP_SCALE_X = APP_SCALE
    APP_SCALE_Y = APP_SCALE
    APP_SCALE_Z = APP_SCALE
    # EXPERIMENTS
    ENSEMBLE_DIR_PATH = ""
    DEFAULT_TIMEOUT = 300

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

    # Type-fixing for args: (name, REQUIRED, cast_callable, default_value)
    argument_cast_extensions = [
        ('learner', True, str, None),
        ('resume', False, argcasts_dict['single_listwrap'], None),
    ]

    def set_problem(self):
        # Building the problem would perform architecture detection for us, but right now the nodes aren't
        # identified by libEwrapper so we have to out-of-order this
        arch = heFFTeArchitecture(machine_identifier=self.MACHINE_IDENTIFIER,
                                  hostfile=self.user_args['node-list-file'],
                                  x=self.APP_SCALE_X,
                                  y=self.APP_SCALE_Y,
                                  z=self.APP_SCALE_Z)
        print(f"Identifying machine as {arch.machine_identifier}"+"\n")
        instance_name = f"heFFTe_{arch.nodes}_{self.APP_SCALE_X}_{self.APP_SCALE_Y}_{self.APP_SCALE_Z}"
        problem = heFFTe_instance_factory.build(instance_name, architecture=arch)
        # Architecture detected nodes from hostfile; if we don't have enough to perform this job then let's exit NOW
        expected_nodes = self.num_sim_workers * (self.MPI_RANKS // arch.ranks_per_node)
        assert arch.nodes >= expected_nodes, "Insufficient nodes to perform this job (need: "+\
               f"(sim_workers={self.num_sim_workers})x((mpi_ranks={self.MPI_RANKS})/(ranks_per_node={arch.ranks_per_node})) = {expected_nodes}, "+\
               f"detected: {arch.nodes})"
        self.problem = problem
        print(str(self.problem))

    def set_machine_info(self):
        super().set_machine_info()
        # Attributes in our heFFTe architecture, not the basic architecture
        self.MACHINE_INFO['gpu_enabled'] = self.problem.architecture.gpu_enabled
        self.MACHINE_INFO['sequence'] = self.problem.architecture.thread_sequence
        self.MACHINE_INFO['topologies'] = self.problem.architecture.mpi_topologies

    def set_symlinkable(self):
        super().set_symlinkable()
        # Copy or symlink needed files into unique directories
        if self.problem.architecture.gpu_enabled:
            self.symlinkable.extend([pathlib.Path('wrapper_components').joinpath(f) for f in ['gpu_cleanup.sh', 'set_affinity_gpu_polaris.sh']])

    def update_sim_specs(self):
        # Set values for sim_specs
        # Expected attributes to be available/needed: self.{sim_specs,num_sim_workers,nworkers,problem,MACHINE_INFO,user_args}
        self.sim_specs['sim_f'] = heFFTe_objective
        self.sim_specs['out'].extend([
            ('gpu_enabled', bool, (1,)),
            ('FLOPS', float, (1,)),
        ])

    def update_gen_specs(self):
        # Set values for gen_specs
        # Expected attributes to be available/needed: self.{gen_specs,num_sim_workers,nworkers,problem,MACHINE_INFO,user_args}
        self.gen_specs['gen_f'] = persistent_ytopt
        self.gen_specs['persis_in'].extend(['gpu_enabled','FLOPS'])
        # Make Ytoptimizer object and add it to user space
        ytoptimizer = Optimizer(
            num_workers = self.num_sim_workers,
            space = self.problem.tunable_params,
            learner = self.user_args['learner'],
            liar_strategy='cl_max',
            acq_func='gp_hedge',
            set_KAPPA=1.96,
            set_SEED=self.YTOPT_SEED,
            set_NI=10,
            )

        # We may be resuming a previous iteration. LibEnsemble won't let the directory be resumed, so
        # results will have to be merged AFTER the fact (probably by libEwrapper.py). To support this
        # behavior, we only interact with the Optimizer here for run_ytopt.py, so only the Optimizer
        # needs to be lied to in order to simulate the past evaluations
        if self.user_args['resume'] is not None:
            resume_from = [_ for _ in self.user_args['resume']]
            print(f"Resuming from records indicated in files: {resume_from}")
            previous_records = pd.concat([pd.read_csv(_) for _ in resume_from])
            print(f"Loaded {len(previous_records)} previous evaluations")

            # Form the fake records using optimizer's preferred lie
            lie = ytoptimizer._get_lie()
            param_cols = self.problem.tunable_params.get_hyperparameter_names()
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
            old_max_evals = self.user_args['max-evals']
            self.user_args['max-evals'] -= n_prepared
            # When resuming, we never want to actually use ask_initial() so have that function point to ask()
            def wrap_initial(n_points=1):
                points = ytoptimizer.ask(n_points=n_points)
                return list(points)[0]
            ytoptimizer.ask_initial = wrap_initial
            print(f"Optimizer updated and ready to resume -- max-evals reduced {old_max_evals} --> {self.user_args['max-evals']}")

        self.gen_specs['user']['ytoptimizer'] = ytoptimizer

if __name__ == '__main__':
    # Parse comms, default options from commandline, then execute
    libE_heFFTe().run()

