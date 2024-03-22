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
import numpy as np
import pandas as pd

from wrapper_components.run_libE import libE_base, argcasts_dict

from GC_TLA.implementations.heFFTe.heFFTe_problem import heFFTeArchitecture, heFFTe_instance_factory
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


from wrapper_components.gctla_asktell import persistent_gctla # Generator function, communicates with ytopt optimizer
from wrapper_components.libE_objective import heFFTe_objective # Simulator function, calls Plopper

def remove_generated_duplicates(samples, history, dtypes):
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
    argument_casts_extensions = [
        ('input', True, argcasts_dict['single_listwrap'], None),
        ('constraint-sys', True, int, None),
        ('constraint-app-x', True, int, None),
        ('constraint-app-y', True, int, None),
        ('constraint-app-z', True, int, None),
        ('ignore', False, argcasts_dict['single_listwrap'], []),
        ('auto-budget', False, argcasts_dict['boolcast'], False),
        ('initial-quantile', False, float, 0.1),
        ('min-quantile', False, float, 0.15),
        ('budget-confidence', False, float, 0.95),
        ('quantile-reduction', False, float, 0.1),
        ('ideal-proportion', False, float, 0.1),
        ('ideal-attrition', False, float, 0.05),
        ('determine-budget-only', False, argcasts_dict['boolcast'], False),
        ('predictions-only', False, argcasts_dict['boolcast'], False),
    ]

    def build_model(self):
        # Load data
        cand_files = [pathlib.Path(_) for _ in self.user_args['input'] if _ not in self.user_args['ignore']]
        found = [_ for _ in cand_files if _.exists()]
        if len(found) != len(cand_files):
            missing = set(cand_files).difference(set(found))
            warnings.warn(f"Input file(s) not found: {missing}", UserWarning)
        data = pd.concat([pd.read_csv(_) for _ in found]).reset_index(names=["CSV_Order"])
        # Drop non-SDV cols by only using SDV-OK cols
        training_cols = [_ for _ in self.problem.tunable_params] + ['mpi_ranks', 'threads_per_node', 'ranks_per_node', 'FLOPS']
        # These columns are needed for consistency, but not for SDV learning
        SDV_NONPREDICT = ['threads_per_node','ranks_per_node','FLOPS']
        # Drop erroneous configurations
        least_infinity = min([self.problem.executor.infinity[_] for _ in MetricIDs if _ != MetricIDs.OK and _ in self.problem.executor.infinity])
        train_data = data.loc[:, training_cols]
        train_data = train_data[train_data['FLOPS'] < least_infinity]
        # Recontextualize topology data
        topo_split = lambda x: [int(_) for _ in x.split(' ') if _ not in ['-ingrid','-outgrid']]
        np_topologies = np.asarray([np.fromstring(_, dtype=int, sep=' ') for _ in self.problem.architecture.mpi_topologies])
        for topology_key, grid_type in zip(['P6','P7'], ['-ingrid', '-outgrid']):
            # Grab string as 3-integer topology, downsample with logarithm
            top_as_np_log = np.log2(np.vstack(train_data[topology_key].apply(topo_split)))
            # Get max logarithm for each dimension/row
            log_mpi_ranks = np.stack([np.log2(train_data['mpi_ranks'])]*3, axis=1)
            # Change proportion of logarithm to new base for transferred topology sizes
            projection = 2 ** (top_as_np_log / log_mpi_ranks * np.log2(self.problem.architecture.mpi_ranks))
            # Use nearest-topology search for best match in new topology space
            distances = np.asarray([((np_topologies - p) ** 2).sum(axis=1) for p in projection])
            best_match = np.argmin(distances, axis=1)
            new_topologies = np_topologies[best_match]
            # Return to string
            str_topologies = [" ".join([grid_type]+[str(_) for _ in topo]) for topo in new_topologies]
            train_data[topology_key] = str_topologies
        # Recontextualize sequences
        if 'P8' in self.problem.tunable_params:
            dest_seq = np.asarray(self.problem.architecture.thread_sequence)
            # Group by value
            for ((t,r), subframe) in train_data.groupby(['threads_per_node','ranks_per_node']):
                max_select = t // r
                cur_vals = subframe['P8']
                # Reconstruct available sequence at time of this program's execution
                cur_seq = np.asarray(self.problem.architecture.make_thread_sequence(t,r))
                # Project values via ratio length in list
                projection = (np.asarray([np.argmax((cur_seq == c)) for c in cur_vals]) / len(cur_seq) * len(dest_seq)).round(0).astype(int)
                train_data.loc[subframe.index, 'P8'] = dest_seq[projection]

        # DATA PREPARED
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(train_data.drop(columns=SDV_NONPREDICT))

        # Condition will make batches of 100 samples at a time
        conditions = [Condition({'mpi_ranks': self.user_args['constraint-sys'],
                                 'P1X': self.user_args['constraint-app-x'],
                                 'P1Y': self.user_args['constraint-app-y'],
                                 'P1Z': self.user_args['constraint-app-z']},
                                num_rows=100)]
        self.conditions = conditions

        # Condition for budget calculation
        mass_conditions = copy.deepcopy(conditions)
        mass_conditions[0].num_rows = self.problem.tuning_space_size

        # Fitting process
        accepted_model = None
        suggested_budget = None
        model = GaussianCopula(metadata, enforce_min_max_values=False)
        model.add_constraints(constraints=problem.constraints)
        while accepted_model is None:
            fittable = train_data[train_data['FLOPS'] <= train_data['FLOPS'].qunatile(self.user_args['initial-quantile'])]
            fittable = fittable.drop(columns=['FLOPS'])
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                model.fit(fittable)
            # Quick-exit if not auto-budgeting
            if not self.user_args['auto-budget']:
                accepted_model = model
                suggested_budget = self.user_args['max-evals']
                continue
            # Check expected budget
            mass_sample = model.sample_from_conditions(mass_condition)
            # HyperGeometric arguments
            total_population = self.problem.tuning_space_size
            sample_population = len(mass_sample.drop_duplicates())
            ideal_samples = int(total_population * self.user_args['ideal-proportion'])
            subideal_samples = max(1, ideal_samples - int((total_population-sample_population) * self.user_args['ideal-attrition']))
            print(f"Population {total_population} | Sampleable {sample_population} | Ideal {ideal_samples} | Ideal with Attrition {subideal_samples}")
            if subideal_samples > sample_population:
                print(f"Autotuning budget indeterminate at quantile {self.user_args['initial-quantile']}")
                suggested_budget = self.user_args['max-evals']
            else:
                suggested_budget = 0
                tightest_budget = min(subideal_samples, self.user_args['max-evals'])
                while suggested_budget < tightest_budget:
                    suggested_budget += 1
                    confidence = sum([hypergeo(subideal_samples, sample_population, _, suggested_budget) for _ in range(1, suggested_budget+1)])
                    # Do not process higher budget with explicitly greater confidence
                    if confidence >= self.user_ags['budget-confidence']:
                        print(f"Autotuning budget {suggested_budget} accepted at quantile {self.user_args['initial-quantile']} (confidence: {confidence})")
                        accepted_model = model
                        break
                if confidence < self.user_args['budget-confidence']:
                    print(f"Autotuning budget at quantile {self.user_args['initial-quantile']} failed to satisfy confidence {self.user_args['budget-confidence']}; max confidence: {confidence}")
            self.user_args['initial-quantile'] -= self.user_args['quantile-reduction']
            if self.user_args['initial-quantile'] <= self.user_args['min-quantile']:
                print("No autotuning budgets can be satisfied under given constraints")
                exit()
        if self.user_args['determine-budget-only']:
            exit()
        self.model = accepted_model

    def set_problem(self):
        # Building the problem would perform architecture detection for us, but right now the nodes aren't
        # identified by libEwrapper so we have to out-of-order this
        arch = heFFTeArchitecture(machine_identifier=self.MACHINE_IDENTIFIER,
                                  hostfile=self.user_args['node-list-file'],
                                  x=self.APP_SCALE_X,
                                  y=self.APP_SCALE_Y,
                                  z=self.APP_SCALE_Z,
                                  workers=self.num_sim_workers)
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

        # Also set up model and conditions here
        self.build_model()

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
        self.gen_specs['user'].update({
            'model': self.model,
            'conditions': self.conditions,
            'remove_duplicates': remove_generated_duplicates,
        })

    def prepare_libE(self):
        super().prepare_libE()
        if self.user_args['predictions_only']:
            raw_predictions = self.model.sample_from_conditions(self.conditions)
            cleaned, history = remove_generated_duplicates(raw_predictions, [], self.gen_specs['out'])
            self.libE_specs['ensemble_dir_path'].mkdir(parents=True, exist_ok=True)
            cleaned.to_csv(self.libE_specs['ensemble_dir_path'].joinpath('predicted_results.csv'), index=False)
            exit()


if __name__ == '__main__':
    # Parse comms, default options from commandline, then execute
    libE_heFFTe().run()


