# Local imports
from .JobInfo import JobInfo

class GCJobInfo(JobInfo):
    basic_job_string = "python3 libEwrapper.py "+\
                        "--system polaris "+\
                        "--ens-template wrapper_components/run_heFFTe_gctla.py "+\
                        "--mpi-ranks {n_ranks} "+\
                        "--gpu-override 4 --gpu-enabled "+\
                        "--application-x {app_scale[0]} --application-y {app_scale[1]} --application-z {app_scale[2]} "+\
                        "--ensemble-workers {workers} --max-evals {n_records} "+\
                        "--configure-environment craympi "+\
                        "--machine-identifier polarisSlingshot11 "+\
                        "--ens-dir-path polarisSlingshot11_{n_nodes}n_{app_scale[0]}_{app_scale[1]}_{app_scale[2]}a "+\
                        "--gc-input source_tasks/*/ensemble_*/manager_results.csv "+\
                        "--gc-ignore source_tasks/polarisSlingshot11_{n_nodes}n_{app_scale[0]}_{app_scale[1]}_{app_scale[2]}a/ensemble_polarisSlingshot11_{n_nodes}n_{app_scale[0]}_{app_scale[1]}_{app_scale[2]}a*/manager_results.csv "+\
                        "--launch-job --display-results"
    # Polaris system : Collecting new datasets
    def __init__(self, spec, n_nodes, n_ranks, app_scale, workers, n_records, resume):
        self.spec = spec
        self.n_nodes = n_nodes
        self.n_ranks = n_ranks
        self.app_scale = app_scale
        self.workers = workers
        self.n_records = n_records
        self.resume = resume
    @property
    def nodes(self):
        return int(self.n_nodes) * self.workers

