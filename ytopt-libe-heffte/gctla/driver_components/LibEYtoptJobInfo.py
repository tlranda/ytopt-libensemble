# Local imports
from .JobInfo import JobInfo

class YtoptJobInfo(JobInfo):
    basic_job_string = "python3 libEwrapper.py "+\
                        "--system polaris "+\
                        "--ens-template wrapper_components/run_heFFTe_ytopt.py "+\
                        "--mpi-ranks {n_ranks} "+\
                        "--gpu-override 4 --gpu-enabled "+\
                        "--application-x {app_scale[0]} --application-y {app_scale[1]} --application-z {app_scale[2]} "+\
                        "--ensemble-workers {workers} --max-evals {n_records} "+\
                        "--configure-environment craympi "+\
                        "--machine-identifier polarisSlingshot11 "+\
                        "--ens-dir-path polarisSlingshot11_{n_nodes}n_{app_scale[0]}_{app_scale[1]}_{app_scale[2]}a "+\
                        "--launch-job --display-results"
    """
    # DEPRECATED Theta system : Extending existing results
    basic_job_string = "python3 libEwrapper.py "+\
                       "--system theta "+\
                       "--mpi-ranks {n_ranks} "+\
                       "--cpu-override 256 --cpu-ranks-per-node 64 "+\
                       "--application-scale {app_scale} "+\
                       "--ensemble-workers {workers} --max-evals {n_records} "+\
                       "--configure-environment craympi "+\
                       "--machine-identifier theta-knl "+\
                       "--ens-dir-path Theta_Extend_{n_nodes}n_{app_scale}a "+\
                       "--resume {resume} --launch-job --display-results"
    """
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

