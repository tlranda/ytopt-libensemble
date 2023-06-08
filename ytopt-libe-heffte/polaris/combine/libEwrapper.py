import subprocess
import argparse

def build():
    parser = argparse.ArgumentParser()
    scaling = parser.add_argument_group("Scaling", "Arguments that control scale of evaluated tests")
    scaling.add_argument("--worker-nodes", type=int, default=2,
                        help="Number of MPI nodes each Worker uses (System Scale; default: %(DEFAULT)s)")
    scaling.add_argument("--worker-timeout", type=int, default=100,
                        help="Timeout for Worker subprocesses (default: %(DEFAULT)s)")
    ensemble = parser.add_argument_group("LibEnsemble", "Arguments that control libEnsemble behavior")
    ensemble.add_argument("--n-workers", type=int, default=1,
                        help="Number of libEnsemble workers (EXCLUDING MANAGER; default: %(DEFAULT)s)")
    ensemble.add_argument("--max-evals", type=int, default=4,
                        help="Maximum evaluations collected by ensemble (default: %(DEFAULT)s)")
    ensemble.add_argument("--comms", choices=['mpi','local'], default='local',
                        help="Which call setup is used for libEnsemble itself (default: %(DEFAULT)s)")
    files = parser.add_argument_group("Files", "Arguments that dictate input and output files")
    files.add_argument("--libensemble-target", type=str, default="ytopt_libensemble.py",
                        help="Script invoked as libEnsemble caller (default: %(DEFAULT)s)")
    files.add_argument("--plopper-target", type=str, default="plopper.py",
                        help="Plopper that libEnsemble caller will use (default: %(DEFAULT)s)")
    files.add_argument("--problem-target", type=str, default="problem.py",
                        help="Problem that libEnsemble caller will use (default: %(DEFAULT)s)")
    files.add_argument("--exe-target", type=str, default="exe.pl",
                        help="Perl execution script that libEnsemble caller eventually gets to (default: %(DEFAULT)s)")
    files.add_argument("--generated-script", type=str, default="qsub.batch",
                        help="Qsub script is written to this location, then executed")
    return parser

def parse(prs=None, args=None):
    if prs is None:
        prs = build()
    if args is None:
        args = prs.parse_args()
    # Designated sed arguments
    args.designated_sed = {
        'worker_nodes': (args.exe_target, "s/N_NODES = [0-9]*;/N_NODES = {};/"),
        'worker_timeout': (args.plopper_target, "s/app_timeout = [0-9]*/app_timeout = {}/"),
    }
    return args

if __name__ == '__main__':
    args = parse()
    # Make all substitutions based on arguments
    for (sed_arg, (target_file, substitution)) in args.designated_sed.items():
        args_value = getattr(args, sed_arg)
        sed_command = ['sed','-i', substitution.format(args_value), target_file]
        #print(" ".join(sed_command))
        proc = subprocess.run(sed_command, capture_output=True)
        #print(proc.stdout.decode('utf-8'))
        #print(proc.stderr.decode('utf-8'))
    # Prepare job
    job_contents = f"""#!/bin/bash -x
#PBS -l walltime=01:00:00
#PBS -l select={1+(args.worker_nodes*args.n_workers)}:system=polaris
#PBS -l filesystems=home:grand:eage
#PBS -A EE-ECP
#PBS -q prod

# Script to run libEnsemble using multiprocessing on launch nodes.
# Assumes Conda environment is set up.

# To be run with central job management
# - Manager and workers run on launch node.
# - Workers submit tasks to the nodes in the job available (using exe.pl)

# Name of calling script
export EXE={args.libensemble_target}

# Communication Method
export COMMS="--comms {args.comms}"

# Number of workers. For multiple nodes per worker, have nworkers be a divisor of nnodes, then add 1
# e.g. for 2 nodes per worker, set nnodes = 12, nworkers = 7
export NWORKERS="--nworkers {1+args.n_workers}"  # extra worker running generator (no resources needed)

export EVALS="--max-evals {args.max_evals}"

# Name of Conda environment
export CONDA_ENV_NAME=ytune_23

export PMI_NO_FORK=1 # Required for python kills on Theta

# Load Polaris modules
module load cudatoolkit-standalone/12.0.0;
module load conda;

# Activate conda environment
export PYTHONNOUSERSITE=1
conda activate \$CONDA_ENV_NAME

# Ensure proper MPI and other libraries are used (not Conda MPI etc)
module swap PrgEnv-gnu PrgEnv-nvhpc/8.3.3;

# Launch libE
pycommand="python $EXE $COMMS $NWORKERS --learner=RF $EVALS" # > out.txt 2>&1"
echo "$pycommand";
eval "$pycommand";
    """
    with open(args.generated_script, 'w') as f:
        f.write(job_contents)


