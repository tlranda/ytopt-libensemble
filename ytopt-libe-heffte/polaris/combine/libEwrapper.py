import subprocess
import argparse
import os, stat, shutil
import warnings

def build():
    parser = argparse.ArgumentParser()
    parser.add_argument("--error-demotion", action='store_true',
                        help="Demote wrapper errors to warnings, may permit unwanted behaviors (default: Error and cease operation)")
    #SCALING
    scaling = parser.add_argument_group("Scaling", "Arguments that control scale of evaluated tests")
    scaling.add_argument("--mpi-ranks", type=int, default=2,
                        help="Number of MPI ranks each Worker uses (System Scale; default: %(default)s)")
    scaling.add_argument("--worker-timeout", type=int, default=100,
                        help="Timeout for Worker subprocesses (default: %(default)s)")
    # choices=[64,128,256,512,1024]
    scaling.add_argument("--application-scale", type=int, default=128,
                        help="Problem size to be optimized (default: %(default)s)")
    # Polaris-*: 64
    # Theta-KNL: 256
    # Theta-GPU: 128
    scaling.add_argument("--system", choices=["polaris", "theta", "theta-gpu", ], default="theta",
                        help="System sets default # CPU ranks per node (default: %(default)s)")
    scaling.add_argument("--cpu-override", type=int, default=None,
                        help="Override automatic CPU detection to set max_cpu value (default: Detect)")
    scaling.add_argument("--gpu-enabled", action="store_true",
                        help="Enable GPU treatment for libensemble (default: Disabled)")
    # ENSEMBLE
    ensemble = parser.add_argument_group("LibEnsemble", "Arguments that control libEnsemble behavior")
    ensemble.add_argument("--ensemble-workers", type=int, default=1,
                        help="Number of libEnsemble workers (EXCLUDING MANAGER; default: %(default)s)")
    ensemble.add_argument("--max-evals", type=int, default=4,
                        help="Maximum evaluations collected by ensemble (default: %(default)s)")
    ensemble.add_argument("--comms", choices=['mpi','local'], default='local',
                        help="Which call setup is used for libEnsemble itself (default: %(default)s)")
    ensemble.add_argument("--configure-environment", choices=["none", "polaris", "craympi"], nargs="*",
                        help="Set up environment based on known systems (default: No customization)")
    ensemble.add_argument("--machine-identifier", type=str, default=None,
                        help="Used to identify if a configuration's performance is already known on this machine (default: HOSTNAME)")
    ensemble.add_argument("--ensemble-dir-path", type=str, default=None,
                        help="Set the ensemble dir path (default: Picks a suitably random extension to avoid collisions)")
    ensemble.add_argument("--ensemble-path-randomization", action='store_true',
                        help="Add randomization to customized ensemble-dir-path's (default: NOT added)")
    ensemble.add_argument("--launch-job", action='store_true',
                        help="Launch job once prepared (default: NOT launched, job script only written)")
    # GaussianCopula
    gc = parser.add_argument_group("GaussianCopula", "Arguments for Gaussian Copula (must use --libensemble-target=run_gctla.py)")
    gc.add_argument("--gc-sys", type=int, default=None,
                    help="# MPI ranks targeted by constrained GC (default: Defers to --mpi-ranks)")
    gc.add_argument("--gc-app", type=int, default=None,
                    help="Application size targeted by constrained GC (default: Defers to --application-scale)")
    # SEEDS
    seeds = parser.add_argument_group("Seeds", "Arguments that control randomization seeding")
    seeds.add_argument("--seed-configspace", type=int, default=1234,
                        help="Seed for the ConfigurationSpace object (default: %(default)s)")
    seeds.add_argument("--seed-ytopt", type=int, default=2345,
                        help="Seed for the Ytopt Optimizer (default: %(default)s)")
    seeds.add_argument("--seed-numpy", type=int, default=1,
                        help="Seed for Numpy default random stream (default: %(default)s)")
    # FILES
    files = parser.add_argument_group("Files", "Arguments that dictate input and output files")
    files.add_argument("--libensemble-target", type=str, default="run_ytopt.py",
                        help="Template script invoked as libEnsemble caller (default: %(default)s)")
    files.add_argument("--libensemble-export", type=str, default="run_ytopt.py",
                        help="Perform modifications to template script to this name -- useful if scheduling multiple jobs that could simultaneously edit the template (default: %(default)s)")
    files.add_argument("--libensemble-randomization", action="store_true",
                        help="Add randomization to customized libensemble export name (default: NOT added)")
    files.add_argument("--generated-script", type=str, default="qsub.batch",
                        help="Qsub script is written to this location, then executed (default: %(default)s)")
    files.add_argument("--display-results", action='store_true',
                        help="ONLY active when --launch-job supplied; display results when job completes (default: NOT displayed)")
    return parser

def parse(prs=None, args=None):
    WarningsAsErrors = "Error thrown due to warning (disable via --error-demotion=True)"
    if prs is None:
        prs = build()
    if args is None:
        args = prs.parse_args()
    # Ensure string is edited in file to be None rather than using None as a str-like object
    if args.ensemble_dir_path is not None:
        args.ensemble_dir_path = '"' + args.ensemble_dir_path
    else:
        args.ensemble_dir_path = ''
    if args.ensemble_path_randomization:
        import secrets
        if args.ensemble_dir_path == '':
            args.ensemble_dir_path = '"'
        args.ensemble_dir_path += "_"+secrets.token_hex(nbytes=4)
    args.ensemble_dir_path += '"' # close quote
    if args.ensemble_dir_path == '"':
        args.ensemble_dir_path = '""'
    # Environments
    if args.configure_environment is None:
        args.configure_environment = []
    # Machine identifier selection
    if args.machine_identifier is None:
        import platform
        args.machine_identifier = f'"{platform.node()}"'
    else:
        args.machine_identifier = f'"{args.machine_identifier}"'
    if args.cpu_override is None:
        args.cpu_override = "None"
    else:
        args.cpu_override = str(args.cpu_override)
    if args.system not in args.machine_identifier:
        MI_System_Mismatch = f"Indicated system ({args.system}) name not found in machine identifier ({args.machine_identifier}). This could result in improper #cpu_ranks_per_node and lead to over- or under-subscription of resources!"
        warnings.warn(MI_System_Mismatch)
        if not args.error_demotion:
            raise ValueError(WarningsAsErrors)
    match args.system:
        case "polaris":
            args.cpu_ranks_per_node = 64
        case "theta":
            args.cpu_ranks_per_node = 256
        case "theta-gpu":
            args.cpu_ranks_per_node = 128
        case _:
            args.cpu_ranks_per_node = 1
    args.gpu_enabled = str(args.gpu_enabled)
    if args.libensemble_randomization:
        import secrets
        try:
            args.libensemble_export, extension = args.libensemble_export.rsplit('.',1)
        except ValueError:
            extension = 'py'
        args.libensemble_export += secrets.token_hex(nbytes=4)
        args.libensemble_export += '.'+extension
    protected = ['run_ytopt.py', 'run_gctla.py']
    if args.libensemble_export in protected:
        backup = args.libensemble_export + ".bak"
        increment = None
        while os.path.exists(backup):
            if increment is None:
                backup += "_1"
                increment = 1
            else:
                increment += 1
                backup = backup.rsplit("_",1)[0] + str(increment)
        TemplateOverride = f"Processing will override template {args.libensemble_export}, which may not be desired!"
        if args.error_demotion:
            TemplateOverride += f" Protecting the original template by backing it up to {backup}"
        warnings.warn(TemplateOverride)
        if not args.error_demotion:
            raise ValueError(WarningsAsErrors)
        shutil.copy2(args.libensemble_export, backup)
    # Gaussian Copula arguments
    if args.gc_sys is None:
        args.gc_sys = args.mpi_ranks
    if args.gc_app is None:
        args.gc_app = args.application_scale
    # Designated sed arguments
    args.designated_sed = {
        'mpi_ranks': [(args.libensemble_export, "s/MPI_RANKS = [0-9]*/MPI_RANKS = {}/"),],
        'worker_timeout': [(args.libensemble_export, "s/'app_timeout': [0-9]*,/'app_timeout': {},/")],
        'application_scale': [(args.libensemble_export, "s/APP_SCALE = [0-9]*/APP_SCALE = {}/")],
        'ensemble_dir_path': [(args.libensemble_export, "s/^ENSEMBLE_DIR_PATH = .*/ENSEMBLE_DIR_PATH = {}/")],
        'machine_identifier': [(args.libensemble_export, "s/MACHINE_IDENTIFIER = .*/MACHINE_IDENTIFIER = {}/")],
        'cpu_override': [(args.libensemble_export, "s/cpu_override = .*/cpu_override = {}/")],
        'cpu_ranks_per_node': [(args.libensemble_export, "s/cpu_ranks_per_node = .*/cpu_ranks_per_node = {}/")],
        'gpu_enabled': [(args.libensemble_export, "s/gpu_enabled = .*/gpu_enabled = {}/")],
        'seed_configspace': [(args.libensemble_export, "s/CONFIGSPACE_SEED = .*/CONFIGSPACE_SEED = {}/")],
        'seed_ytopt': [(args.libensemble_export, "s/YTOPT_SEED = .*/YTOPT_SEED = {}/")],
        'seed_numpy': [(args.libensemble_export, "s/NUMPY_SEED = .*/NUMPY_SEED = {}/")],
    }
    if 'gc' in args.libensemble_export:
        args.bonus_runtime_args = f"--constraint-sys {args.gc_sys} --constraint-app {args.gc_app}"
    else:
        args.bonus_runtime_args = ""
    return args

if __name__ == '__main__':
    args = parse()
    # Make all substitutions based on arguments
    shutil.copy2(args.libensemble_target, args.libensemble_export)
    for (sed_arg, substitution_specs) in args.designated_sed.items():
        for (target_file, substitution) in substitution_specs:
            args_value = getattr(args, sed_arg)
            print(f"Set {sed_arg} to {args_value} in {target_file}")
            sed_command = ['sed','-i', substitution.format(args_value), target_file]
            proc = subprocess.run(sed_command, capture_output=True)
            if proc.returncode != 0:
                print(proc.stdout.decode('utf-8'))
                print(proc.stderr.decode('utf-8'))
                raise ValueError(f"sed Substitution for '{sed_arg}' in '{target_file}' Failed")
            else:
                print("Adjustment OK!")
    known_environments = {
    "none": "# pass",
    "polaris": """
# Name of Conda environment
export CONDA_ENV_NAME=ytune_23

export PMI_NO_FORK=1 # Required for python kills on Theta

# Load Polaris modules
module load cudatoolkit-standalone/12.0.0;
module load conda;

# Activate conda environment
export PYTHONNOUSERSITE=1
conda activate $CONDA_ENV_NAME

# Ensure proper MPI and other libraries are used (not Conda MPI etc)
module swap PrgEnv-gnu PrgEnv-nvhpc/8.3.3;
""",
    "craympi": """
export IBV_FORK_SAFE=1; # May fix some MPI issues where processes call fork()
"""
    }
    # Prepare job
    env_adjust = "\n".join([known_environments[env] for env in args.configure_environment])
    job_contents = f"""#!/bin/bash -x
#PBS -l walltime=01:00:00
#PBS -l select={1+(args.mpi_ranks*args.ensemble_workers)}:system={args.system}
#PBS -l filesystems=home:grand:eagle
#PBS -A EE-ECP
#PBS -q prod

# Script output should indicate basic information
echo "$HOSTNAME";
date;
echo;

# Script to run libEnsemble using multiprocessing on launch nodes.
# Assumes Conda environment is set up.

# To be run with central job management
# - Manager and workers run on launch node.
# - Workers submit tasks to the nodes in the job available via scheduler (mpirun, aprun, etc) in plopper

# Name of calling script
export EXE={args.libensemble_export}

# Communication Method
export COMMS="--comms {args.comms}"

# Number of workers. For multiple nodes per worker, have nworkers be a divisor of nnodes, then add 1
# e.g. for 2 nodes per worker, set nnodes = 12, nworkers = 7
export NWORKERS="--nworkers {1+args.ensemble_workers}"  # extra worker running generator (no resources needed)

export EVALS="--max-evals {args.max_evals}"

# ADJUST ENVIRONMENT
{env_adjust}

# Launch libE
pycommand="python $EXE $COMMS $NWORKERS --learner=RF $EVALS {args.bonus_runtime_args}" # > out.txt 2>&1"
echo "$pycommand";
eval "$pycommand";
echo;
date;
echo;
"""
    print(f"Produce job script {args.generated_script}")
    with open(args.generated_script, 'w') as f:
        f.write(job_contents)
    # Set RWX for owner, RX for all others
    os.chmod(args.generated_script, stat.S_IRWXU |
                                    stat.S_IRGRP | stat.S_IXGRP |
                                    stat.S_IROTH | stat.S_IXOTH)
    print("OK!")
    ensemble_operating_dir = f"./ensemble_{args.ensemble_dir_path[1:-1]}"
    if args.launch_job:
        proc = subprocess.run(f"./{args.generated_script}")
        migrated_job_script = f"{ensemble_operating_dir}/{args.generated_script}"
        migrations = [(args.generated_script, migrated_job_script, "Job Script"),
                      ('ensemble.log', f"{ensemble_operating_dir}/ensemble.log", "Ensemble logs"),]
        # If the target was modified in-place, do NOT move it
        if args.libensemble_target != args.libensemble_export:
            migrations.append((args.libensemble_export, f"{ensemble_operating_dir}/{args.libensemble_export}", "LibEnsemble script"))
        for migration_from, migration_to, identifier in migrations:
            try:
                os.rename(migration_from, migration_to)
            except:
                print(f"{identifier} could not be migrated -- {migration_from}")
            else:
                print(f"{identifier} migrated to ensemble directory -- {migration_to}")
        if proc.returncode == 0 and args.display_results:
            import pandas as pd
            print("Finished evaluations")
            print(pd.read_csv(f"{ensemble_operating_dir}/manager_results.csv"))
            try:
                print("Unfinished evaluations")
                print(pd.read_csv(f"{ensemble_operating_dir}/unfinished_results.csv"))
            except:
                print("No unfinished evaluations to view")

