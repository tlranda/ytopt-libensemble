import subprocess
import argparse
import os, pathlib, stat, shutil
import secrets
import warnings
import signal

def build():
    parser = argparse.ArgumentParser()
    parser.add_argument("--error-demotion", action='store_true',
                        help="Demote wrapper errors to warnings, may permit unwanted behaviors (default: Error and cease operation)")
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
    # Files
    files = parser.add_argument_group("Files", "Arguments to control file management")
    files.add_argument("--ens-dir-path", default=None,
                       help="Name the ensemble directory suffix (default: prefix 'ensemble_' has no custom suffix)")
    files.add_argument("--ens-static-path", action='store_true',
                       help="Disable randomization of ensemble directory suffix (default: Added to prevent filename collisions LibEnsemble cannot tolerate)")
    files.add_argument("--ens-template", default="run_ytopt.py",
                       help="Template for libEnsemble driver (default: %(default)s)")
    files.add_argument("--ens-script", default="qsub.batch",
                       help="Template for job submission script that calls libEnsemble driver (default: %(default)s)")
    files.add_argument("--ens-static", action='store_true',
                       help="Override templates rather than copying to unique name (default: Vary name to prevent multi-job clobbering)")
    files.add_argument("--resume", nargs="*", default=None,
                       help="CSV files to treat as prior history without re-evaluating on the system (default: none; only operable for --ens-template=run_ytopt.py)")
    # SEEDS
    seeds = parser.add_argument_group("Seeds", "Arguments that control randomization seeding")
    seeds.add_argument("--seed-configspace", type=int, default=1234,
                        help="Seed for the ConfigurationSpace object (default: %(default)s)")
    seeds.add_argument("--seed-ytopt", type=int, default=2345,
                        help="Seed for the Ytopt Optimizer (default: %(default)s)")
    seeds.add_argument("--seed-numpy", type=int, default=1,
                        help="Seed for Numpy default random stream (default: %(default)s)")
    # Job management
    job = parser.add_argument_group("Job", "Arguments to control job behavior")
    job.add_argument("--launch-job", action='store_true',
                     help="Execute job script after writing it (default: prepare script only)")
    job.add_argument("--display-results", action='store_true',
                     help="ONLY active when --launch-job supplied; display results when job completes (default: NOT displayed)")
    #SCALING
    scaling = parser.add_argument_group("Scaling", "Arguments that control scale of evaluated tests")
    scaling.add_argument("--mpi-ranks", type=int, default=2,
                        help="Number of MPI ranks each Worker uses (System Scale; default: %(default)s)")
    scaling.add_argument("--worker-timeout", type=int, default=100,
                        help="Timeout for Worker subprocesses (default: %(default)s)")
    # choices=[64,128,256,512,1024]
    scaling.add_argument("--application-scale", type=int, default=128,
                        help="Default FFT dimension scale for unspecified --application-[xyz] arguments (default: %(default)s)")
    scaling.add_argument("--application-x", type=int, default=None,
                        help="FFT size in X-dimension (default: --application-scale value)")
    scaling.add_argument("--application-y", type=int, default=None,
                        help="FFT size in Y-dimension (default: --application-scale value)")
    scaling.add_argument("--application-z", type=int, default=None,
                        help="FFT size in Z-dimension (default: --application-scale value)")
    # Polaris-*: 64
    # Theta-KNL: 256
    # Theta-GPU: 128
    scaling.add_argument("--system", choices=["polaris", "theta", "theta-gpu", ], default="theta",
                        help="System sets default # CPU ranks per node (default: %(default)s)")
    scaling.add_argument("--cpu-override", type=int, default=None,
                        help="Override automatic CPU detection to set max_cpu value (default: Detect)")
    scaling.add_argument("--cpu-ranks-per-node", type=int, default=None,
                        help="Override #ranks per node on CPUs (default: #threads on cpu or --cpu-override when specified)")
    scaling.add_argument("--gpu-enabled", action="store_true",
                        help="Enable GPU treatment for libensemble (default: Disabled)")
    scaling.add_argument("--gpu-override", type=int, default=None,
                        help="Override automatic GPU detection to set max_gpu value (default: Detect when --gpu-enabled)")
    # GaussianCopula
    gc = parser.add_argument_group("GaussianCopula", "Arguments for Gaussian Copula (must use --libensemble-target=run_gctla.py)")
    gc.add_argument("--gc-sys", type=int, default=None,
                    help="GC's target # MPI ranks (default: Defers to --mpi-ranks)")
    gc.add_argument("--gc-app", type=int, default=None,
                    help="Default FFT dimension scale for unspecified --gc-app-[xyz] arguments (default: Defers to --application-scale)")
    gc.add_argument("--gc-app-x", type=int, default=None,
                    help="GC's target FFT size in the X-dimension (default: --gc-app value)")
    gc.add_argument("--gc-app-y", type=int, default=None,
                    help="GC's target FFT size in the Y-dimension (default: --gc-app value)")
    gc.add_argument("--gc-app-z", type=int, default=None,
                    help="GC's target FFT size in the Z-dimension (default: --gc-app value)")
    gc.add_argument("--gc-input", nargs="+", default=None,
                    help="Inputs to provide to the GC (no default; must be specified!)")
    gc.add_argument("--gc-ignore", nargs="*", default=None,
                    help="Ignore list for the input list (helpful for over-eager globbing; no default)")
    gc.add_argument("--gc-auto-budget", action='store_true',
                    help="Utilize auto-budgeting in libensemble-target (default: not used)")
    gc.add_argument("--gc-determine-budget-only", action='store_true',
                    help="Exit immediately after auto-budgeting in libensemble-target (default: continue with autotuning)")
    gc.add_argument("--gc-initial-quantile", type=float, default=None,
                    help="Initial quantile to start auto-budgeting from (default: libensemble-target's default)")
    gc.add_argument("--gc-min-quantile", type=float, default=None,
                    help="Minimum quantile to use in auto-budgeting (default: libensemble-target's default)")
    gc.add_argument("--gc-budget-confidence", type=float, default=None,
                    help="Required confidence to accept a budget in auto-budgeting (default: libensemble-target's default")
    gc.add_argument("--gc-quantile-reduction", type=float, default=None,
                    help="Amount to reduce target quantile by each auto-budgeting iteration (default: libensemble-target's default)")
    gc.add_argument("--gc-ideal-proportion", type=float, default=None,
                    help="Ideal proportion of search space to target in auto-budgeting (default: libensemble-target's default)")
    gc.add_argument("--gc-ideal-attrition", type=float, default=None,
                    help="Attrition rate from ideal portion after the GC constrains the space (default: libensemble-target's default)")
    gc.add_argument("--gc-predictions-only", action='store_true',
                    help="GC only produces predictions (default: empirically evaluate predictions for TL autotuning)")
    return parser

def parse(prs=None, args=None):
    WarningsAsErrors = "Error thrown due to warning (disable via --error-demotion=True)"
    if prs is None:
        prs = build()
    if args is None:
        args = prs.parse_args()
    # Quote the ensemble dir name and add randomization
    if args.ens_dir_path is None:
        args.ens_dir_path = ''
    if not args.ens_static_path:
        dpath = pathlib.Path(args.ens_dir_path)
        args.ens_dir_path = dpath.parent.joinpath(dpath.stem + "_" + secrets.token_hex(nbytes=4))
    args.ens_dir_path = f'"{args.ens_dir_path}"'

    # Similar treatment for template and script
    args.ens_template_export = args.ens_template
    args.ens_script_export = args.ens_script
    if not args.ens_static:
        template = pathlib.Path(args.ens_template_export)
        args.ens_template_export = template.parent.joinpath(template.stem + '_' + secrets.token_hex(nbytes=4) + template.suffix)
        script = pathlib.Path(args.ens_script_export)
        args.ens_script_export = script.parent.joinpath(script.stem + '_' + secrets.token_hex(nbytes=4) + script.suffix)
    if type(args.resume) is str:
        args.resume = [args.resume]

    # Environments
    if args.configure_environment is None:
        args.configure_environment = []

    # Machine identifier selection
    if args.machine_identifier is None:
        import platform
        args.machine_identifier = f'"{platform.node()}"'
    else:
        args.machine_identifier = f'"{args.machine_identifier}"'

    # Scaling and system overides
    if args.application_x is None:
        args.application_x = 'APP_SCALE'
    if args.application_y is None:
        args.application_y = 'APP_SCALE'
    if args.application_z is None:
        args.application_z = 'APP_SCALE'
    if args.cpu_override is None:
        args.cpu_override = "None"
    else:
        args.cpu_override = str(args.cpu_override)
    if args.cpu_ranks_per_node is None:
        match args.system:
            case "polaris":
                args.cpu_ranks_per_node = 64
            case "theta":
                args.cpu_ranks_per_node = 256
            case "theta-gpu":
                args.cpu_ranks_per_node = 128
            case _:
                args.cpu_ranks_per_node = 1
    args.cpu_ranks_per_node = str(args.cpu_ranks_per_node)
    if args.system not in args.machine_identifier:
        MI_System_Mismatch = f"Indicated system ({args.system}) name not found in machine identifier ({args.machine_identifier}). This could result in improper #cpu_ranks_per_node and lead to over- or under-subscription of resources!"
        warnings.warn(MI_System_Mismatch)
        if not args.error_demotion:
            raise ValueError(WarningsAsErrors)
    args.gpu_enabled = str(args.gpu_enabled)

    # Gaussian Copula arguments
    if args.gc_sys is None:
        args.gc_sys = args.mpi_ranks
    # First, FFT based on app-scale unless explicitly specified
    if args.gc_app is None:
        args.gc_app = args.application_scale
    # Then each dimension based on FFT default unless explicitly specified
    if args.gc_app_x is None:
        args.gc_app_x = args.gc_app
    if args.gc_app_y is None:
        args.gc_app_y = args.gc_app
    if args.gc_app_z is None:
        args.gc_app_z = args.gc_app
    if type(args.gc_input) is str:
        args.gc_input = [args.gc_input]
    # Argparse is bad and doesn't put nargs>=1 into lists by default
    if type(args.gc_ignore) is str:
        args.gc_ignore = [args.gc_ignore]
    # Join to string because this is a command line argument and can't obey normal python __str__ semantics for lists
    if args.gc_ignore is not None:
        args.gc_ignore = " ".join(args.gc_ignore)
    # Provide runtime args to tempaltes as needed / defined
    args.bonus_runtime_args = ""
    if args.resume is not None:
        args.bonus_runtime_args += f" --resume {' '.join(args.resume)}"
    if 'gc' in args.ens_template_export.stem:
        try:
            args.bonus_runtime_args += f" --constraint-sys {args.gc_sys} --constraint-app-x {args.gc_app_x} --constraint-app-y {args.gc_app_y} --constraint-app-z {args.gc_app_z} --input {' '.join(args.gc_input)} --auto-budget={args.gc_auto_budget}"
            # These can be set for any GC run
            gc_args = ['gc_ignore', 'gc_predictions_only', 'gc_initial_quantile', ]
            target_args = ['--ignore', '--predictions-only', '--initial-quantile', ]
            for argname, bonusargname in zip(gc_args, target_args):
                local_arg = getattr(args, argname)
                if local_arg is not None:
                    args.bonus_runtime_args += f" {bonusargname} {local_arg}"
            # These are only set when auto-budgeting is enabled
            if args.gc_auto_budget:
                autobudget_args = ['gc_min_quantile', 'gc_budget_confidence', 'gc_quantile_reduction',
                                   'gc_ideal_proportion', 'gc_ideal_attrition', 'gc_determine_budget_only', ]
                target_autobudget = ['--min-quantile', '--budget-confidence', '--quantile-reduction',
                                     '--ideal-proportion', '--ideal-attrition', '--determine-budget-only', ]
                for argname, bonusargname in zip(autobudget_args, target_autobudget):
                    local_arg = getattr(args, argname)
                    if local_arg is not None:
                        args.bonus_runtime_args += f" {bonusargname} {local_arg}"
        except TypeError: # args.gc_input is Nonetype, cannot be iterated
            raise ValueError(f"Must supply --gc-input arguments to run Gaussian Copula")

    # Set up `sed` arguments
    # Key: Tuple of args attribute names (in order) to use to fill template string values
    # Value: List of substitutions using these subsitutions
    #           Substitutions defined as tuple of filename and the template string itself
    args.seds = {
        ('ens_dir_path',): [(args.ens_template_export, "s/^ENSEMBLE_DIR_PATH = .*/ENSEMBLE_DIR_PATH = {}/"),],
        ('machine_identifier',): [(args.ens_template_export, "s/MACHINE_IDENTIFIER = .*/MACHINE_IDENTIFIER = {}/"),],
        ('seed_configspace',): [(args.ens_template_export, "s/CONFIGSPACE_SEED = .*/CONFIGSPACE_SEED = {}/"),],
        ('seed_ytopt',): [(args.ens_template_export, "s/YTOPT_SEED = .*/YTOPT_SEED = {}/"),],
        ('seed_numpy',): [(args.ens_template_export, "s/NUMPY_SEED = .*/NUMPY_SEED = {}/"),],
        ('mpi_ranks',): [(args.ens_template_export, "s/MPI_RANKS = [0-9]*/MPI_RANKS = {}/"),],
        ('worker_timeout',): [(args.ens_template_export, "s/'app_timeout': [0-9]*,/'app_timeout': {},/"),],
        ('application_scale',): [(args.ens_template_export, "s/APP_SCALE = [0-9A-Z_]*/APP_SCALE = {}/"),],
        ('application_x',): [(args.ens_template_export, "s/APP_SCALE_X = [0-9A-Z_]*/APP_SCALE_X = {}/"),],
        ('application_y',): [(args.ens_template_export, "s/APP_SCALE_Y = [0-9A-Z_]*/APP_SCALE_Y = {}/"),],
        ('application_z',): [(args.ens_template_export, "s/APP_SCALE_Z = [0-9A-Z_]*/APP_SCALE_Z = {}/"),],
        ('cpu_override',): [(args.ens_template_export, "s/cpu_override = .*/cpu_override = {}/"),],
        ('cpu_ranks_per_node',): [(args.ens_template_export, "s/cpu_ranks_per_node = .*/cpu_ranks_per_node = {}/"),],
        ('gpu_override',): [(args.ens_template_export, "s/gpu_override = .*/gpu_override = {}/"),],
        ('gpu_enabled',): [(args.ens_template_export, "s/gpu_enabled = .*/gpu_enabled = {}/"),],
    }
    return args

def sed_error(sed_command, sed_arg, target_file):
    print(sed_command)
    print(" ".join(sed_command))
    print(proc.stdout.decode('utf-8'))
    print(proc.stderr.decode('utf-8'))
    SedSubstitutionFailure = f"Substitution '{sed_arg}' in '{target_file}' failed"
    raise ValueError(SedSubstitutionFailure)

args, ensemble_operating_dir, proc = None, None, None
def cleanup(signum=None, frame=None):
    migrations = [(args.ens_script_export, ensemble_operating_dir.joinpath(args.ens_script_export), "Job script"),
                  (args.ens_template_export, ensemble_operating_dir.joinpath(args.ens_template_export), "LibEnsemble driver"),
                  ('ensemble.log', ensemble_operating_dir.joinpath("ensemble.log"), "Ensemble logs"),
                  ('ytopt.log', ensemble_operating_dir.joinpath("ytopt.log"), "Ytopt logs"),]
    curdir = pathlib.Path('.')
    stats = [_ for _ in curdir.glob('libE_stats.txt')]
    migrations += [(s.name, ensemble_operating_dir.joinpath(s.name), "LibE stats") for s in stats]
    # Aborted by signal
    if signum is not None:
        import time
        # The files may need a second to appear. This doesn't always work but we can't wait forever
        time.sleep(1)
        # Try to find the abort files and add them to migration list
        history = [_ for _ in curdir.glob('libE_history_at_abort_*.npy')]
        migrations += [(h.name, ensemble_operating_dir.joinpath(h.name), "LibE aborted history") for h in history]
        persis = [_ for _ in curdir.glob('libE_persis_info_at_abort_*.pickle')]
        migrations == [(p.name, ensemble_operating_dir.joinpath(p.name), "LibE aborted persis") for p in persis]
    for migration_from, migration_to, identifier in migrations:
        try:
            shutil.move(migration_from, migration_to)
        except:
            print(f"{identifier} could not be migrated <-- {migration_from}")
        else:
            print(f"{identifier} migrated --> {migration_to}")
    if proc.returncode == 0 and args.display_results:
        import pandas as pd
        try:
            finished = pd.read_csv(f"{ensemble_operating_dir}/manager_results.csv")
        except FileNotFoundError:
            print("Manager Results unavailable")
        else:
            print("Finished evaluations")
            print(finished)
        try:
            unfinished = pd.read_csv(f"{ensemble_operating_dir}/unfinished_results.csv")
        except FileNotFoundError:
            print("No unfinished evaluations to view")
        else:
            print("Unfinished evaluations")
            print(unfinished)
signal.signal(signal.SIGTERM, cleanup)
signal.signal(signal.SIGINT, cleanup)

if __name__ == '__main__':
    args = parse()
    # Copy files
    shuffle = [(args.ens_template, args.ens_template_export, 'LibEnsemble template'),
               (args.ens_script, args.ens_script_export, 'Job script'),]
    for (copy_from, copy_to, identifier) in shuffle:
        try:
            shutil.copy2(copy_from, copy_to)
        except:
            print(f"Failed to copy {identifier} <-- {copy_from}")
        else:
            print(f"Copy {identifier} --> {copy_to}")
    # Edit files
    for (sed_arg, substitution_specs) in args.seds.items():
        args_values = tuple([getattr(args, arg) for arg in sed_arg])
        for (target_file, template_string) in substitution_specs:
            print(f"Set {sed_arg} to {args_values} in {target_file}")
            sed_command = ['sed', '-i', template_string.format(*args_values), str(target_file)]
            proc = subprocess.run(sed_command, capture_output=True)
            if proc.returncode != 0:
                if any(["sed: -I or -i may not be used with stdin" in output.decode('utf-8') for output in [proc.stdout, proc.stderr]]):
                    sed_command.insert(2,"''")
                    proc = subprocess.run(sed_command, capture_output=True)
                    if proc.returncode != 0:
                        sed_error(sed_command, sed_arg, target_file)
                else:
                    sed_error(sed_command, sed_arg, target_file)
            else:
                print("  -- SED OK")

    # Environment
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
#PBS -l filesystems=home
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
export EXE="{args.ens_template_export}"

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
    print(f"Produce job script {args.ens_script_export}")
    with open(args.ens_script_export, 'w') as f:
        f.write(job_contents)
    # Set RWX for owner, RX for all others
    os.chmod(args.ens_script_export, stat.S_IRWXU |
                                     stat.S_IRGRP | stat.S_IXGRP |
                                     stat.S_IROTH | stat.S_IXOTH)
    print("Script written")
    ensemble_operating_dir = pathlib.Path(f"./ensemble_{args.ens_dir_path[1:-1]}")
    if args.launch_job:
        try:
            proc = subprocess.run(f"./{args.ens_script_export}")
        except KeyboardInterrupt:
            cleanup(signum=signal.SIGINT)
        else:
            cleanup()

