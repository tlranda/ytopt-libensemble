import subprocess
import argparse
import os, pathlib, stat, shutil
import secrets
import warnings

def build():
    prs = argparse.ArgumentParser()
    prs.add_argument("--error-demotion", action='store_true',
                     help="Demote some wrapper errors to warnings, may permit some unwanted behaviors (default: Error and halt)")
    # Scaling
    # Ensemble
    ensemble = prs.add_argument_group("LibEnsemble", "Arguments to control libEnsemble behavior")
    ensemble.add_argument("--ensemble-workers", type=int, default=1,
                          help="Number of libEnsemble workers (EXCLUDING manager; default %(default)s)")
    ensemble.add_argument("--max-evals", type=int, default=4,
                          help="Maximum evaluations collected by ensemble (default: %(default)s)")
    ensemble.add_argument("--comms", choices=['mpi','local', default='local',
                          help="Which communicator is used within ensemble (default: %(default)s)")
    ensemble.add_argument("--configure-environment", choices=["none", "polaris", "craympi"], nargs="*",
                          help="Set up environment based on known patterns (default: No setup)")
    ensemble.add_argument("--machine-identifier", default=None,
                          help="Used to name the executing machine in records (default: detect suitable name)")
    # Files
    files = prs.add_argument("Files", "Arguments to control file management")
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
    # Seeds
    seeds = prs.add_argument_group("Seeds", "Arguments to set various random seeds")
    seeds.add_argument("--seed-configspace", type=int, default=1234,
                       help="Seed for ConfigurationSpace object (default: %(default)s)")
    seeds.add_argument("--seed-ytopt", type=int, default=2345,
                       help="Seed for Ytopt library (default: %(default)s)")
    seeds.add_argument("--seed-numpy", type=int, default=1,
                       help="Seed for Numpy library (default: %(default)s)")
    # Job management
    job = prs.add_argument_group("Job", "Arguments to control job behavior")
    job.add_argument("--launch-job", action='store_true',
                     help="Execute job script after writing it (default: prepare script only)")
    return prs

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
        args.ens_template_export = template.parent.joinpath(template.stem + '_' + secrets.token_hex(nbytes=4))
        script = pathlib.Path(args.ens_script_export)
        args.ens_script_export = script.parent.joinpath(script.stem + '_' + secrets.token_hex(nbytes=4))

    # Environments
    if args.configure_environment is None:
        args.configure_environment = []

    # Machine identifier
    if args.machine_identifer is None:
        import platform
        args.machine_identifier = f'"{platform.node()}"'
    else:
        args.machine_identifier = f'"{args.machine_identifier}"'

    # Set up `sed` arguments
    # Key: Tuple of args attribute names (in order) to use to fill template string values
    # Value: List of substitutions using these subsitutions
    #           Substitutions defined as tuple of filename and the template string itself
    args.seds = {
        ('ens_dir_path',): [(args.ens_template, "s/^ENSEMBLE_DIR_PATH = .*/ENSEMBLE_DIR_PATH = {}"),],
        ('machine_identifier',): [(args.ens_template, "s/MACHINE_IDENTIFIER = .*/MACHINE_IDENTIIFER = {}"),],
        ('seed_configspace',): [(args.ens_template, "s/CONFIGSPACE_SEED = .*/CONFIGSPACE_SEED = {}"),],
        ('seed_ytopt',): [(args.ens_template, "s/YTOPT_SEED = .*/YTOPT_SEED = {}"),],
        ('seed_numpy',): [(args.ens_template, "s/NUMPY_SEED = .*/NUMPY_SEED = {}"),],
    }

    return args

def main(args=None):
    args = parse(args)
    # Copy files
    shutil.copy2(args.ens_template, args.ens_template_export)
    shutil.copy2(args.ens_script, args.ens_script_export)
    # Edit files
    for (sed_arg, substitution_specs) in args.seds.items():
        args_values = tuple([getattr(args, arg) for arg in sed_arg])
        for (target_file, template_string) in substitution_specs:
            print(f"Set {sed_arg} to {args_values} in {target_file}")
            sed_command = ['sed', '-i', template_string.format(*args_values), target_file]
            proc = subprocess.run(sed_command, capture_output=True)
            if proc.returncode != 0:
                print(proc.stdout.decode('utf-8'))
                print(proc.stderr.decode('utf-8'))
                SedSubstitutionFailure = f"Substitution '{sed_arg}' in '{target_file}' failed"
                raise ValueError(SedSubstitutionFailure)
            else:
                print("  -- SED OK")

    # Environment
    known_envs = {
    "none": "#pass",
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

source /lus/grand/projects/LibPressioTomo/setup.sh
spack env activate /lus/grand/projects/LibPressioTomo/roibin
""",
    "craympi": """
export IBV_FORK_SAFE=1; # May fix some MPI issues where processes call fork()
"""
    }
    env_adjust = "\n".join([known_envs[env] for env in args.configure_environment])

    # Write script
    job_contents = f"""#!/bin/bash -x
#PBS -l walltime=01:00:00
#PBS -l select={1+(args.ensemble_workers)}:system=polaris
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
    print(f"Produce script: {args.ens_script_export}")
    with open(args.ens_script_export, 'w') as f:
        f.write(job_contents)
    os.chmod(args.ens_script_export, stat.S_IRWXU |
                                     stat.S_IRGRP | stat.S_IXGRP |
                                     stat.S_IROTH | stat.S_IXOTH)
    print("Script written")

    ens_operating_dir = pathlib.Path(f"./ensemble_{args.ens_dir_path[1:-1]}")
    if args.launch_job:
        proc = subprocess.run(f"./{args.ens_script_export}")
        migrations = [(args.ens_script_export, ens_operating_dir.joinpath(args.ens_script_export), "Job script"),
                      (args.ens_template_export, ens_operating_dir.joinpath(args.ens_template_export), "LibEnsemble driver"),
                      ('ensemble.log', ens_operating_dir.joinpath("ensemble.log"), "Ensemble logs"),
                      ('ytopt.log', ens_operating_dir.joinpath("ytopt.log"), "Ytopt logs"),]
        for migration_from, migration_to, name in migrations:
            try:
                shutil.move(migration_from, migration_to)
            except:
                print(f"{name} could not be migrated <-- {migration_from}")
            else:
                print(f"{name} migrated --> {migration_to}")

