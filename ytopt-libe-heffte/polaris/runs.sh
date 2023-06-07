#!/bin/bash

# set the number of nodes
let nnds=1
# set the total number of MPI ranks
let nranks=2
# set the number of workers (number of nodes/nranks plus 1)
let nws=2
# set the maximum application runtime(s) as timeout baseline for each evaluation
let appto=100

let nt=$((nranks * 1))

# set the maximum number of evaluations
let evals=4

#--- process processexe.pl to change the number of nodes (no change)
./processcp.pl ${nranks}

# set the MPI ranks partition
./processry.pl ${nt}

# set application timeout
echo "Set app timeout"
sed -i "s/app_timeout = [0-9]*/app_timeout = ${appto}/" plopper.py
grep "app_timeout = " plopper.py
#./plopper.pl plopper.py ${appto}

# find the conda path
cdpath=$(conda info | grep -i 'base environment')
arr=(`echo ${cdpath}`)
cpath="$(echo ${arr[3]})/etc/profile.d/conda.sh"

#-----This part creates a submission script---------
#cat >batch.job <<EOF
#!/bin/bash -x
#PBS -l walltime=00:60:00
#PBS -l select=${nnds}:system=polaris
#PBS -l filesystems=home:grand:eagle
#PBS -A EE-ECP
#PBS -q debug-scaling

# Script to run libEnsemble using multiprocessing on launch nodes.
# Assumes Conda environment is set up.

# To be run with central job management
# - Manager and workers run on launch node.
# - Workers submit tasks to the nodes in the job available (using exe.pl)

# Name of calling script
export EXE=run_ytopt.py

# Communication Method
export COMMS="--comms local"

# Number of workers. For multiple nodes per worker, have nworkers be a divisor of nnodes, then add 1
# e.g. for 2 nodes per worker, set nnodes = 12, nworkers = 7
export NWORKERS="--nworkers ${nws}"  # extra worker running generator (no resources needed)
# Adjust exe.pl so workers correctly use their resources

export EVALS="--max-evals ${evals}"

# Name of Conda environment
export CONDA_ENV_NAME=ytune_23

export PMI_NO_FORK=1 # Required for python kills on Theta

# Load Polaris modules
module load conda cudatoolkit-standalone/11.8.0

# Activate conda environment
export PYTHONNOUSERSITE=1
conda activate \$CONDA_ENV_NAME

# Launch libE
pycommand="python $EXE $COMMS $NWORKERS --learner=RF $EVALS" # > out.txt 2>&1"
echo "$pycommand";
eval "$pycommand";
#EOF
#-----This part submits the script you just created--------------
#chmod +x batch.job
#qsub batch.job

