#!/bin/bash
#This script is for running an app on a laptop using mpirun without any scheduler

# set the number of nodes for the MPI ranks per run (for a single node, number of MPI ranks)
let nranks=4
# set the maximum application runtime(s) as timeout baseline for each evaluation
let appto=300

#--- process processexe.pl to change the number of nodes (no change)
# set the MPI ranks per run
./processcp.pl ${nranks}
# Change exe.pl
./processexe.pl

# set the MPI ranks partition
./processry.pl ${nranks}

# set application timeout
echo "Set app timeout"
sed -i "s/app_timeout = [0-9]*/app_timeout = ${appto}/" plopper.py
grep "app_timeout = " plopper.py

# find the conda path
cdpath=$(conda info | grep -i 'base environment')
arr=(`echo ${cdpath}`)
cpath="$(echo ${arr[3]})/etc/profile.d/conda.sh"

#-----This part creates a submission script---------
cat >batch.job <<EOF
#!/bin/bash -x

# Launch ytopt
python -m ytopt.search.ambs --evaluator subprocess --problem problem.Problem --learner=RF --max-evals=4 #> out.txt 2>&1
EOF
#-----This part submits the script you just created--------------
chmod +x batch.job
./batch.job
