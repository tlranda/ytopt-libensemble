#!/bin/bash
#This script is for running an app on a laptop using mpirun without any scheduler

# set the number of nodes for the MPI ranks per run (for a single node, number of MPI ranks)
let nranks=2
# set the maximum application runtime(s) as timeout baseline for each evaluation
let appto=300

# set the MPI ranks per run
echo "Set number of MPI nodes per run";
# Find the pattern "[[:digits:]]; # REPLACE" and edit the # digits according to nranks
sed -i "s/[0-9]*;\ #\ REPLACE/${nranks};\ #\ REPLACE/" exe.pl;
# Show what the exe.pl file reads at the replace point
grep ";\ #\ REPLACE" exe.pl | awk '{$1=$1};1' | sed "s/#\ REPLACE//";
sed -i "s/request_passthrough_nodes\ =\ [0-9]*/request_passthrough_nodes\ =\ ${nranks}/" problem.py;
grep "request_passthrough_nodes" problem.py | awk '{$1=$1};1';
echo;
# set application timeout
echo "Set app timeout";
# Find the pattern "app timeout = [[:digits:]]" and edit the # digits according to appto
sed -i "s/app_timeout = [0-9]*/app_timeout = ${appto}/" plopper.py;
# Show what the plopper.py file reads at the replace point
grep "app_timeout = " plopper.py | awk '{$1=$1};1';
echo;

# find the conda path
#cdpath=$(conda info | grep -i 'base environment')
#arr=(`echo ${cdpath}`)
#cpath="$(echo ${arr[3]})/etc/profile.d/conda.sh"

runstr="python -m ytopt.search.ambs --evaluator subprocess --problem problem.Problem --learner=RF --max-evals=4" #> out.txt 2>&1

#-----This part creates a submission script---------
#cat >batch.job <<EOF
#!/bin/bash -x

# Launch ytopt
#${runstr}
#EOF
#-----This part submits the script you just created--------------
#chmod +x batch.job
#./batch.job
echo $runstr;
eval $runstr;
