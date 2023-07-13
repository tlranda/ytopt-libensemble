#!/bin/bash -x
#PBS -l walltime=06:00:00
#PBS -l select=65:system=polaris
#PBS -l filesystems=home:grand:eagle
#PBS -A EE-ECP
#PBS -q prod

source /home/trandall/polaris_cpu_heffte_env.sh;
cd /home/trandall/ytune_23/tlranda-ytopt-libensemble/ytopt-libe-heffte/polaris/combine;

app_scales=( 512 );
mpi_ranks=( 64 128 256 512 1024 );
#mpi_ranks=( 64 128 256 512 1024 4096 7872 );
#  nodes =    1   2   4   8   16   64  123
workers=( 4 );
## MAX nodes = 4 * 123 + 1 = 493 (88% cluster capacity)
# MAX nodes = 4 * 16 + 1 = 65 (12% cluster capacity)
# POLARIS has MAX 496 on PROD queue
calls=0;
for app_scale in ${app_scales[@]}; do
for n_ranks in ${mpi_ranks[@]}; do
for n_workers in ${workers[@]}; do
    echo "Calling on ${n_workers} workers with ${n_ranks} mpi ranks per worker for size ${app_scale}";
    call="python libEwrapper.py --mpi-ranks ${n_ranks} --worker-timeout 300 --application-scale ${app_scale} --ensemble-workers ${n_workers} --max-evals 200 --configure-environment craympi --machine-identifier cpu-polaris --ensemble-dir-path CPU_Polaris_${n_ranks}r_${app_scale}a --ensemble-path-randomization --libensemble-export libE_Run_.py --libensemble-randomization --launch-job --display-results";
    date;
    echo "${call}";
    eval "${call}";
    date;
    calls=$(( ${calls} + 1 ));
done;
echo;
done;
echo;
done;
echo "Requested ${calls} calls";

