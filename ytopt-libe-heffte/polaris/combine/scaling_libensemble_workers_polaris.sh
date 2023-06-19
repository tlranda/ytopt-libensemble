#!/bin/bash -x

#PBS -l walltime=03:00:00
#PBS -l select=10:system=polaris
#PBS -l filesystems=home:grand:eagle
#PBS -A EE-ECP
#PBS -q prod

source /home/trandall/polaris_gpu_env.sh;
cd /home/trandall/ytune_23/tlranda-ytopt-libensemble/ytopt-libe-heffte/polaris/combine;

app_scales=( 64 128 256 512 1024 );
mpi_ranks=( 1 4 8 );
workers=( 1 2 4 );
calls=0;
for app_scale in ${app_scales[@]}; do
for n_ranks in ${mpi_ranks[@]}; do
for n_workers in ${workers[@]}; do
    echo "Calling on ${n_workers} workers with ${n_ranks} mpi ranks per worker for size ${app_scale}";
    call="python libEwrapper.py --mpi-ranks ${n_ranks} --worker-timeout 300 --application-scale ${app_scale} --gpu-enabled --ensemble-workers ${n_workers} --max-evals 200 --configure-environment craympi --machine-identifier polaris-gpu --ensemble-dir-path ScalingPolaris_${n_workers}libE_${n_ranks}mpi_${app_scale}app --ensemble-path-randomization --launch-job --display-results";
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

