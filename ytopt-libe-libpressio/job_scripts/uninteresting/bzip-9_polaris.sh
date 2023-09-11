#!/bin/bash -x
#PBS -l walltime=03:00:00
#PBS -l select=10:system=polaris
#PBS -l filesystems=home:grand
#PBS -A LibPressioTomo
#PBS -q prod

source /home/trandall/polaris_libpressio_env.sh;
cd /home/trandall/ytune_23/tlranda-ytopt-libensemble/ytopt-libe-libpressio;

jsons=( 'bzip-9' ); # 'sz3'
mpi_ranks=( 2 4 6 8 12 16 24 32 );
#mpi_ranks=( 4 8 16 32 64 256 492 );
#  nodes =   1 2  4  8 16  64 123
workers=( 4 );
# MAX nodes = 10 (2% cluster capacity)
# POLARIS has MAX 496 on PROD queue
calls=0;
for n_ranks in ${mpi_ranks[@]}; do
for json in ${jsons[@]}; do
for n_workers in ${workers[@]}; do
    echo "Calling on ${n_workers} workers with ${n_ranks} mpi ranks per worker for ${json}";
    call="python libEwrapper.py --mpi-ranks ${n_ranks} --worker-timeout 300 --gpu-enabled --ensemble-workers ${n_workers} --max-evals 200 --machine-identifier polaris-gpu --system polaris --ens-dir-path Polaris_${n_ranks}r_${json} --launch-job --display-results --json ${json}";
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

