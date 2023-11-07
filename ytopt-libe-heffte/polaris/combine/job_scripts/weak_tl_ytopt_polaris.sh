#!/bin/bash
#PBS -l walltime=02:00:00
#PBS -l select=65:system=polaris
#PBS -l filesystems=home
#PBS -A EE-ECP
#PBS -q prod

source /home/trandall/polaris_heffte_env.sh;
cd /home/trandall/ytune_23/tlranda-ytopt-libensemble/ytopt-libe-heffte/polaris/combine;

# App and MPI should scale together as weak-scalars
#app_scales=( 64 128 256  512 1024 1400 );
#mpi_ranks=(   8  16  32   64  128  256 );
#n_nodes=(     2   4   8   16   32   64 );
#n_workers=(   4   4   4    4    2    1 );
app_scales=( 1024 1400 );
mpi_ranks=(   128  256 );
n_nodes=(      32   64 );
n_workers=(     2    1 );
## MAX nodes = 4 * 123 + 1 = 493 (88% cluster capacity)
# MAX nodes = 4 * 64 + 1 = 265 (53% cluster capacity)
# POLARIS has MAX 496 on PROD queue
calls=0;

for rank_idx in ${!mpi_ranks[@]}; do
    n_ranks=${mpi_ranks[$rank_idx]};
    n_nodes=${n_nodes[$rank_idx]};
    app_scale=${app_scales[$rank_idx]};
    workers=${n_workers[$rank_idx]};
    echo "Calling on ${workers} workers with ${n_ranks} mpi ranks (${n_nodes} nodes) per worker for size ${app_scale}";
    call="python libEwrapper.py --mpi-ranks ${n_ranks} --worker-timeout 300 --application-scale ${app_scale} --gpu-enabled --ensemble-workers ${workers} --max-evals 200 --configure-environment craympi --machine-identifier polaris-gpu --system polaris --ens-dir-path Polaris_YTOPT_NO_TOP_${n_nodes}n_${app_scale}a --ens-template run_ytopt.py --ens-script qsub_tl.batch --launch-job --display-results";
    date;
    echo "${call}";
    eval "${call}";
    date;
    calls=$(( ${calls} + 1 ));
done;
echo;
echo "Requested ${calls} calls";
date;

