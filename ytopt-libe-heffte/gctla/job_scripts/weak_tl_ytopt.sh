#!/bin/bash 
#COBALT -t 06:00:00
#COBALT -n 513
#COBALT --attrs filesystems=home
#COBALT -A EE-ECP
#COBALT -q default

source /home/trandall/theta_knl_heffte_env.sh;
cd /home/trandall/ytune_23/tlranda-ytopt-libensemble/ytopt-libe-heffte/polaris/combine;

# App and MPI should scale together as weak-scalars
app_scales=( 64 128 256  512 1024 1400 1400 );
mpi_ranks=( 128 256 512 1024 2048 4096 8192 );
n_nodes=(     2   4   8   16   32   64  128 );
# nodes =     2   4   8   16   32   64  128
n_workers=4;
# MAX nodes = 128 * 4 + 1 = 513 (12% cluster capacity)
# THETA has MAX 4392 on DEFAULT queue
calls=0;

for rank_idx in ${!mpi_ranks[@]}; do
    n_ranks=${mpi_ranks[$rank_idx]};
    n_nodes=${n_nodes[$rank_idx]};
    app_scale=${app_scales[$rank_idx]};
    echo "Calling on ${n_workers} workers with ${n_ranks} mpi ranks (${n_nodes} nodes) per worker for size ${app_scale}";
    call="python libEwrapper.py --mpi-ranks ${n_ranks} --worker-timeout 300 --application-scale ${app_scale} --cpu-override 256 --cpu-ranks-per-node 64 --ensemble-workers ${n_workers} --max-evals 200 --configure-environment craympi --machine-identifier theta-knl --system theta --ens-dir-path Theta_YTOPT_NO_TOP_${n_nodes}n_${app_scale}a --ens-template run_ytopt.py --ens-script qsub_tl.batch --launch-job --display-results";
    date;
    echo "${call}";
    eval "${call}";
    date;
    calls=$(( ${calls} + 1 ));
done;
echo;
echo "Requested ${calls} calls";
date;

