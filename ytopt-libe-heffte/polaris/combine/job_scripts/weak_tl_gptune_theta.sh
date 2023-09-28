#!/bin/bash 
#COBALT -t 06:00:00
#COBALT -n 256
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
n_workers=1; # GPTune doesn't have a way to do multi-evaluations
# MAX nodes = 128 + 1 = 129 (3% cluster capacity)
# THETA has MAX 4392 on DEFAULT queue
calls=0;

# Form the dataset to use as inputs
dataset_basis="logs/ThetaSourceTasks/Theta_"
weak_dataset=( );
for rank_index in ${!app_scales[@]}; do
    str_app=${app_scales[$rank_index]};
    while [[ ${#str_app} -lt 4 ]]; do
        str_app="0${str_app}";
    done;
    str_nodes=${n_nodes[$rank_index]};
    while [[ ${#str_nodes} -lt 3 ]]; do
        str_nodes="0${str_nodes}";
    done;
    weak_dataset+=( "${dataset_basis}${str_nodes}n_${str_app}a/manager_results.csv" );
done;

for rank_idx in ${!weak_dataset[@]}; do
    # Skip ones already collected, but need to keep values in the list to properly construct
    # the inputs
    if [[ ${rank_idx} -eq 0 ]]; then
        continue;
    fi
    left_out=${weak_dataset[$rank_idx]};
    n_ranks=${mpi_ranks[$rank_idx]};
    n_nodes=${n_nodes[$rank_idx]};
    app_scale=${app_scales[$rank_idx]};
    localized_dataset=("${weak_dataset[@]}")
    unset localized_dataset[$rank_idx]
    echo "Calling on ${n_workers} workers with ${n_ranks} mpi ranks (${n_nodes} nodes) per worker for size ${app_scale}";
    # Aggregate things safely here:
    store_dir=GPTune_Results/Theta_${n_nodes}n_${app_scale}a;
    mkdir -p ${store_dir};
    call="python gptune_heffte.py --sys ${n_ranks} --app ${app_scale} --max-evals 30 --preserve-history --inputs ${localized_dataset[@]} --log gptune_${n_nodes}n_${app_scale}a.csv";
    date;
    echo "${call}";
    eval "${call}";
    mv gptune_${n_nodes}n_${app_scale}a.csv ${store_dir};
    mv tmp_files ${store_dir};
    date;
    calls=$(( ${calls} + 1 ));
done;
echo;
echo "Requested ${calls} calls";
date;

