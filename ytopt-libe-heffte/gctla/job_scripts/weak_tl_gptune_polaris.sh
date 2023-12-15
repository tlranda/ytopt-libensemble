#!/bin/bash
#PBS -l walltime=01:00:00
#PBS -l select=65:system=polaris
#PBS -l filesystems=home
#PBS -A EE-ECP
#PBS -q prod

source /home/trandall/polaris_knight_env.sh;
cd /home/trandall/ytune_23/tlranda-ytopt-libensemble/ytopt-libe-heffte/gctla;

# App and System scales
app3d_scales=( "0256_0256_0256" "0256_0256_0512" "0256_0512_0512" "0512_0512_0512" "0512_0512_1024" "0512_1024_1024" );
mpi_ranks=( 8 16 32 64 128 256 );
n_nodes=(   2  4  8 16  32  64 );
# MAX nodes = 65 (13% cluster capacity)
# Polaris has MAX 496 nodes
calls=0;

# Form the dataset for inputs
dataset_basis="logs/PolarisSourceTasks/Polaris_";
weak_dataset=( );
for rank_index in ${!app3d_scales[@]}; do
    str_app=${app3d_scales[$rank_index]};
    str_nodes=${n_nodes[$rank_index]};
    while [[ ${#str_nodes} -lt 3 ]]; do
        str_nodes="0${str_nodes}";
    done;
    weak_dataset+=( "${dataset_basis}${str_nodes}n_${str_app}a/manager_results.csv" );
done;

# Sanity check for requested files
dataset_exists=1;
for w in ${weak_dataset[@]}; do
    if [[ ! -f "${w}" ]]; then
        dataset_exists=0;
        echo "MISSING DATASET FILE: ${w}";
    fi
done;
if [[ ${dataset_exists} -ne 1 ]]; then
    echo "ABORT";
    exit;
fi

# Form calls
for rank_idx in ${!weak_dataset[@]}; do
    left_out=${weak_dataset[$rank_idx]};
    n_ranks=${mpi_ranks[$rank_idx]};
    n_nodes=${n_nodes[$rank_idx]};
    app_scale=${app3d_scales[$rank_idx]};
    oIFS="$IFS";
    IFS=_;
    app_xyz=( ${app_scale} );
    IFS="$oIFS";
    echo "Calling on ${n_workers} workers with ${n_ranks} mpi ranks (${n_nodes} nodes) per worker for size ${app_scale}";
    call="python3 wrapper_components/gptune_heffte.py --sys ${n_ranks} --app-x ${app_xyz[0]} --app-y ${app_xyz[1]} --app-z ${app_xyz[2]} --gpu-enabled --log GPTune_${n_nodes}n_${app_scale}a.csv --inputs ${weak_dataset[@]} --ignore ${left_out} --preserve-history";
    date;
    echo "${call}";
    eval "${call}";
    date;
    calls=$(( ${calls} + 1 ));
done;
echo;
echo "Requested ${calls} calls";
date;

