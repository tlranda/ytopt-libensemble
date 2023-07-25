#!/bin/bash
#COBALT -t 01:00:00
#COBALT -n 129
#COBALT --attrs filesystems=home,grand,eagle
#COBALT -A EE-ECP
#COBALT -q default

source /home/trandall/theta_knl_heffte_env.sh;
cd /home/trandall/ytune_23/tlranda-ytopt-libensemble/ytopt-libe-heffte/polaris/combine;

app_scales=( 64 128 256 512 1024 1400 );
mpi_ranks=( 64 128 256 512 1024 4096 8192 );
# nodes =    1   2   4   8   16   64  128
workers=( 1 );
# MAX nodes = 128 + 1 = 129 (3% cluster capacity)
# THETA has MAX 4392 on DEFAULT queue
calls=0;
mkdir ThetaDefault
for app_scale in ${app_scales[@]}; do
for n_ranks in ${mpi_ranks[@]}; do
for n_workers in ${workers[@]}; do
    echo "Calling on ${n_workers} workers with ${n_ranks} mpi ranks per worker for size ${app_scale}";
    call="{ time aprun -n ${n_ranks} -N 64 -cc depth ./default_fftw.sh ${app_scale} > ThetaDefault/Theta_${n_ranks}r_${app_scale}a_default.log 2>&1 ; } 2> ThetaDefault/Theta_${n_ranks}r_${app_scale}a_default_time.txt"
    date;
    echo "${call}";
    eval "${call}";
    date;
    # Record in default dataset
    python - ThetaDefault/defaults.csv ThetaDefault/Theta_${n_ranks}r_${app_scale}a_default.log ThetaDefault/Theta_${n_ranks}r_${app_scale}a_default_time.txt <<EOF
import pandas as pd, pathlib, sys

columns=[f'p{_}' for _ in range(10)]+['c0','FLOPS','elapsed_sec','machine_identifier','mpi_ranks','threads_per_node','ranks_per_node','gpu_enabled','libE_id','libE_workers']
row=['float',${app_scale},]+[' ']*5+[1.0,1.0,0.0,'fftw',]
sysinfo=['theta-knl',${n_ranks},256,64,False,2,1]

csv_file, flop_log, time_log = tuple(sys.argv[1:])
print("Read logs from:", flop_log)
with open(flop_log, 'r') as f:
    lines = [_.rstrip() for _ in f.readlines()]
    for line in lines:
        if "Performance: " in line:
            split = [_ for _ in line.split(' ') if len(_) > 0]
            flops = -1 * float(split[1])
            break
print(flops, "GFLOP/s")
print("Read elapsed time from:", time_log)
with open(time_log, 'r') as f:
    lines = [_.rstrip() for _ in f.readlines()]
    for line in lines:
        if 'real' in line:
            split = [_ for _ in line.split(' ') if len(_) > 0]
            try:
                focus = split[1]
            except:
                focus = split[0].split('\t')[1]
            m,s = focus[:-1].split('m',1)
            elapsed_sec = 60*int(m) + float(s)
print(elapsed_sec, "seconds")
csv = pathlib.Path(csv_file)
if csv.exists():
    data = [pd.read_csv(csv)]
    elapsed_sec += data[0]['elapsed_sec'].max()
else:
    data = []
new_data = pd.DataFrame([row+[flops,elapsed_sec]+sysinfo], columns=columns, index=[0])
data.append(new_data)
combined = pd.concat(data)
print("Write to:", csv_file)
print(combined)
combined.to_csv(csv_file, index=False)
EOF
    calls=$(( ${calls} + 1 ));
done;
echo;
done;
echo;
done;
echo "Requested ${calls} calls";


