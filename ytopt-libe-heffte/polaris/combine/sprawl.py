import subprocess

ranks = [64,128,256,512,1024,2048,4096,8192]
nodes = [ 1,  2,  4,  8,  16,  32,  64, 128]
apps = [64,128,256,512,1024]

template = "python3 libEwrapper.py --mpi-ranks {RANKS} --application-scale {APP} --system theta --cpu-override 256 --cpu-ranks-per-node 64 --ensemble-workers 1 --max-evals 30 --comms local --configure-environment craympi --machine-identifier theta-knl --ensemble-dir-path gctla --ensemble-path-randomization --launch-job --gc-input {INPUT} --libensemble-target run_gctla.py --libensemble-export libE_runner.py --gc-auto-budget --gc-determine-budget-only --gc-ideal-proportion 0.06"

print(len(ranks), ranks)
print(len(nodes), nodes)
print(len(apps), apps)

for app in apps:
    for current_rank, leave_out_nodes in zip(ranks, nodes):
        logs = " ".join([f"logs/ThetaSourceTasks/Theta_{node}n_{app}a/manager_results.csv" for node in sorted(set(nodes).difference(set([leave_out_nodes])))])
        filled_template = template.format(RANKS=current_rank, APP=app, INPUT=logs)
        subprocess.call(filled_template, shell=True)

"logs/ThetaSourceTasks/Theta_1n_${APP}a/manager_results.csv logs/ThetaSourceTasks/Theta_2n_${APP}a/manager_results.csv logs/ThetaSourceTasks/Theta_4n_${APP}a/manager_results.csv logs/ThetaSourceTasks/Theta_8n_${APP}a/manager_results.csv logs/ThetaSourceTasks/Theta_16n_${APP}a/manager_results.csv logs/ThetaSourceTasks/Theta_32n_${APP}a/manager_results.csv logs/ThetaSourceTasks/Theta_64n_${APP}a/manager_results.csv logs/ThetaSourceTasks/Theta_128n_${APP}a/manager_results.csv"
