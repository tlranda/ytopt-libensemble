"""
This module is a wrapper around a GC_TLA objective function
"""
__all__ = ['init_obj']

import numpy as np
import time
import pathlib
import copy

start_time = time.time()

def init_obj(H, persis_info, sim_specs, libE_info):
    point = {}
    for field in sim_specs['in']:
        point[field] = np.squeeze(H[field])
    problem = copy.deepcopy(sim_specs['user']['problem'])
    nodefile = sim_specs['user']['nodefile_dict'][libE_info['workerID']]
    worker_output_dir = pathlib.Path('.').joinpath('tmp_files').resolve()
    timeout = sim_specs['user']['machine_info']['app_timeout']

    print(f"[libE simulator - {libE_info['workerID']}] submits point: {point}")
    y = problem.evaluateConfiguration(point, nodefile=nodefile, output_dir=worker_output_dir, timeout=timeout)
    print(f"[libE simulator - {libE_info['workerID']}] receives objective for point: {point} -> {y}")
    H_o = np.zeros(len(sim_specs['out']), dtype=sim_specs['out'])
    H_o['FLOPS'] = y
    H_o['elapsed_sec'] = time.time() - start_time
    # Wrap in list for ask-tell processing as a CSV
    # Passed back for processing in CSV records
    H_o['libE_id'] = [libE_info['workerID']]
    machine_info = sim_specs['user']['machine_info']
    H_o['machine_identifier'] = [machine_info['identifier']]
    H_o['mpi_ranks'] = [machine_info['mpi_ranks']]
    H_o['threads_per_node'] = [machine_info['threads_per_node']]
    H_o['ranks_per_node'] = [machine_info['ranks_per_node']]
    H_o['gpu_enabled'] = [machine_info['gpu_enabled']]
    H_o['libE_workers'] = [machine_info['libE_workers']]

    return H_o, persis_info

