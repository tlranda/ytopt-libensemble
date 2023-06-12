"""
This module is a wrapper around an example ytopt objective function
"""
__all__ = ['init_obj', 'myobj']

import numpy as np
import os
import time
import itertools
from plopper import Plopper

start_time = time.time()

def init_obj(H, persis_info, sim_specs, libE_info):
    point = {}
    for field in sim_specs['in']:
        point[field] = np.squeeze(H[field])
    # Pass along machine info to point for topology preparation
    machine_info = sim_specs['user']['machine_info']
    point['machine_info'] = machine_info

    y = myobj(point, sim_specs['in'], libE_info['workerID'])  # ytopt objective wants a dict
    H_o = np.zeros(len(sim_specs['out']), dtype=sim_specs['out'])
    H_o['FLOPS'] = y
    H_o['elapsed_sec'] = time.time() - start_time
    # Wrap in list for ask-tell processing as a CSV
    # Passed back for processing in CSV records
    H_o['machine_identifier'] = [machine_info['identifier']]
    H_o['mpi_ranks'] = [machine_info['mpi_ranks']]
    H_o['ppn'] = [machine_info['ppn']]
    H_o['gpu_enabled'] = [machine_info['gpu_enabled']]
    H_o['libE_workers'] = [machine_info['libE_workers']]

    return H_o, persis_info

candidate_orders = [_ for _ in itertools.product([0,1,2], repeat=3) if len(_) == len(set(_))]
topology_keymap = {'p7': '-ingrid', 'p8': '-outgrid'}
topology_cache = {}
def make_topology(budget: int) -> list[tuple[int,int,int]]:
    # Powers of 2 that can be represented in topology X/Y/Z
    factors = [2 ** x for x in range(int(np.log2(budget)),-1,-1)]
    topology = []
    for candidate in itertools.product(factors, repeat=3):
        # All topologies need to have product that == budget
        # Reordering the topology is not considered a relevant difference, so reorderings are discarded
        if np.prod(candidate) != budget or \
           np.any([tuple([candidate[_] for _ in order]) in topology for order in candidate_orders]):
            continue
        topology.append(candidate)
    # Add the null space
    topology += [' ']
    return topology
def topology_interpret(config: dict) -> dict:
    machine_info = config.pop('machine_info') # Raises KeyError if not present
    budget = machine_info['mpi_ranks'] * machine_info['ppn']
    if budget not in topology_cache.keys():
        topology_cache[budget] = make_topology(budget)
    topology = topology_cache[budget]
    # Replace each key with uniform bucketized value
    for topology_key in topology_keymap.keys():
        selection = min(int(config[topology_key] * len(topology)), len(topology)-1)
        selected_topology = topology[selection]
        if type(selected_topology) is not str:
            selected_topology = f"{topology_keymap[topology_key]} {' '.join([str(_) for _ in selected_topology])}"
        config[topology_key] = selected_topology
    # Fix numpy zero-dimensional arrays
    for k,v in config.items():
        if k not in topology_keymap.keys() and type(v) is np.ndarray and v.shape == ():
            config[k] = v.tolist()
    return config

def myobj(point: dict, params: list, workerID: int) -> float:
    try:
        point = topology_interpret(point)
        print(f"Worker {workerID} receives point {point}")
        x = np.array([point[f'p{i}'] for i in range(len(point))])
        def plopper_func(x, params):
            # Should utilize machine identifier
            obj = Plopper('./speed3d.sh', './')
            x = np.asarray_chkfinite(x)
            value = [point[param] for param in params]
            os.environ["OMP_NUM_THREADS"] = str(value[9])
            params = [i.upper() for i in params]
            result = obj.findRuntime(value, params, workerID)
            return result

        results = plopper_func(x, params)
        # print('CONFIG and OUTPUT', [point, results], flush=True)
        print(f"Worker {workerID} returns point {results}")
        return results
    except Exception as e:
        bonus_context = f"point: {point} | params: {params} | workerID: {workerID} | "
        e.args = tuple([bonus_context+e.args[0]])
        raise e

