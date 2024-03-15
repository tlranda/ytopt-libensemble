"""
This module is a wrapper around an example ytopt objective function
"""
__all__ = ['init_obj', 'myobj']

import numpy as np
import os
import time
import pathlib
import itertools
from . import plopper
Plopper = plopper.Plopper

start_time = time.time()

def init_obj(H, persis_info, sim_specs, libE_info):
    point = {}
    for field in sim_specs['in']:
        point[field] = np.squeeze(H[field])
    # Pass along machine info to point for topology preparation
    machine_info = sim_specs['user']['machine_info']
    point['machine_info'] = machine_info

    print(f"[libE simulator - {libE_info['workerID']}] submits point: {point}")
    y = myobj(point, sim_specs['in'], libE_info['workerID']) # ytopt objective wants a dict
    print(f"[libE simulator - {libE_info['workerID']}] receives objective for point: {point}")
    H_o = np.zeros(len(sim_specs['out']), dtype=sim_specs['out'])
    H_o['FLOPS'] = y
    H_o['elapsed_sec'] = time.time() - start_time
    # Wrap in list for ask-tell processing as a CSV
    # Passed back for processing in CSV records
    H_o['machine_identifier'] = [machine_info['identifier']]
    H_o['mpi_ranks'] = [machine_info['mpi_ranks']]
    H_o['threads_per_node'] = [machine_info['threads_per_node']]
    H_o['ranks_per_node'] = [machine_info['ranks_per_node']]
    H_o['gpu_enabled'] = [machine_info['gpu_enabled']]
    H_o['libE_id'] = [libE_info['workerID']]
    H_o['libE_workers'] = [machine_info['libE_workers']]

    return H_o, persis_info

def myobj(point: dict, params: list, workerID: int) -> float:
    try:
        machine_info = point.pop('machine_info')

        import os
        full_nodefile, worker_nodefile = None, None
        if 'nodelist' in machine_info:
            full_nodefile = machine_info['nodelist']
        elif 'PBS_NODEFILE' in os.environ:
            full_nodefile = os.environ['PBS_NODEFILE']

        # We've been given a set of nodes, ensure a subset is properly passed on to the worker
        if full_nodefile is not None:
            with open(full_nodefile, 'r') as f:
                avail_nodes = [_.rstrip() for _ in f.readlines()]
            worker_nodefile = pathlib.Path(f"worker_{workerID}_nodefile")
            if worker_nodefile.exists():
                with open(worker_nodefile,'r') as f:
                    worker_nodes = [_.rstrip() for _ in f.readlines()]
                for node in worker_nodes:
                    # Should be a tiny amount of busywork, but ensure that ValueError is raised
                    # if this node would send work to a node that ISN'T in our full nodefile's
                    # set of available nodes
                    found_index = avail_nodes.index(node)
            else:
                # Check if we have a bonus node provisioned for the generator or not
                if len(avail_nodes) > 1 and len(avail_nodes) % machine_info['libE_workers'] == 1:
                    # Generator gets its own node
                    avail_nodes = avail_nodes[1:]
                nodes_per_worker = len(avail_nodes) // machine_info['libE_workers']
                # LibEnsemble workerID's are allocated as follows:
                #   1 - Generator
                #   2 - Worker #0
                #   3 - Worker #1
                #   ...
                # So we have to adjust the workerID value to get a proper index
                worker_slice = slice((workerID-2)*nodes_per_worker, (workerID-1)*nodes_per_worker)
                worker_nodes = avail_nodes[worker_slice]
                with open(worker_nodefile, "w") as f:
                    f.write("\n".join(worker_nodes)+"\n")

        # Machine identifier changes the proper invocation to utilize allocated resources
        # Also customize timeout based on application scale per system
        if 'polaris' in machine_info['identifier']:
            depth_substr = "--depth {depth} --cpu-bind depth --env OMP_NUM_THREADS={depth} "
            machine_format_str = "mpiexec -n {mpi_ranks} --ppn {ranks_per_node} "
            if 'P8' in params:
                machine_format_str += depth_substr
            if worker_nodefile is not None:
                machine_format_str += f"-hostfile {worker_nodefile} "
            machine_format_str += "sh ./set_affinity_gpu_polaris.sh {interimfile}"
        elif 'theta' in machine_info['identifier']:
            machine_format_str = "aprun -n {mpi_ranks} -N {ranks_per_node} -cc depth -d {depth} -j {j} -e OMP_NUM_THREADS={depth} sh {interimfile}"
        else:
            machine_format_str = None

        # Set known timeouts to be more specific
        known_timeouts = {}
        if 'knl' in machine_info['identifier'] or 'cpu' in machine_info['identifier']:
            cpu_timeouts = {(64,64,64): 40.0,
                            (128,128,128): 80.0,
                            (256,256,256): 120.0,
                            (512,512,512): 300.0,
                            (1024,1024,1024): 300.0,
                           }
            known_timeouts.update(cpu_timeouts)
        elif 'gpu' in machine_info['identifier']:
            gpu_timeouts = {(64,64,64): 20.0,
                            (128,128,128): 20.0,
                            (256,256,256): 30.0,
                            (512,512,512): 40.0,
                            (1024,1024,1024): 60.0,
                           }
            known_timeouts.update(gpu_timeouts)
        xyz = (int(point['p1x']), int(point['p1y']), int(point['p1z']))
        if xyz in known_timeouts.keys():
            machine_info['app_timeout'] = known_timeouts[xyz]

        # Swap plopper templates / alter arguments when needed
        plopper_template = "./speed3d.sh"
        if max(xyz) >= 1024:
            # Prevent indexing overflow errors
            point['p0'] = str(point['p0'])+"-long"
            # Disable GPU aware MPI so we can run successfully
            # No need to check if on cpu--this argument shouldn't have an affect in that case
            plopper_template = "./speed3d_no_gpu_aware.sh"
        print(f"[worker {workerID} - obj] receives point {point}")
        x = np.asarray_chkfinite(point.values())
        obj = Plopper(plopper_template, './', machine_format_str)
        values = [point[param] for param in params]
        # Fix topology
        values[8] = f"-ingrid {values[8]}"
        values[9] = f"-outgrid {values[9]}"
        if 'p8' in params:
            os.environ["OMP_NUM_THREADS"] = str(point['p8'])
        params = [i.upper() for i in params]
        results = obj.findRuntime(values, params, workerID,
                                  machine_info['libE_workers'],
                                  machine_info['app_timeout'],
                                  machine_info['mpi_ranks'],
                                  machine_info['ranks_per_node'],
                                  1 # n_repeats
                                  )
        # print('CONFIG and OUTPUT', [point, results], flush=True)
        print(f"[worker {workerID} - obj] returns point {results}")
        return results
    except Exception as e:
        bonus_context = f"point: {point} | params: {params} | workerID: {workerID} | workingDirectory: {os.getcwd()} "
        e.args = tuple([bonus_context+str(e.args[0])])
        raise e

