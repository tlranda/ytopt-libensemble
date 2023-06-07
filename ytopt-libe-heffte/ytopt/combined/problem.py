import numpy as np
from numpy import abs, cos, exp, mean, pi, prod, sin, sqrt, sum
from autotune import TuningProblem
from autotune.space import *
import os
import sys
import time
import json
import math

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from ConfigSpace import ConfigurationSpace, EqualsCondition
from skopt.space import Real, Integer, Categorical
import itertools

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, os.path.dirname(HERE))
from plopper import Plopper

cs = CS.ConfigurationSpace(seed=1234)
# arg1  precision
p0 = CSH.CategoricalHyperparameter(name='p0', choices=["double", "float"], default_value="float")
# arg2  3D array dimension size
p1 = CSH.OrdinalHyperparameter(name='p1', sequence=[64,128,256,512,1024], default_value=128)
# arg3  reorder
p2 = CSH.CategoricalHyperparameter(name='p2', choices=["-no-reorder", "-reorder"," "], default_value=" ")
# arg4 alltoall
p3 = CSH.CategoricalHyperparameter(name='p3', choices=["-a2a", "-a2av", " "], default_value=" ")
# arg5 p2p
p4 = CSH.CategoricalHyperparameter(name='p4', choices=["-p2p", "-p2p_pl"," "], default_value=" ")
# arg6 reshape logic
p5 = CSH.CategoricalHyperparameter(name='p5', choices=["-pencils", "-slabs"," "], default_value=" ")
# arg7
p6 = CSH.CategoricalHyperparameter(name='p6', choices=["-r2c_dir 0", "-r2c_dir 1","-r2c_dir 2", " "], default_value=" ")
# arg8
p7 = CSH.UniformFloatHyperparameter(name='p7', lower=0, upper=1)
#p7 = CSH.CategoricalHyperparameter(name='p7', choices=["-ingrid 4 1 1", "-ingrid 2 2 1", "-ingrid 2 1 2","-ingrid 1 2 2", " "], default_value=" ")
# arg9
p8 = CSH.UniformFloatHyperparameter(name='p8', lower=0, upper=1)
#p8 = CSH.CategoricalHyperparameter(name='p8', choices=["-outgrid 4 1 1", "-outgrid 2 2 1", "-outgrid 2 1 2","-outgrid 1 2 2"," "], default_value=" ")
#number of threads
p9= CSH.UniformIntegerHyperparameter(name='p9', lower=2, upper=8, default_value=8, q=2)

cs.add_hyperparameters([p0, p1, p2, p3, p4, p5, p6, p7, p8, p9])

# problem space
task_space = None

input_space = cs

output_space = Space([
     Real(0.0, inf, name="time")
])

dir_path = os.path.dirname(os.path.realpath(__file__))
kernel_idx = dir_path.rfind('/')
kernel = dir_path[kernel_idx+1:]
obj = Plopper(dir_path+'/speed3d.sh',dir_path)

x1=['p0','p1','p2','p3','p4','p5','p6','p7','p8','p9']

candidate_orders = [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]
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
    return topology
def topology_interpret(config: dict) -> dict:
    budget = config.pop('nodes') # Raises KeyError if not present
    if budget not in topology_cache.keys():
        topology_cache[budget] = make_topology(budget)
    topology = topology_cache[budget]+[' ']
    # Replace each key with uniform bucketized value
    for topology_key in topology_keymap.keys():
        selection = min(int(config[topology_key] * len(topology)), len(topology)-1)
        selected_topology = topology[selection]
        if type(selected_topology) is not str:
            selected_topology = f"{topology_keymap[topology_key]} {' '.join([str(_) for _ in selected_topology])}"
        config[topology_key] = selected_topology
    return config

def myobj(point: dict):
    point = topology_interpret(point)
    x = np.asarray_chkfinite([point[f'p{i}'] for i in range(len(point))]) # ValueError if any NaN or Inf
    value = list(point.values())
    params = list([_.upper() for _ in point.keys()])
    print('CONFIG:',point)
    results = obj.findRuntime(value, params)
    print('OUTPUT: ',results)
    return results

Problem = TuningProblem(
    task_space=None,
    input_space=input_space,
    output_space=output_space,
    objective=myobj,
    constraints=None,
    model=None
    )
Problem.request_machine_identifier = 'polaris-gpu'
Problem.request_passthrough_nodes = 2

