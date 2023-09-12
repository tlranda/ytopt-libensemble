from GC_TLA.base_problem import libe_problem_builder
from GC_TLA.base_plopper import LibE_Plopper
import os, subprocess, numpy as np
import itertools, math
from collections import UserDict
HERE = os.path.dirname(os.path.abspath(__file__))

input_space = [('Categorical',
                {'name': 'p0',
                 'choices': ['double', 'float'],
                 'default_value': 'float',
                }
               ),
               ('Ordinal',
                {'name': 'p1',
                 'sequence': ['64','128','256','512','1024'],
                 'default_value': '128',
                }
               ),
               ('Categorical',
                {'name': 'p2',
                 'choices': ['-no-reorder', '-reorder', ' '],
                 'default_value': ' ',
                }
               ),
               ('Categorical',
                {'name': 'p3',
                 'choices': ['-a2a', '-a2av', ' '],
                 'default_value': ' ',
                }
               ),
               ('Categorical',
                {'name': 'p4',
                 'choices': ['-p2p', '-p2p_pl', ' '],
                 'default_value': ' ',
                }
               ),
               ('Categorical',
                {'name': 'p5',
                 'choices': ['-pencils', '-slabs', ' '],
                 'default_value': ' ',
                }
               ),
               ('Categorical',
                {'name': 'p6',
                 'choices': ['-r2c_dir 0', '-r2c_dir 1', '-r2c_dir 2', ' '],
                 'default_value': ' ',
                }
               ),
               ('UniformFloat',
                {'name': 'p7',
                 'lower': 0,
                 'upper': 1,
                 'default_value': 1,
                }
               ),
               ('UniformFloat',
                {'name': 'p8',
                 'lower': 0,
                 'upper': 1,
                 'default_value': 1,
                }
               ),
               ('UniformFloat',
                {'name': 'p9',
                 'lower': 0,
                 'upper': 1,
                 'default_value': 1,
                }
               ),
               ('Constant',
                {'name': 'c0',
                 'value': 'BACKEND',
                }
               ),
              ]

def customize_space(self, class_size):
    # Exception if architecture is not sufficiently defined by self or plopper
    REQ_ATTRS = {'threads_per_node', 'gpus', 'ranks_per_node'}
    defined_by_self = [_ for _ in REQ_ATTRS if hasattr(self, _)]
    REQ_ATTRS = REQ_ATTRS.difference(set(defined_by_self))
    defined_by_plopper = [_ for _ in REQ_ATTRS if hasattr(self.plopper, _)]
    REQ_ATTRS = REQ_ATTRS.difference(set(defined_by_plopper))
    if len(REQ_ATTRS) > 0:
        raise ValueError("Self and Plopper do not sufficiently define attributes to customize space\nMissing attributes: ", REQ_ATTRS)

    altered_space = self.input_space
    self.node_count, self.app_scale = class_size

    # App scale sets constant size
    altered_space[1] = ('Constant', {'name': 'p1', 'value': self.app_scale})

    # Node scale determines depth scalability
    self.max_cpus = self.threads_per_node if 'threads_per_node' in defined_by_self else self.plopper.threads_per_node
    self.gpus = self.gpus if 'gpus' in defined_by_self else self.plopper.gpus
    self.ppn = self.ranks_per_node if 'ranks_per_node' in defined_by_self else self.plopper.ranks_per_node
    c0_value = 'cufft' if self.gpus > 0 else 'fftw'
    altered_space[10] = ('Constant', {'name': 'c0', 'value': c0_value})

    self.node_scale = self.node_count * self.ppn
    # Don't exceed #threads across total ranks
    self.max_depth = self.max_cpus // self.ppn
    sequence = [2**_ for _ in range(1,10) if (2**_) <= self.max_depth]
    if len(sequence) >= 2:
        intermediates = []
        prevpow = sequence[1]
        for rawpow in sequence[2:]:
            if rawpow+prevpow >= self.max_depth:
                break
            intermediates.append(rawpow+prevpow)
            prevpow = rawpow
        sequence = sorted(intermediates + sequence)
    # Ensure max_depth is always in the list
    if np.log2(self.max_depth)-int(np.log2(self.max_depth)) > 0:
        sequence = sorted(sequence+[self.max_depth])
    #altered_space[9] = ('Ordinal', {'name': 'p9', 'sequence': sequence, 'default_value': self.max_depth})
    self.sequence = sequence
    self.input_space = altered_space

APP_SCALES = [64,128,256,512,1024,1400,2048]
APP_SCALE_NAMES = ['N','S','M','L','XL','H','XH']
NODE_SCALES = [1,2,4,8,16,32,64,128]
lookup_ival = dict(((k1,k2),f"{v2}_{k1}") for (k2,v2) in zip(APP_SCALES, APP_SCALE_NAMES) for k1 in NODE_SCALES)

class TopologyCache(UserDict):
    # We utilize this dictionary as a hashmap++, so KeyErrors don't matter
    # If the key doesn't exist, we'll create it and its value, then want to store it
    # to operate as a cache for known keys. As such, this subclass permits the behavior with
    # light subclassing of the UserDict object

    candidate_orders = [_ for _ in itertools.product([0,1,2], repeat=3) if len(_) == len(set(_))]

    def __getitem__(self, key):
        if key not in self.data:
            self.data[key] = self.make_topology(key)
        return super().__getitem__(key)

    def make_topology(self, budget: int) -> list[tuple[int,int,int]]:
        # Powers of 2 that can be represented in topology X/Y/Z
        factors = [2 ** x for x in range(int(np.log2(budget)),-1,-1)]
        topology = []
        for candidate in itertools.product(factors, repeat=3):
            # All topologies need to have product that == budget
            # Reordering the topology is not considered a relevant difference, so reorderings are discarded
            if np.prod(candidate) != budget or \
               np.any([tuple([candidate[_] for _ in order]) in topology for order in self.candidate_orders]):
                continue
            topology.append(candidate)
        # Add the null space
        topology += [' ']
        return topology

    def __repr__(self):
        return "TopologyCache:"+super().__repr__()


# Introduce the post-sampling space alteration for scale
class heffte_plopper(LibE_Plopper):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Bash these values with a hammer its ok
        self.evaluation_tries = 1
        self.topology_keymap = {'P7': '-ingrid', 'P8': '-outgrid'}
        self.topology_cache = TopologyCache()

    def gpu_cleanup(self, outfile, attempt, dictVal, *args, **kwargs):
        if not hasattr(self, 'gpus') or self.gpus < 1:
            return
        # Re-identify run string
        runstr = self.runString(outfile, attempt, dictVal, *args, **kwargs)
        if '--hostfile' not in runstr:
            return
        cmdList = runstr.split()
        nodefile = cmdList[cmdList.index('--hostfile')+1]
        with open(nodefile, 'r') as f:
            n_nodes = len(f.readlines())
        if "aprun" in runstr:
            cleanup_cmd = f"aprun -n {n_nodes} -N 1 --hostfile {nodefile} ./gpu_cleanup.sh speed3d_r2c"
        elif "mpiexec" in runstr:
            cleanup_cmd = f"mpiexec -n {n_nodes} --ppn 1 --hostfile {nodefile} ./gpu_cleanup.sh speed3d_r2c"
        else:
            raise ValueError("Not aprun or mpiexec -- no GPU cleanup known")
        status = subprocess.run(cleanup-cmd, shell=True)
        if status.returncode != 0:
            raise ValueError(f"Cleanup Command failed to run (Return code: {status.returncode})")

    def set_architecture_info(self, **kwargs):
        super().set_architecture_info(**kwargs)
        # Machine identifier changes proper invocation to utilize allocated resources
        # Also customize timeout based on application scale per system
        self.known_timeouts = {}
        if 'polaris' in self.machine_identifier:
            self.cmd_template = "mpiexec -n {mpi_ranks} --ppn {ranks_per_node} --depth {depth} --cpu-bind depth --env OMP_NUM_THREADS={depth} sh ./set_affinity_gpu_polaris.sh {interimfile}"
            polaris_timeouts = {64:10.0, 128: 10.0, 256: 10.0, 512: 10.0}
            self.known_timeouts.update(polaris_timeouts)
        elif 'theta' in self.machine_identifier:
            self.cmd_template = "aprun -n {mpi_ranks} -N {ranks_per_node} -cc depth -d {depth} -j {j} -e OMP_NUM_THREADS={depth} sh {interimfile}"
            theta_timeouts = {64: 20.0, 128: 40.0, 256: 60.0}
            self.known_timeouts.update(theta_timeouts)
        self.node_scale = self.nodes * self.ranks_per_node
        # Don't exceed #threads across total ranks
        self.max_depth = max(1, self.threads_per_node // self.ranks_per_node)
        sequence = [2**_ for _ in range(1,10) if (2**_) <= self.max_depth]
        if len(sequence) >= 2:
            intermediates = []
            prevpow = sequence[1]
            for rawpow in sequence[2:]:
                if rawpow+prevpow >= self.max_depth:
                    break
                intermediates.append(rawpow+prevpow)
                prevpow = rawpow
            sequence = sorted(intermediates + sequence)
        # Ensure max_depth is always in the list
        if np.log2(self.max_depth)-int(np.log2(self.max_depth)) > 0:
            sequence = sorted(sequence+[self.max_depth])
        self.sequence = sequence

    def topology_interpret(self, config: dict) -> dict:
        machine_info = config['machine_info']
        if config['P1'] in self.known_timeouts.keys():
            config['machine_info']['app_timeout'] = self.known_timeouts[config['P1']]
            machine_info['app_timeout'] = self.known_timeouts[config['P1']]
        budget = machine_info['mpi_ranks']
        topology = self.topology_cache[budget]
        # Replace each key with uniform bucketized value
        for topology_key in self.topology_keymap.keys():
            selection = min(int(float(config[topology_key]) * len(topology)), len(topology)-1)
            selected_topology = topology[selection]
            if type(selected_topology) is not str:
                selected_topology = f"{self.topology_keymap[topology_key]} {' '.join([str(_) for _ in selected_topology])}"
            config[topology_key] = selected_topology
        # Replace sequence value
        config['P9'] = machine_info['sequence'][int(len(machine_info['sequence']) * float(config['P9']))]
        # Fix numpy zero-dimensional
        for k,v in config.items():
            if k not in self.topology_keymap.keys() and type(v) is np.ndarray and v.shape == ():
                config[k] = v.tolist()
        return config

    def floatcast(self, DataFrame, machine_info, real_topology=False):
        # Return a copy, leave original frame alone
        copied = DataFrame.copy()

        sequence = np.asarray(machine_info['sequence'])

        if real_topology:
            seq_len = 3*(len(str(max(DataFrame['mpi_ranks'])))+1)
            altered_topologies = np.empty((len(DataFrame), len(self.topology_keymap.keys())), dtype=f"<U{seq_len}")
        else:
            altered_topologies = np.empty((len(DataFrame), len(self.topology_keymap.keys())), dtype=int)
        altered_sequence = np.empty((len(DataFrame), 1), dtype=int)

        # Figure out whether P9 is upper/lower case
        p9_key = 'p9' if 'p9' in DataFrame.columns else 'P9'
        assert p9_key in DataFrame.columns
        # Topology keymap is always in upper case, so may have to temp-cast it
        if p9_key.lower() == p9_key:
            topkeys = [k.lower() for k in self.topology_keymap.keys()]
        else:
            topkeys = list(self.topology_keymap.keys())

        # Groupby budgets for more efficient processing
        for (gidx, group) in DataFrame.groupby('mpi_ranks'):
            budget = group.loc[group.index[0], 'mpi_ranks']
            # Topology
            topology = self.topology_cache[budget]
            # Topology must be differentiably cast, but doesn't need to be representative per se
            if real_topology:
                topology = np.asarray([" ".join([str(v) for v in t]) for t in topology[:-1]]+[topology[-1]])
            else:
                topology = np.arange(len(topology))
            for tidx, topology_key in enumerate(topkeys):
                # Initial selection followed by boundary fixing, then substitute from array
                # Gaussian Copula CAN over/undersample, so you have to fix that too
                selection = (group[topology_key] * len(topology)).astype(int)
                selection = selection.apply(lambda s: max(min(s, len(topology)-1), 0))
                selection = topology[selection]
                altered_topologies[group.index, tidx] = selection
            # Sequence
            selection = (group[p9_key] * len(sequence)).astype(int)
            selection = selection.apply(lambda s: max(min(s, len(sequence)-1), 0))
            altered_sequence[group.index] = sequence[selection].reshape(len(selection),1)
        # Substitute values and return
        for key, replacement in zip(topkeys+[p9_key],
                                    np.hsplit(altered_topologies, altered_topologies.shape[1])+[altered_sequence]):
            copied[key] = replacement
        return copied

    def createDict(self, x, params, *args, **kwargs):
        dictVal = {}
        for p, v in zip(params, x):
            if type(v) is np.ndarray and v.shape == ():
                v = v.tolist()
            dictVal[p] = v
        # Machine info should be available via kwargs['extrakeys']['machine_info']
        dictVal.setdefault('machine_info', kwargs['extrakeys']['machine_info'])
        dictVal = self.topology_interpret(dictVal)
        return dictVal


import pdb
#pdb.set_trace()
__getattr__ = libe_problem_builder(lookup_ival, input_space, HERE, name='heFFTe_Problem',
                                    customize_space=customize_space, plopper_class=heffte_plopper)

