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
                {'name': 'p1x',
                 'sequence': ['64','128','256','512','1024'],
                 'default_value': '128',
                }
               ),
               ('Ordinal',
                {'name': 'p1y',
                 'sequence': ['64','128','256','512','1024'],
                 'default_value': '128',
                }
               ),
               ('Ordinal',
                {'name': 'p1z',
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
               # BELOW HERE ARE DUMMY VALUES
               # They should be overwritten by the customize_space() call based on available resources
               # p7/p8 represent topologies as space-delimited strings with integer #ranks/dim for X/Y/Z
               # p9 represents selectable #threads as integers
               # c0 represents the FFT backend selection as a string (usually 'cufft' for GPUs, 'fftw' for CPUs)
               #
               # Implementers should be able to copy this template and update these values to fit
               # the actual tuning space. Any instantiated object's space should be updated via a
               # call to its plopper.set_architecture_info() and set_space() functions
               ('Categorical',
                {'name': 'p7',
                 'choices': ['1 1 1'],
                 'default_value': '1 1 1',
                }
               ),
               ('Categorical',
                {'name': 'p8',
                 'choices': ['1 1 1'],
                 'default_value': '1 1 1',
                }
               ),
               ('Categorical',
                {'name': 'p9',
                 'choices': [1],
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
    self.node_count, self.app_scale_x, self.app_scale_y, self.app_scale_z = class_size

    # App scale sets constant size
    altered_space[1] = ('Constant', {'name': 'p1x', 'value': self.app_scale_x})
    altered_space[2] = ('Constant', {'name': 'p1y', 'value': self.app_scale_y})
    altered_space[3] = ('Constant', {'name': 'p1z', 'value': self.app_scale_z})

    # Node scale determines depth scalability
    self.max_cpus = self.threads_per_node if 'threads_per_node' in defined_by_self else self.plopper.threads_per_node
    self.gpus = self.gpus if 'gpus' in defined_by_self else self.plopper.gpus
    self.ppn = self.ranks_per_node if 'ranks_per_node' in defined_by_self else self.plopper.ranks_per_node
    c0_value = 'cufft' if self.gpus > 0 else 'fftw'
    altered_space[12] = ('Constant', {'name': 'c0', 'value': c0_value})

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
def lookup_ival(nodes, x_dim, y_dim, z_dim):
    if nodes not in NODE_SCALES:
        Unknown_Node_Scale = f"{__file__} does not define node scale {nodes}. Available sizes: {NODE_SCALES}"
        raise ValueError(Unknown_Node_Scale)
    if any([scale not in APP_SCALES for scale in [x_dim, y_dim, z_dim]]):
        Unknown_App_Scale = f"{__file__} does not define one or more app scales {x_dim},{y_dim},{z_dim}. Available sizes: {APP_SCALES}"
        raise ValueError(Unknown_App_Scale)
    app_i2s = dict((i,s) for (i,s) in zip(APP_SCALES, APP_SCALE_NAMES))
    return f"{app_i2s[x_dim]}_{app_i2s[y_dim]}_{app_i2s[z_dim]}_{nodes}"
def inv_lookup_ival(s,default=False):
    if default:
        return (1,64,64,64)
    nodes = int(s.split('_')[-1])
    if nodes not in NODE_SCALES:
        Unknown_Node_Scale = f"{__file__} does not define node scale {nodes}. Available sizes: {NODE_SCALES}"
        raise ValueError(Unknown_Node_Scale)
    x,y,z = (_ for _ in s.split('_')[:-1])
    app_s2i = dict((s,i) for (i,s) in zip(APP_SCALES, APP_SCALE_NAMES))
    if any([scale not in APP_SCALE_NAMES for scale in [x, y, z]]):
        Unknown_App_Scale = f"{__file__} does not define one or more app scales {x},{y},{z}. Available sizes: {APP_SCALE_NAMES}"
        raise ValueError(Unknown_App_Scale)
    x,y,z = (app_s2i[_] for _ in [x,y,z])
    return (nodes, x, y, z)
#lookup_ival = dict(((k1,k2),f"{v2}_{k1}") for (k2,v2) in zip(APP_SCALES, APP_SCALE_NAMES) for k1 in NODE_SCALES)

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
            polaris_timeouts = {(64,64,64): 10.0,
                                (128,128,128): 10.0,
                                (256,256,256): 10.0,
                                (512,512,512): 10.0,
                               }
            self.known_timeouts.update(polaris_timeouts)
        elif 'theta' in self.machine_identifier:
            self.cmd_template = "aprun -n {mpi_ranks} -N {ranks_per_node} -cc depth -d {depth} -j {j} -e OMP_NUM_THREADS={depth} sh {interimfile}"
            theta_timeouts = {(64,64,64): 20.0,
                              (128,128,128): 40.0,
                              (256,256,256): 60.0,
                             }
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

    def createDict(self, x, params, *args, **kwargs):
        dictVal = {}
        for p, v in zip(params, x):
            if type(v) is np.ndarray and v.shape == ():
                v = v.tolist()
            # Insert -ingrid/-outgrid flags
            if p == 'P7':
                v = '-ingrid '+v
            elif p == 'P8':
                v = '-outgrid '+v
            dictVal[p] = v
        # Machine info should be available via kwargs['extrakeys']['machine_info']
        dictVal.setdefault('machine_info', kwargs['extrakeys']['machine_info'])
        machine_info = dictVal['machine_info']
        xyz = (int(dictVal['P1X']), int(dictVal['P1Y']), int(dictVal['P1Z']))
        if xyz in self.known_timeouts.keys():
            dictVal['machine_info']['app_timeout'] = self.known_timeouts[xyz]
            machine_info['app_timeout'] = self.known_timeouts[xyz]
        return dictVal


__getattr__ = libe_problem_builder(lookup_ival, inv_lookup_ival, input_space, HERE, name='heFFTe_Problem',
                                    customize_space=customize_space, plopper_class=heffte_plopper)

