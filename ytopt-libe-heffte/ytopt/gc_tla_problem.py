from GC_TLA.base_problem import libe_problem_builder
from GC_TLA.base_plopper import LibE_Plopper
import os
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
               ('UniformInt',
                {'name': 'p9',
                 'lower': 2,
                 'upper': 8,
                 'q': 2,
                 'default_value': 8,
                }
               ),
              ]

import itertools, numpy as np
topology_cache = {}
# Permutations of triplet orders that will be ignored as 'unique' topologies (first come first serve)
triplet_permutations = [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]
def make_topology(num_nodes: int) -> list[tuple[int,int,int]]:
    # Powers of two that can be represented in 3D topology
    factors = [2 ** x for x in range(int(np.log2(num_nodes)),-1,-1)]
    topology = []
    for candidate in itertools.product(factors, repeat=3):
        # 1) All topologies must multiplicatively use all nodes
        # 2) X/Y/Z order is not considered a relevant permutation to be a "new" topology
        if np.prod(candidate) != num_nodes or \
           np.any([tuple([candidate[_] for _ in order]) in topology for order in triplet_permutations]):
            continue
        topology.append(candidate)
    return topology

def interpret_topology(config: dict):
    num_nodes = config['nodes']
    if num_nodes not in topology_cache.keys():
        topology_cache[num_nodes] = make_topology(num_nodes)
    topology = topology_cache[num_nodes] + [' ']
    # Replace keys with uniform bucketized values
    for topology_key in ['ingrid', 'outgrid']:
        # Replace float with parameterization
        selection = min(int(config[topology_key] * len(topology)), len(topology)-1)
        selected_topology = topology[selection]
        if type(selected_topology) is not str:
            selected_topology = str(selected_topology)
            #selected_topology = f"-{topology_key} {' '.join([_ for _ in selected_topology])}"
        config[topology_key] = selected_topology
    return config

class heffte_Plopper(LibE_Plopper):
    def plotValues(self, outputfile, dictVal, *args, findReplace=None, **kwargs):
        dictVal = interpret_topology(dictVal)
        super().plotValues(outputfile, dictVal, *args, findReplace=findReplace, **kwargs)

    def runString(self, outfile, dictVal, *args, **kwargs):
        return f"/opt/cray/pe/pals/1.1.7/bin/mpiexec -n {dictVal['nodes']} --ppn 1 --depth {dictVal['P9']} sh {outputfile}"

    def getTime(self, process, out, errs, dictVal, *arg, **kwargs):
        # Looking for the line with 'Performance' in it
        # Should be XXX YYY ZZZ
        # Return -1 * YYY
        for handler in [out, errs]:
            try:
                for line in handler.decode('utf-8'):
                    if "Performance:" in line:
                        return -1 * line.split(' ')[1]
            except:
                pass
        UnableToParse = f"Unable to parse performance metric for {dictVal}"
        raise ValueError(UnableToParse)

lookup_ival = {16: ('N', 'MINI'), 40: ('S', 'SMALL')}
__getattr__ = libe_problem_builder(lookup_ival, input_space, HERE, name='heFFTe_Problem', plopper_class=heffte_Plopper)

