from GC_TLA.base_problem import libe_problem_builder
import os, subprocess, numpy as np
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

def customize_space(self):
    altered_space = self.space
    self.app_scale, self.node_scale

    # App scale sets constant size
    altered_space[1] = ('Constant', {'value': self.app_scale})

    # Node scale determines depth scalability
    proc = subprocess.run(['nproc'], capture_output=True)
    if proc.returncode == 0:
        self.max_cpus = int(proc.stdout.decode('utf-8').strip())
    else:
        proc = subprocess.run(['lscpu'], capture_output=True)
        for line in proc.stdout.decode('utf-8'):
            if 'CPU(s):' in line:
                self.max_cpus = int(line.rstrip().rsplit(' ',1)[1])
                break
    print(f"Detected {self.max_cpus} CPUs on this machine")

    gpu_enabled = True
    if gpu_enabled:
        proc = subprocess.run('nvidia-smi -L'.split(' '), capture_output=True)
        if proc.returncode != 0:
            raise ValueError("No GPUs Detected, but in GPU mode")
        self.gpus = len(proc.stdout.decode('utf-8').strip().split('\n'))
        print(f"Detected {self.gpus} GPUs on this machine")
        self.ppn = self.gpus
    else:
        self.gpus = 0
        self.ppn = 2
    print(f"Set PPN to {self.ppn}"+"\n")

    self.node_count = self.node_scale // self.ppn
    print(f"APP_SCALE (AKA Problem Size X, X, X) = {self.app_scale} x3")
    print(f"NODE_SCALE (AKA System Size X * Y = Z) = {self.node_count} * {self.ppn} = {self.node_scale}")
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
    print(f"Given {self.node_count} nodes * {self.ppn} processes-per-node (={self.node_scale}) and {self.max_cpus} CPUS on this node...")
    print(f"Selectable depths are: {sequence}"+"\n")
    altered_space[9] = ('Ordinal', {'name': 'p9', 'sequence': sequence, 'default_value': self.max_depth})
    self.space = altered_space

APP_SCALES = [64,128,256,512,1024]
APP_SCALE_NAMES = ['N','S','M','L','XL']
NODE_SCALES = [2,4,6,8]
lookup_ival = dict(((k1,k2),f"{v2}_{k1}") for (k2,v2) in zip(APP_SCALES, APP_SCALE_NAMES) for k1 in NODE_SCALES)

import pdb
pdb.set_trace()
__getattr__ = libe_problem_builder(lookup_ival, input_space, HERE, name='heFFTe_Problem',
                                    customize_space=customize_space)

