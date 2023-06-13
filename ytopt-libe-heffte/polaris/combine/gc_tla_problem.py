from GC_TLA.base_problem import libe_problem_builder
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
APP_SCALES = [64,128,256,512,1024]
APP_SCALE_NAMES = ['N','S','M','L','XL']
NODE_SCALES = [2,4,6,8]
lookup_ival = dict(((k1,k2),f"{v2}_{k1}") for (k2,v2) in zip(APP_SCALES, APP_SCALE_NAMES) for k1 in NODE_SCALES)

def customize_space(self):
    self.space = altered_space

__getattr__ = libe_problem_builder(lookup_ival, input_space, HERE, name='heFFTe_Problem')

