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
               ('Categorical',
                {'name': 'p7',
                 'choices': ['-ingrid 4 1 1', '-ingrid 2 2 1', '-ingrid 2 1 2', ' '],
                 'default_value': ' ',
                }
               ),
               ('Categorical',
                {'name': 'p8',
                 'choices': ['-outgrid 4 1 1', '-outgrid 2 2 1', '-outgrid 2 1 2', ' '],
                 'default_value': ' ',
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

lookup_ival = {16: ('N', 'MINI'), 40: ('S', 'SMALL')}
__getattr__ = libe_problem_builder(lookup_ival, input_space, HERE, name='heFFTe_Problem')

