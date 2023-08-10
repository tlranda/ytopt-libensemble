from ytopt.search.optimizer import Optimizer
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from ConfigSpace import ConfigurationSpace, EqualsCondition

CONFIGSPACE_SEED = 1234
YTOPT_SEED = 2345

cs = CS.ConfigurationSpace(seed=CONFIGSPACE_SEED)
c0 = CSH.Constant(name='c0', value='x')
p0 = CSH.UniformIntegerHyperparameter(name='p0', lower=1, upper=60) # MPI Threads
p1 = CSH.UniformIntegerHyperparameter(name='p1', lower=1, upper=4) # Roibin Threads
p2 = CSH.UniformIntegerHyperparameter(name='p2', lower=1, upper=4) # Binning Threads
p3 = CSH.UniformIntegerHyperparameter(name='p3', lower=1, upper=4) # Blosc internal threads
cs.add_hyperparameters([c0,p0,p1,p2,p3])

ytoptimizer = Optimizer(
    num_workers = 4,
    space = cs,
    learner = 'RF',
    liar_strategy = 'cl_max',
    acq_func = 'gp_hedge',
    set_KAPPA = 1.96,
    set_SEED = YTOPT_SEED,
    set_NI = 3,
)

# ADJUST HOW WARNINGS WORK
import traceback
import warnings
import sys

def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))
warnings.showwarning = warn_with_traceback

import pdb
pdb.set_trace()
"""
# Relevant portion of the trace for warning hunting

  /home/trandall/ytune_23/python_package_links/ytopt/search/optimizer/optimizer.py(155)ask_initial() -- OK
-> self._optimizer.tell(x,y)
  /home/trandall/ytune_23/python_package_links/skopt/optimizer/optimizer.py(611)tell() -- OK
-> return self._tell(x, y, fit=fit)

VARIANTS::
  /home/trandall/ytune_23/python_package_links/skopt/optimizer/optimizer.py(654)_tell()
-> Xtt = self.space.imp_const.fit_transform(self.space.transform(self.Xi))
  /home/trandall/ytune_23/python_package_links/skopt/optimizer/optimizer.py(675)_tell()
-> X = self.space.imp_const.fit_transform(self.space.transform(X_s))
  /home/trandall/ytune_23/python_package_links/skopt/optimizer/optimizer.py(738)_tell() -- 654, 675 OK but not 738
-> self._next_x = self.space.inverse_transform(next_x.reshape((1, -1)))[0]
END VARIANTS::

DEVIATION FOR 738 ONLY::
  /home/trandall/ytune_23/python_package_links/skopt/space/space.py(1308)inverse_transform() -- different route
-> Xt = self.imp_const_inv.fit_transform(Xt)
  /home/trandall/.conda/envs/ytune_23/lib/python3.10/site-packages/sklearn/utils/_set_output.py(140)wrapped()
-> data_to_wrap = f(self, X, *args, **kwargs)
END END DEVIATION::

  /home/trandall/.conda/envs/ytune_23/lib/python3.10/site-packages/sklearn/base.py(878)fit_transform()
-> return self.fit(X, **fit_params).transform(X)
  /home/trandall/.conda/envs/ytune_23/lib/python3.10/site-packages/sklearn/impute/_base.py(429)fit()
-> self.statistics_ = self._dense_fit(
> /home/trandall/.conda/envs/ytune_23/lib/python3.10/site-packages/sklearn/impute/_base.py(533)_dense_fit()
-> return np.full(X.shape[1], fill_value, dtype=X.dtype)

"""

init_points = ytoptimizer.ask_initial(n_points=4)
init_results = [-0.136147, -0.140752, -0.137717, -0.138195]

results = []
for point, res in zip(init_points, init_results):
    field_params = {}
    for field in ['c0'] + [f'p{_}' for _ in range(4)]:
        field_params[field] = point[field]
    results += [(field_params, res)]

ytoptimizer.tell(results)
