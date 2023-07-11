import numpy as np
NUMPY_SEED = 1
np.random.seed(NUMPY_SEED)
import pandas as pd
import os

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from ConfigSpace import ConfigurationSpace, EqualsCondition
from sdv.single_table import GaussianCopulaSynthesizer as GaussianCopula
from sdv.metadata import SingleTableMetadata
from sdv.sampling.tabular import Condition
from sdv.constraints import ScalarRange

# Customize behavior
dirtarget = "xingfu_openmc" # Dir to crawl for data
cutoff = 0.3 # Training cutoff
GPU_TARGET = 64 # Number of GPUs to sample for
N_CONFIGS = 100000 # Representation of space size
IDEAL = 0.10 # Target for GC to identify
ATTRITION = 0.05 # Proportion of actually ideal samples the GC cannot sample after fitting
DEFAULT_SAMPLE = 30 # If budget can't be generated, export this many samples
SUCCESS = 0.95 # Required confidence in budget

try:
    from math import comb
except ImportError:
    from math import factorial
    def comb(n,k):
        return factorial(n) / (factorial(k) * factorial(n-k))
def hypergeo(i,p,t,k):
    return (comb(i,t)*comb((p-i),(k-t))) / comb(p,k)

# Load data
static_imports = [f"{dirtarget}/{_}" for _ in os.listdir(dirtarget)]
loaded = []
for _ in static_imports:
    csv = pd.read_csv(_)
    # Fix the list indexing
    for idx in range(7):
        csv[f'p{idx}'] = csv[f'p{idx}'].apply(lambda x: str(x[2:-2]) if x[1] == "'" else int(x[1:-1]))
    index = csv.index.tolist()
    entries = len(index)
    # Add data from filename
    fname = os.path.basename(_).rsplit(".",1)[0]
    nGPUs = int(fname.split('-',1)[0][7:])
    nWorkers = int(fname.split('-')[3])
    csv.insert(0, 'nGPUs', [nGPUs] * entries)
    csv.insert(0, 'nWorkers', [nWorkers] * entries)
    # Drop to top-30%
    top_quant = np.quantile(csv.RUNTIME.values, cutoff)
    #print(csv.RUNTIME.max(), top_quant, csv.RUNTIME.min())
    csv = csv.loc[csv['RUNTIME'] <= top_quant]
    # Ensure numerical columns are numerical
    loaded.append(csv)
data = pd.concat(loaded)
# NEED TO DROP TO TOP-30% by PERFORMANCE ('RUNTIME')
data_trimmed = data[[f'p{_}' for _ in range(7)]+['nGPUs', 'nWorkers']]
# Create model
conditions = [Condition({'nGPUs': GPU_TARGET,}, num_rows=N_CONFIGS)]
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data_trimmed)
constraints = [{'constraint_class': 'ScalarRange',
                    'constraint_parameters': {
                        'column_name': 'nGPUs',
                        'low_value': 1,
                        'high_value': 512,
                        'strict_boundaries': False,},
               },
               {'constraint_class': 'ScalarRange',
                    'constraint_parameters': {
                        'column_name': 'p1',
                        'low_value': 500000,
                        'high_value': 8000000,
                        'strict_boundaries': False,}
               },
               {'constraint_class': 'ScalarRange',
                    'constraint_parameters': {
                        'column_name': 'p2',
                        'low_value': 100,
                        'high_value': 100000,
                        'strict_boundaries': False,}
               },
               {'constraint_class': 'ScalarRange',
                    'constraint_parameters': {
                        'column_name': 'p3',
                        'low_value': 0,
                        'high_value': 100000,
                        'strict_boundaries': False,}
               },
              ]
model = GaussianCopula(metadata, enforce_min_max_values=False)
model.add_constraints(constraints=constraints)
model.fit(data_trimmed)
# Oversampling to determine budget
sampled = model.sample_from_conditions(conditions).drop_duplicates()
Initial_I = int(N_CONFIGS * IDEAL)
C = len(sampled)
Reduce_I = int(ATTRITION * (N_CONFIGS - C))
I = max(1, Initial_I - Reduce_I)
print(f"Initial space size: {N_CONFIGS}")
print(f"Model generated: {C} ({100*C/N_CONFIGS:.2f}% coverage from top {cutoff} with {len(data_trimmed)} rows of input")
print(f"Initial IDEAL size: {Initial_I}")
print(f"Model Assumed IDEAL size: {I} (-{100*(1-(I/Initial_I)):.2f}% with attrition of {ATTRITION})")
if I > C:
    print(f"Ideal population smaller than biased population!")
    n_sample = DEFAULT_SAMPLE
else:
    k = 1
    while k < I:
        confidence = sum([hypergeo(I,C,_,k) for _ in range(1,k+1)])
        if confidence >= SUCCESS:
            break
        k += 1
        print(f"Attempt: {k}/{I} | Confidence: {confidence:.4f}", end='\r')
    print()
    confidence = sum([hypergeo(I,C,_,k) for _ in range(1, k+1)])
    if k < I or confidence >= SUCCESS:
        n_sample = k
        print(f"Reach {SUCCESS} confidence with {n_sample} samples")
    else:
        n_sample = DEFAULT_SAMPLE
        print(f"DID NOT REACH {SUCCESS} CONFIDENCE :: Max {100*confidence:.2f}% probability with {k} samples")
sampled.reset_index(drop=True).iloc[:n_sample].to_csv("tl_proposed.csv", index=False)

