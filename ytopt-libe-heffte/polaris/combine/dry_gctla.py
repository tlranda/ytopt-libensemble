import os, pathlib
import glob
import numpy as np
NUMPY_SEED = 1
np.random.seed(NUMPY_SEED)
import pandas as pd
import itertools
import subprocess
from copy import deepcopy
import warnings
import argparse

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from ConfigSpace import ConfigurationSpace, EqualsCondition
from sdv.single_table import GaussianCopulaSynthesizer as GaussianCopula
from sdv.metadata import SingleTableMetadata
from sdv.sampling.tabular import Condition
from sdv.constraints import ScalarRange

def build():
    prs = argparse.ArgumentParser()
    prs.add_argument("--max-evals", type=int, default=30, help="Number of records to generate with GC")
    prs.add_argument("--input", nargs="+", default=None, help="CSVs to use as input")
    prs.add_argument("--quantile", type=float, help="Quantile to remove from inputs")
    prs.add_argument("--sys", type=int, help="Target system scale")
    prs.add_argument("--app", type=int, help="Target application scale")
    return prs

def parse(args=None, prs=None):
    if prs is None:
        prs = build()
    if args is None:
        args = prs.parse_args()
    return args

args = parse()

APP_SCALE = args.app
MPI_RANKS = args.sys
CONFIGSPACE_SEED = 1234
YTOPT_SEED = 2345

# Load model
cs = CS.ConfigurationSpace(seed=CONFIGSPACE_SEED)
# arg1  precision
p0 = CSH.CategoricalHyperparameter(name='p0', choices=["double", "float"], default_value="float")
# arg2  3D array dimension size
p1 = CSH.Constant(name='p1', value=APP_SCALE)
#p1 = CSH.OrdinalHyperparameter(name='p1', sequence=[64,128,256,512,1024], default_value=128)
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
# arg9
p8 = CSH.UniformFloatHyperparameter(name='p8', lower=0, upper=1)
# number of threads is hardware-dependent
p9 = CSH.UniformFloatHyperparameter(name='p9', lower=0, upper=1)

# Cross-architecture is out-of-scope for now so we determine this for the current platform and leave it at that
cpu_override = None
gpu_enabled = False
cpu_ranks_per_node = 1

c0 = CSH.Constant('c0', value='cufft' if gpu_enabled else 'fftw')

if cpu_override is None:
    proc = subprocess.run(['nproc'], capture_output=True)
    if proc.returncode == 0:
        threads_per_node = int(proc.stdout.decode('utf-8').strip())
    else:
        proc = subprocess.run(['lscpu'], capture_output=True)
        for line in proc.stdout.decode('utf-8'):
            if 'CPU(s):' in line:
                threads_per_node = int(line.rstrip().rsplit(' ',1)[1])
                break
    print(f"Detected {threads_per_node} CPU threads on this machine")
else:
    threads_per_node = cpu_override
    print(f"Override indicates {threads_per_node} CPU threads on this machine")
if cpu_ranks_per_node is None:
    cpu_ranks_per_node = threads_per_node
if gpu_enabled:
    proc = subprocess.run('nvidia-smi -L'.split(' '), capture_output=True)
    if proc.returncode != 0:
        raise ValueError("No GPUs Detected, but in GPU mode")
    gpus = len(proc.stdout.decode('utf-8').strip().split('\n'))
    print(f"Detected {gpus} GPUs on this machine")
    ranks_per_node = gpus
else:
    ranks_per_node = cpu_ranks_per_node
    print(f"CPU mode; force {ranks_per_node} ranks per node")
print(f"Set ranks_per_node to {ranks_per_node}"+"\n")

NODE_COUNT = max(MPI_RANKS // ranks_per_node,1)
print(f"APP_SCALE (AKA Problem Size X, X, X) = {APP_SCALE} x3")
print(f"MPI_RANKS (AKA System Size X * Y = Z) = {NODE_COUNT} * {ranks_per_node} = {MPI_RANKS}")
# Don't exceed #threads across total ranks
max_depth = threads_per_node // ranks_per_node
sequence = [2**_ for _ in range(1,10) if (2**_) <= max_depth]
if len(sequence) >= 2:
    intermediates = []
    prevpow = sequence[1]
    for rawpow in sequence[2:]:
        if rawpow+prevpow >= max_depth:
            break
        intermediates.append(rawpow+prevpow)
        prevpow = rawpow
    sequence = sorted(intermediates + sequence)
# Ensure max_depth is always in the list
if np.log2(max_depth)-int(np.log2(max_depth)) > 0:
    sequence = sorted(sequence+[max_depth])
if max_depth not in sequence:
    sequence = sorted(sequence+[max_depth])
print(f"Depths are based on {threads_per_node} threads on each node, shared across {ranks_per_node} MPI ranks on each node")
print(f"Selectable depths are: {sequence}"+"\n")
# arg10 number threads per MPI process
#p9 = CSH.OrdinalHyperparameter(name='p9', sequence=sequence, default_value=max_depth)

cs.add_hyperparameters([p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, c0])

MACHINE_IDENTIFIER = "theta-knl"

# For efficiency's sake, the condition makes batches of 100 samples at a time
conditions = [Condition({'mpi_ranks': args.sys,
                         'p1': args.app},
                        num_rows=100)]
constraints = [{'constraint_class': 'ScalarRange', # App scale limit
                    'constraint_parameters': {
                        'column_name': 'p1',
                        'low_value': 64,
                        'high_value': 2048,
                        'strict_boundaries': False,},
                    },
               {'constraint_class': 'ScalarRange', # System scale limit
                    'constraint_parameters': {
                        'column_name': 'mpi_ranks',
                        'low_value': 1,
                        'high_value': 16384,
                        'strict_boundaries': False,},
                    },
              ]
# Fetch problem instance and set its space based on alterations
import gc_tla_problem
app_scale_name = gc_tla_problem.lookup_ival[(NODE_COUNT, APP_SCALE)]
warnings.simplefilter('ignore') # I want the problem class to raise this warning, but I know about it and will properly handle it. No need to hear about the warning
problem = getattr(gc_tla_problem, app_scale_name) #f"{app_scale_name}_{NODE_COUNT}")
warnings.simplefilter('default')
problem.plopper.set_architecture_info(threads_per_node = ranks_per_node,
                                      gpus = ranks_per_node if gpu_enabled else 0,
                                      nodes = NODE_COUNT,
                                      mpi_ranks = MPI_RANKS,
                                      machine_identifier = MACHINE_IDENTIFIER,
                                      )
problem.set_space(cs)

cand_files = args.input # Can restore ignore pruning later
# In case some files are specified that don't exist, emit warning and best-effort continue
data_files = []
warned = False
for cand in cand_files:
    if pathlib.Path(cand).exists():
        data_files.append(cand)
    else:
        if not warned:
            print("\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
            warned = True
        print(f"WARNING: Indicated TL source file: {cand} does NOT exist!")
if warned:
    print("\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
    print("Will continue on best-effort basis with remaining files")
print(f"GC will be fitted against data from: {data_files}")
data = pd.concat([pd.read_csv(_) for _ in data_files])
data_trimmed = data[['c0',]+[f'p{_}' for _ in range(10)]+['mpi_ranks', 'FLOPS']]
# Drop configurations that had errors (not runtime failures); indicated by FLOPS >= 2.0
data_trimmed = data_trimmed[data_trimmed['FLOPS'] < 2.0]
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data_trimmed.drop(columns=['FLOPS']))

# Fitting
fittable = data_trimmed[data_trimmed['FLOPS'] <= data_trimmed['FLOPS'].quantile(args.quantile)]
fittable = fittable.drop(columns=["FLOPS"])
warnings.simplefilter('ignore')
model = GaussianCopula(metadata, enforce_min_max_values=False)
model.add_constraints(constraints=constraints)
model.fit(fittable)
warnings.simplefilter('default')

def remove_generated_duplicates(samples, history, dtypes):
    default_machine_info = {'sequence': sequence}
    casted = problem.plopper.floatcast(samples, default_machine_info)
    # Duplicate checking and selection
    casted.insert(0, 'source', ['cast'] * len(casted))
    if len(history) > 0:
        combined = pd.concat((history, casted)).reset_index(drop=False)
    else:
        combined = casted.reset_index(drop=False)
    match_on = list(set(combined.columns).difference(set(['source'])))
    duplicated = np.where(combined.duplicated(subset=match_on))[0]
    sample_idx = combined.loc[duplicated]['index']
    combined = combined.drop(index=duplicated)
    if len(duplicated) > 0:
        print(f"Dropping {len(duplicated)} duplicates from generation")
    else:
        print("No duplicates to remove")
    # Extract non-duplicated samples and ensure history is ready for future iterations
    samples.drop(index=sample_idx)
    combined['source'] = ['history'] * len(combined)
    if 'index' in combined.columns:
        combined = combined.drop(columns=['index'])
    return samples, combined

out_dtypes = [
    # MUST MATCH ORDER OF THE CONFIGSPACE HYPERPARAMETERS EXACTLY
    ('c0', "<U24", (1,)),
    ('p0', "<U24", (1,)),
    ('p1', int, (1,)),
    ('p2', "<U24", (1,)),
    ('p3', "<U24", (1,)),
    ('p4', "<U24", (1,)),
    ('p5', "<U24", (1,)),
    ('p6', "<U24", (1,)),
    ('p7', float, (1,)),
    ('p8', float, (1,)),
    ('p9', float, (1,)),
]

raw_predictions = model.sample_from_conditions(conditions)
cleaned, history = remove_generated_duplicates(raw_predictions, [], out_dtypes)
outdir = pathlib.Path('dry_gctla')
outdir.mkdir(parents=True, exist_ok=True)
cleaned.to_csv(outdir.joinpath('predicted_results.csv'), index=False)

