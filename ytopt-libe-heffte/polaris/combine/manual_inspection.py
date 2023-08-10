import pandas as pd, numpy as np
import pathlib
import warnings
import itertools
from scipy.stats import entropy
import matplotlib.pyplot as plt
from sdv.single_table import GaussianCopulaSynthesizer as GaussianCopula
from sdv.metadata import SingleTableMetadata
from sdv.sampling.tabular import Condition
from sdv.constraints import ScalarRange

compare_cols = [f'p{_}' for _ in range(10)]
convert_cols = [f'p{_}' for _ in range(7)]

def make_dist(value_dict, data):
    # Use value dictionary to get distribution histogram for this dataset
    breakdown = {}
    common_denom = len(data)
    for key in convert_cols:
        values = value_dict[key]
        keydata = list(data[key])
        breakdown[key] = [keydata.count(val) / common_denom for val in values]
    # These need to be bucketized separately
    for col in sorted(set(compare_cols).difference(set(convert_cols))):
        key_ranges = value_dict[col][0]
        breakdown[col] = [0] * (len(key_ranges)-1)
        for idx, (a,b) in enumerate(zip(key_ranges[:-1], key_ranges[1:])):
            breakdown[col][idx] += len(data[np.logical_and(data[col] < b, data[col] >= a)])  / common_denom
    return breakdown

# Get the compared distribution
truth = pd.read_csv('logs/ThetaSourceTasks/Theta_001n_0064a/manager_results.csv')
truth['FLOPS'] *= -1
truth = truth[truth['FLOPS'] > 0]
truth = truth.sort_values(by=['FLOPS',]).reset_index(drop=True)
print(f"Loaded {len(truth)} values for YTOPT true distribution")
# Filter to what we'd like to see things look like
quantile = 0.8
best = truth[truth['FLOPS'] >= truth['FLOPS'].quantile(quantile)]
print(f"Selected {len(best)} values for YTOPT best distribution (quantile = {quantile})")


# Determine available values ish
value_dict = dict((k, sorted(set(truth[k]))) for k in convert_cols)
candidate_orders = [_ for _ in itertools.product([0,1,2], repeat=3) if len(_) == len(set(_))]
budget = 64
factors = [2 ** x for x in range(int(np.log2(budget)),-1,-1)]
topology = []
for candidate in itertools.product(factors, repeat=3):
    if np.prod(candidate) != budget or np.any([tuple([candidate[_] for _ in order]) in topology for order in candidate_orders]):
        continue
    topology.append(candidate)
topology = [','.join([str(_) for _ in candidate]) for candidate in topology]
topology.append(' ')
for col in ['p7', 'p8']:
    value_dict[col] = (np.append(np.arange(len(topology))/len(topology), 1.0), topology)
for col in ['p9']:
    value_dict[col] = (np.append(np.arange(2)/2, 1.0), [2,4])

# Get source data to TL from
source_files = [
'logs/ThetaSourceTasks/Theta_002n_0064a/manager_results.csv',
'logs/ThetaSourceTasks/Theta_004n_0064a/manager_results.csv',
'logs/ThetaSourceTasks/Theta_008n_0064a/manager_results.csv',
'logs/ThetaSourceTasks/Theta_016n_0064a/manager_results.csv',
'logs/ThetaSourceTasks/Theta_032n_0064a/manager_results.csv',
'logs/ThetaSourceTasks/Theta_064n_0064a/manager_results.csv',
'logs/ThetaSourceTasks/Theta_128n_0064a/manager_results.csv',
]
sources = [pd.read_csv(_) for _ in source_files]
for source in sources:
    source['FLOPS'] *= -1
    source = source.sort_values(by=['FLOPS',]).reset_index(drop=True)
source_dist = pd.concat(sources).reset_index(drop=True)
print(f"Loaded {len(source_dist)} values for TL distribution")

# Set up TL model
conditions = [Condition({'mpi_ranks': 64,
                         'p1': 64,},
                        num_rows=200)]
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
               {'constraint_class': 'ScalarRange', # P7 limit
                    'constraint_parameters': {
                        'column_name': 'p7',
                        'low_value': 0,
                        'high_value': 1,
                        'strict_boundaries': False,},
                    },
               {'constraint_class': 'ScalarRange', # P8 limit
                    'constraint_parameters': {
                        'column_name': 'p8',
                        'low_value': 0,
                        'high_value': 1,
                        'strict_boundaries': False,},
                    },
               {'constraint_class': 'ScalarRange', # P9 limit
                    'constraint_parameters': {
                        'column_name': 'p9',
                        'low_value': 0,
                        'high_value': 1,
                        'strict_boundaries': False,},
                    },
              ]
train_data = source_dist[['c0',]+compare_cols+['mpi_ranks', 'FLOPS']]
train_data = train_data[train_data['FLOPS'] > train_data['FLOPS'].quantile(quantile)].drop(columns=['FLOPS'])
print(f"Selected {len(train_data)} values for TL Training distribution (quantile = {quantile})")
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(train_data)
warnings.simplefilter('ignore')
model = GaussianCopula(metadata, enforce_min_max_values=False)
model.add_constraints(constraints=constraints)
model.fit(train_data)
warnings.simplefilter('default')
# Conditional sampling
cond_samples = model.sample_from_conditions(conditions)

# Make the comparable distributions
true_dist = make_dist(value_dict, truth)
best_dist = make_dist(value_dict, best)
tl_dist = make_dist(value_dict, source_dist)
cond_dist = make_dist(value_dict, cond_samples)

# Compare
kl_div = np.zeros((len(compare_cols), 4))
for idx, col in enumerate(compare_cols):
    for subidx, (name, a_dist, b_dist) in enumerate(zip(['TL vs Truth', 'TL vs Best', 'CS vs Truth', 'CS vs Best'],
                                                     [tl_dist, tl_dist, cond_dist, cond_dist],
                                                     [true_dist, best_dist, true_dist, best_dist])):
        kl_div[idx,subidx] = entropy(a_dist[col], b_dist[col])
        if not np.isfinite(kl_div[idx,subidx]):
            kl_div[idx,subidx] = entropy(b_dist[col], a_dist[col])
    td = np.asarray(true_dist[col])
    bs = np.asarray(best_dist[col])
    tld = np.asarray(tl_dist[col])
    csd = np.asarray(cond_dist[col])
    warnings.simplefilter('ignore')
    tld_truth_arr = np.asarray([_ for _ in td * np.log(td / tld) if np.isfinite(_)])
    csd_truth_arr = np.asarray([_ for _ in td * np.log(td / csd) if np.isfinite(_)])
    tld_best_arr = np.asarray([_ for _ in bs * np.log(bs / tld) if np.isfinite(_)])
    csd_best_arr = np.asarray([_ for _ in bs * np.log(bs / csd) if np.isfinite(_)])
    warnings.simplefilter('default')
    if col in convert_cols:
        options = value_dict[col]
    else:
        options = value_dict[col][1]
    print(col, "KL Div:\t", kl_div[idx])
    print("Options:\t",options)
    print("True Dist:\t", td, (td * len(truth)).astype(int), sum((td * len(truth)).astype(int)))
    print("Best Dist:\t", bs, (bs * len(best)).astype(int), sum((bs * len(best)).astype(int)))
    print("TL Data Dist:\t", tld, (tld * len(source_dist)).astype(int), sum((tld * len(source_dist)).astype(int)))
    print("Conditional Dist:\t", csd, (csd * len(cond_samples)).astype(int), sum((csd * len(cond_samples)).astype(int)))
    print("TL vs Truth Partial KL:\t", tld_truth_arr, tld_truth_arr.sum())
    print("CS vs Truth Partial KL:\t", csd_truth_arr, csd_truth_arr.sum())
    print("TL vs Best Partial KL:\t", tld_best_arr, tld_best_arr.sum())
    print("CS vs Best Partial KL:\t", csd_best_arr, csd_best_arr.sum())
    print()
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(options)), td, label="True Dist", marker='.')
    ax.plot(np.arange(len(options)), bs, label="Best Dist", marker='+')
    ax.plot(np.arange(len(options)), tld, label="TL Dist", linestyle='--', marker='o')
    ax.plot(np.arange(len(options)), csd, label="CS Dist", linestyle='--', marker='*')
    ax.set_xticks(np.arange(len(options)))
    ax.set_xticklabels(options)
    ax.legend()
    ax.set_title(f"KL Divergence of {col}: {kl_div[idx]}")
    fig.savefig(f"KL_Div_{col}.png", dpi=400)
#plt.show()

