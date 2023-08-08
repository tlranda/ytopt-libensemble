import pandas as pd, numpy as np
import pathlib
import warnings
import itertools
from scipy.stats import entropy
import matplotlib.pyplot as plt

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
# Filter to what we'd like to see things look like
best = truth[truth['FLOPS'] >= truth['FLOPS'].quantile(0.8)]
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
#'logs/ThetaSourceTasks/Theta_008n_0064a/manager_results.csv',
#'logs/ThetaSourceTasks/Theta_016n_0064a/manager_results.csv',
#'logs/ThetaSourceTasks/Theta_032n_0064a/manager_results.csv',
#'logs/ThetaSourceTasks/Theta_064n_0064a/manager_results.csv',
#'logs/ThetaSourceTasks/Theta_128n_0064a/manager_results.csv',
]
sources = [pd.read_csv(_) for _ in source_files]
for source in sources:
    source['FLOPS'] *= -1
    source = source.sort_values(by=['FLOPS',]).reset_index(drop=True)
    # Filter as we would normally
    source = source[source['FLOPS'] >= source['FLOPS'].quantile(0.8)]
source_dist = pd.concat(sources).reset_index(drop=True)

# Make the comparable distributions
true_dist = make_dist(value_dict, truth)
tl_dist = make_dist(value_dict, source_dist)

# Compare
kl_div = np.zeros((len(compare_cols), 1))
for idx, col in enumerate(compare_cols):
    kl_div[idx,0] = entropy(true_dist[col], tl_dist[col])
    if not np.isfinite(kl_div[idx,0]):
        kl_div[idx,0] = entropy(tl_dist[col], true_dist[col])
    td = np.asarray(true_dist[col])
    tl = np.asarray(tl_dist[col])
    warnings.simplefilter('ignore')
    arr = np.asarray([_ for _ in td * np.log(td / tl) if np.isfinite(_)])
    warnings.simplefilter('default')
    if col in convert_cols:
        options = value_dict[col]
    else:
        options = value_dict[col][1]
    print(col, "KL Div:\t", kl_div[idx,0])
    print("Options:\t",options)
    print("True Dist:\t", td, td * len(truth))
    print("TL Dist:\t", tl, tl * len(source_dist))
    print("Partial KL:\t", arr, arr.sum())
    print()
    fig, ax = plt.subplots()
    ax.plot(np.arange(len(options)), td, label="True Dist")
    ax.plot(np.arange(len(options)), tl, label="TL Dist")
    ax.set_xticks(np.arange(len(options)))
    ax.set_xticklabels(options)
    ax.legend()
    ax.set_title(f"KL Divergence of {col}: {kl_div[idx,0]}")
plt.show()

