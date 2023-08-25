import numpy as np
import pandas as pd
import pathlib
import itertools
import matplotlib.pyplot as plt

from deinterpret import TopologyCache

#SYSTEM = "Polaris"
SYSTEM = "Theta"
INSPECT = sorted(pathlib.Path(f"logs/{SYSTEM}SourceTasks/").glob(f"{SYSTEM}_*n_*a/manager_results.csv"))

param_cols = [f'p{_}' for _ in range(10)]
topCache = TopologyCache()
top_keymap = {'P7': '-ingrid', 'P8': '-outgrid'}
alter_keys = ['p7_replace', 'p8_replace', 'p9_replace']
catkeys = [f'p{_}' for _ in range(7)]
renames = dict((k, k+'_replace') for k in catkeys)
catvals = [['double','float'], #p0
           [64,128,256,512,1024,1400], #p1
           ['-no-reorder','-reorder',' '], #p2
           ['-a2a','-a2av',' '], #p3
           ['-p2p','-p2p_pl',' '], #p4
           ['-pencils','-slabs',' '], #p5
           ['-r2c_dir 0','-r2c_dir 1','-r2c_dir 2',' '], #p6
          ]

selections = []

for file in INSPECT:
    original_csv = pd.read_csv(file)
    original_csv['FLOPS'] *= -1
    # Quantiles based on successful evaluations only
    csv = original_csv[original_csv['FLOPS'] > 0].sort_values(by=['FLOPS'])
    # Convert flexible parameters
    TPN = list(set(csv['threads_per_node']))[0]
    RPN = list(set(csv['ranks_per_node']))[0]
    max_depth = TPN // RPN
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
    if max_depth not in sequence:
        sequence = sorted(sequence+[max_depth])

    # De-interpret the topology
    altered_topologies = np.empty((len(csv), len(top_keymap.keys())), dtype=object)
    altered_sequence = np.empty((len(csv), 1), dtype=int)
    sequence = np.asarray(sequence)

    # Figure out whether P9 is upper/lower case
    p9_key = 'p9' if 'p9' in csv.columns else 'P9'
    # Topology keymap is always in upper case, so may have to temp-cast it
    if p9_key.lower() == p9_key:
        topkeys = [k.lower() for k in top_keymap.keys()]
    else:
        topkeys = list(top_keymap.keys())

    # Groupby budgets for more efficient processing
    for (gidx, group) in csv.groupby('mpi_ranks'):
        budget = group.loc[group.index[0], 'mpi_ranks']
        # Topology
        topology = np.asarray(topCache[budget], dtype=object)
        # Topology must be differentiably cast, but doesn't need to be representative per se
        for tidx, topology_key in enumerate(topkeys):
            # Initial selection followed by boundary fixing, then substitute from array
            # Gaussian Copula CAN over/undersample, so you have to fix that too
            altering = (group[topology_key] * len(topology)).astype(int)
            altering = altering.apply(lambda s: max(min(s, len(topology)-1), 0))
            altering = topology[altering]
            # Can't directly use group.index, have to put it relative to the subset
            relative_index = [csv.index.tolist().index(_) for _ in group.index]
            altered_topologies[relative_index, tidx] = altering
        # Sequence
        altering = (group[p9_key] * len(sequence)).astype(int)
        altering = altering.apply(lambda s: int(max(min(s, len(sequence)-1), 0)))
        # Can't directly use group.index, have to put it relative to the subset
        relative_index = [csv.index.tolist().index(_) for _ in group.index]
        altered_sequence[relative_index] = sequence[altering].reshape(len(altering),1)
    # Substitute altered values
    for key, replacement in zip(alter_keys,
                                np.hsplit(altered_topologies, altered_topologies.shape[1])+[altered_sequence]):
        csv.loc[csv.index, key] = replacement
    # Typecast fix
    csv = csv.astype({'p9_replace': 'int64'})
    # Convert categorical keys to floats
    for catkey, catval, rename in zip(catkeys, catvals, renames.items()):
        new_vals = np.zeros(len(csv))
        for idx, newval in enumerate(csv[catkey]):
            new_vals[idx] = catval.index(newval)/(len(catval)-1)
        csv.rename(columns=dict([rename]), inplace=True)
        csv.insert(0,catkey,new_vals)
    selections.append(csv)

del (topCache, top_keymap)

# Gower distance for similarity
# Based on wwwjk366's Gower library (github.com/wwwjk366/gower)
def gower(x,y, weights=None):
    x_n_rows, x_n_cols = x.shape
    y_n_rows, y_n_cols = y.shape
    # Logical not of when X dtype for column is a numpy number type for all columns
    cat_features = ~np.vectorize(np.issubdtype)(x.dtypes, np.number)
    x, y = x.to_numpy(), y.to_numpy()
    z = np.concatenate((x,y))
    x_index = np.arange(0, x_n_rows)
    y_index = np.arange(x_n_rows, x_n_rows+y_n_rows)

    Z_numeric = z[:, np.logical_not(cat_features)].astype(np.float64)
    Z_categor = z[:, cat_features]
    num_cols = Z_numeric.shape[1]
    num_ranges, num_max = np.zeros(num_cols), np.zeros(num_cols)

    if weights is None:
        weights = np.ones(z.shape[1])
    weight_cat = weights[cat_features]
    weight_num = weights[np.logical_not(cat_features)]
    weight_sum = weights.sum()

    # Normalize numeric data
    for col in range(num_cols):
        col_array = Z_numeric[:, col]
        col_min, col_max = np.nanmin(col_array), np.nanmax(col_array)
        if np.isnan(col_min):
            col_min = 0.0
        if np.isnan(col_max):
            col_max = 0.0
        num_max[col] = col_max
        num_ranges[col] = np.abs(1 - col_min/col_max) if (col_max != 0) else 0.0
    Z_numeric = np.divide(Z_numeric, num_max, out=np.zeros_like(Z_numeric), where=num_max != 0)

    # Splitting by kinds
    out = np.zeros((x_n_rows, y_n_rows), dtype=np.float64)
    X_cat = Z_categor[x_index,]
    X_num = Z_numeric[x_index,]
    Y_cat = Z_categor[y_index,]
    Y_num = Z_numeric[y_index,]

    for i in range(x_n_rows):
        j_start = 0 if x_n_rows != y_n_rows else i
        # Categorical
        sij_cat = np.where(X_cat[i,:] == Y_cat[j_start:y_n_rows,:], np.zeros_like(X_cat[i,:]), np.ones_like(X_cat[i,:]))
        sum_cat = np.multiply(weight_cat, sij_cat).sum(axis=1)
        # Numerical
        abs_delta = np.absolute(X_num[i,:]-Y_num[j_start:y_n_rows,:])
        sij_num = np.divide(abs_delta, num_ranges, out=np.zeros_like(abs_delta), where=num_ranges!=0)
        sum_num = np.multiply(weight_num, sij_num).sum(axis=1)
        # Combine
        sums = np.divide(np.add(sum_cat, sum_num), weight_sum)
        out[i,j_start:] = sums
        if x_n_rows == y_n_rows:
            out[i:, j_start] = sums
    return out

fig, ax = plt.subplots()
for idx, (file, selected) in enumerate(zip(INSPECT, selections)):
    # Compare distances to BEST performing data
    best = selected.iloc[-1]
    best_vals = best[param_cols]
    # TODO: Weight columns by importance
    best_frame = pd.DataFrame(best.values.reshape((1,-1)), columns=selected.columns).infer_objects()
    similarities = gower(best_frame, selected)
    if '1024a' in str(file):
        line = ax.plot(np.arange(similarities.shape[1]), similarities[0,:], label=file.parent.stem)
ax.set_ylabel("Gower Similarity to Known Optimum")
ax.set_xlabel("FLOP/s ascending rank")
ax.legend()
plt.show()

