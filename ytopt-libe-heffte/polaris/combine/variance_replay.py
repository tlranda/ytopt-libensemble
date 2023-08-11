import numpy as np
import pandas as pd
import pathlib
import itertools
import time

from plopper import Plopper
from deinterpret import TopologyCache

#SYSTEM = "Polaris"
SYSTEM = "Theta"
TARGET_REPLAY = sorted(pathlib.Path(f"logs/{SYSTEM}SourceTasks/").glob(f"{SYSTEM}_*n_*a/manager_results.csv"))
INCREMENT = 0.2
QUANTILES = np.arange(0,1+INCREMENT,INCREMENT)

REPEATS = 3
N_TO_RUN = 0
EXPECTED_RUNTIME = 0

param_cols = [f'p{_}' for _ in range(10)] + ['c0']
topCache = TopologyCache()
top_keymap = {'P7': '-ingrid', 'P8': '-outgrid'}

selections = []

for file in TARGET_REPLAY:
    original_csv = pd.read_csv(file)
    original_csv['FLOPS'] *= -1
    # Prepare elapsed times
    prev_worker_time = dict((v+2, 0) for v in range(original_csv.iloc[0]['libE_workers']))
    elapsed_diff = []
    for idx, record in original_csv.iterrows():
        diff = record['elapsed_sec'] - prev_worker_time[record['libE_id']]
        # Concatenated runs evidently may have incorrectly joined elapsed times
        if diff < 0.0:
            diff = record['elapsed_sec']
        prev_worker_time[record['libE_id']] = record['elapsed_sec']
        elapsed_diff.append(diff)
    # Quantiles based on successful evaluations only
    csv = original_csv[original_csv['FLOPS'] > 0].sort_values(by='FLOPS')
    # In some cases, extremely few successful evaluations could mean < len(QUANTILES) evaluations exist
    idxs = sorted(set([int(min(len(csv)-1, len(csv) * quant)) for quant in QUANTILES]))
    selected = csv.iloc[idxs]
    N_TO_RUN += (len(selected) * REPEATS)
    expectations = np.zeros(len(idxs))
    for immediate_idx, idx in enumerate(idxs):
        expectations[immediate_idx] = elapsed_diff[csv.iloc[idx].name]
    selected.insert(len(selected.columns), "Expected Runtime", expectations)
    EXPECTED_RUNTIME += (expectations.sum() * REPEATS)
    # Convert flexible parameters
    TPN = list(set(selected['threads_per_node']))[0]
    RPN = list(set(selected['ranks_per_node']))[0]
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
    altered_topologies = np.empty((len(selected), len(top_keymap.keys())), dtype=object)
    altered_sequence = np.empty((len(selected), 1), dtype=int)
    sequence = np.asarray(sequence)

    # Figure out whether P9 is upper/lower case
    p9_key = 'p9' if 'p9' in selected.columns else 'P9'
    # Topology keymap is always in upper case, so may have to temp-cast it
    if p9_key.lower() == p9_key:
        topkeys = [k.lower() for k in top_keymap.keys()]
    else:
        topkeys = list(top_keymap.keys())

    # Groupby budgets for more efficient processing
    for (gidx, group) in selected.groupby('mpi_ranks'):
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
            relative_index = [selected.index.tolist().index(_) for _ in group.index]
            altered_topologies[relative_index, tidx] = altering
        # Sequence
        altering = (group[p9_key] * len(sequence)).astype(int)
        altering = altering.apply(lambda s: max(min(s, len(sequence)-1), 0))
        # Can't directly use group.index, have to put it relative to the subset
        relative_index = [selected.index.tolist().index(_) for _ in group.index]
        altered_sequence[relative_index] = sequence[altering].reshape(len(altering),1)
    # Substitute altered values
    for key, replacement in zip(topkeys+[p9_key],
                                np.hsplit(altered_topologies, altered_topologies.shape[1])+[altered_sequence]):
        selected.loc[selected.index, key] = replacement
    selections.append(selected)
    del (original_csv, prev_worker_time, elapsed_diff, csv, idxs)

del (param_cols, topCache, top_keymap)

print("OVERALL:", N_TO_RUN, "Unique Runtimes")
print("\t", EXPECTED_RUNTIME, "(in seconds)")
print("\t", EXPECTED_RUNTIME/60.0, "(in minutes)")
print("\t", EXPECTED_RUNTIME/3600.0, "(in hours)")
print()

start_time = time.time()

for idx, (file, selected) in enumerate(zip(TARGET_REPLAY, selections)):
    print(file, '--', REPEATS * selected["Expected Runtime"].sum())
    pretend_flops, pretend_elapses = [1] * len(selected), [1] * len(selected)
    selected.insert(len(selected.columns), "FLOPS_R1", pretend_flops)
    selected.insert(len(selected.columns), "RUNTIME_R1", pretend_elapses)
    selected.insert(len(selected.columns), "FLOPS_R2", pretend_flops)
    selected.insert(len(selected.columns), "RUNTIME_R2", pretend_elapses)
    selected.insert(len(selected.columns), "FLOPS_R3", pretend_flops)
    selected.insert(len(selected.columns), "RUNTIME_R3", pretend_elapses)
    print(selected)

