import numpy as np
import pandas as pd
import pathlib
import itertools

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
    csv = original_csv[original_csv['FLOPS'] > 0].sort_values(by='FLOPS')
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

import pdb
pdb.set_trace()
for idx, (file, selected) in enumerate(zip(INSPECT, selections)):
    #print(file)
    # Compare distances to BEST performing data
    best = selected.iloc[-1]
    best_vals = best[param_cols]
    # TODO: Weight columns by importance

