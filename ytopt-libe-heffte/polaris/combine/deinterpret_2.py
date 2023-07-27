import pandas as pd, numpy as np
from collections import UserDict
import itertools
import argparse

class TopologyCache(UserDict):
    # We utilize this dictionary as a hashmap++, so KeyErrors don't matter
    # If the key doesn't exist, we'll create it and its value, then want to store it
    # to operate as a cache for known keys. As such, this subclass permits the behavior with
    # light subclassing of the UserDict object

    candidate_orders = [_ for _ in itertools.product([0,1,2], repeat=3) if len(_) == len(set(_))]

    def __getitem__(self, key):
        if key not in self.data:
            self.data[key] = self.make_topology(key)
        return super().__getitem__(key)

    def make_topology(self, budget: int) -> list[tuple[int,int,int]]:
        # Powers of 2 that can be represented in topology X/Y/Z
        factors = [2 ** x for x in range(int(np.log2(budget)),-1,-1)]
        topology = []
        for candidate in itertools.product(factors, repeat=3):
            # All topologies need to have product that == budget
            # Reordering the topology is not considered a relevant difference, so reorderings are discarded
            if np.prod(candidate) != budget or \
               np.any([tuple([candidate[_] for _ in order]) in topology for order in self.candidate_orders]):
                continue
            topology.append(candidate)
        # Add the null space
        topology += [' ']
        return topology

    def __repr__(self):
        return "TopologyCache:"+super().__repr__()

def build():
    prs = argparse.ArgumentParser()
    prs.add_argument('--csv', nargs="+", default=None, help="CSVs to de-interpret")
    prs.add_argument('--count-collisions', action='store_true', help="Determine number of identical records post-interpretation")
    prs.add_argument('--show', action='store_true', help="Print de-interpreted CSv")
    prs.add_argument('--save', nargs="+", default=None, help="Save each input CSV to a name (must be 1:1 with # of args to --csv)")
    return prs

def parse(args=None, prs=None):
    if prs is None:
        prs = build()
    if args is None:
        args = prs.parse_args()
    # List-ify if only one argument is present
    for name in ['csv', 'save']:
        local = getattr(args, name)
        if type(local) is str:
            setattr(args, name, [local])
    if args.save is not None and len(args.csv) != len(args.save):
        raise ValueError("--csv and --save must have same number of entries when --save is specified")
    elif args.save is None:
        args.save = [None] * len(args.csv)
    return args

def deinterpret(csvs, args):
    param_cols = [f'p{_}' for _ in range(10)] + ['c0']
    topCache = TopologyCache()
    top_keymap = {'P7': '-ingrid', 'P8': '-outgrid'}
    import pdb
    pdb.set_trace()
    for (csv, save) in zip(csvs, args.save):
        original_len = len(csv)
        # Reconstruct architecture info from records
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
        altered_topologies = np.empty((original_len, len(top_keymap.keys())), dtype=int)
        altered_sequence = np.empty((original_len, 1), dtype=int)
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
            topology = topCache[budget]
            # Topology must be differentiably cast, but doesn't need to be representative per se
            topology = np.arange(len(topology))
            for tidx, topology_key in enumerate(topkeys):
                # Initial selection followed by boundary fixing, then substitute from array
                # Gaussian Copula CAN over/undersample, so you have to fix that too
                selection = (group[topology_key] * len(topology)).astype(int)
                selection = selection.apply(lambda s: max(min(s, len(topology)-1), 0))
                selection = topology[selection]
                altered_topologies[group.index, tidx] = selection
            # Sequence
            selection = (group[p9_key] * len(sequence)).astype(int)
            selection = selection.apply(lambda s: max(min(s, len(sequence)-1), 0))
            altered_sequence[group.index] = sequence[selection].reshape(len(selection),1)
        # Substitute values and return
        for key, replacement in zip(topkeys+[p9_key],
                                    np.hsplit(altered_topologies, altered_topologies.shape[1])+[altered_sequence]):
            if key in topkeys:
                replacement = f"{top_keymap[key.upper()]} {' '.join([str(_) for _ in replacement])}"
            csv[key] = replacement
        if args.count_collisions:
            print(f"Collions: {original_len} --> {len(csv.drop_duplicates(subset=param_cols))}")
        if args.show:
            print(csv)
        if save is not None:
            csv.to_csv(save, index=False)

def main(args=None):
    args = parse(args)
    csvs = []
    for name in args.csv:
        try:
            csvs.append(pd.read_csv(name))
        except:
            print(f"Unable to load csv '{name}'")
    deinterpret(csvs, args)

if __name__ == '__main__':
    main()


