import pandas as pd, numpy as np
from collections import UserDict
import itertools
import pathlib
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
        # Convert topologies to strings
        topology = [' '.join([str(_) for _ in candidate]) for candidate in topology]
        # Add the null space
        topology += [' ']
        return topology

    def __repr__(self):
        return "TopologyCache:"+super().__repr__()

def build():
    prs = argparse.ArgumentParser()
    prs.add_argument('--csv', nargs="+", default=None, help="CSVs to de-interpret")
    prs.add_argument('--count-collisions', action='store_true', help="Determine number of identical records post-interpretation")
    prs.add_argument('--cols', nargs="*", default=None, help="Columns to show in CSV printing routines (default: ALL)")
    prs.add_argument('--sort-flops', action='store_true', help="Sort rows by flops column (default: no sorting)")
    prs.add_argument('--pandas-display', type=int, default=pd.get_option('display.max_rows'), help='Change number of rows pandas prints before omitting results (default: %(default)s)')
    prs.add_argument('--show', action='store_true', help="Print de-interpreted CSV")
    prs.add_argument('--unique-show', action='store_true', help="Print uniques from de-interpreted CSV")
    prs.add_argument('--unique-all', action='store_true', help="ALWAYS show min/mean/max/variance breakdown, even for unique records")
    prs.add_argument('--enumerate', action='store_true', help="Count number of times each option represented in a CSV column appears")
    prs.add_argument('--save', nargs="+", default=None, help="Save each input CSV to a name (must be 1:1 with # of args to --csv)")
    prs.add_argument('--auto', default=None, help="Automatic renaming for CSV saving (not used by default; this suffix is added to filenames before the extension)")
    prs.add_argument('--def-threads-per-node', default=None, type=int, help="Provide default number of threads per node if not present in CSV")
    prs.add_argument('--def-ranks-per-node', default=None, type=int, help="Provide default number of ranks per node if not present in CSV")
    return prs

def parse(args=None, prs=None):
    if prs is None:
        prs = build()
    if args is None:
        args = prs.parse_args()
    # Autosave names
    if args.auto is not None:
        args.save = []
        for name in args.csv:
            p = pathlib.Path(name)
            args.save.append(p.with_stem(p.stem+args.auto))
    # List-ify if only one argument is present
    for name in ['csv', 'save', 'cols']:
        local = getattr(args, name)
        if type(local) is str:
            setattr(args, name, [local])
    if args.save is not None and len(args.csv) != len(args.save):
        raise ValueError("--csv and --save must have same number of entries when --save is specified")
    elif args.save is None:
        args.save = [None] * len(args.csv)
    return args

def deinterpret(csvs, names, args):
    param_cols = [f'p{_}' for _ in range(10)] + ['c0']
    topCache = TopologyCache()
    top_keymap = {'P7': '-ingrid', 'P8': '-outgrid'}
    for (csv, name, save) in zip(csvs, names, args.save):
        original_len = len(csv)
        # Reconstruct architecture info from records
        try:
            TPN = list(set(csv['threads_per_node']))[0]
        except:
            if hasattr(args, 'def_threads_per_node') and args.def_threads_per_node is not None:
                TPN = args.def_threads_per_node
            else:
                raise
        try:
            RPN = list(set(csv['ranks_per_node']))[0]
        except:
            if hasattr(args, 'def_ranks_per_node') and args.def_ranks_per_node is not None:
                RPN = args.def_ranks_per_node
            else:
                raise
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
        altered_topologies = np.empty((original_len, len(top_keymap.keys())), dtype=object)
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
            topology = np.asarray(topCache[budget], dtype=object)
            # Topology must be differentiably cast, but doesn't need to be representative per se
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
            csv[key] = replacement
        if args.count_collisions:
            print(f"{name} Collions: {original_len} --> {len(csv.drop_duplicates(subset=param_cols))}")
        if args.show:
            print(name)
            if args.sort_flops and 'FLOPS' in csv.columns:
                csv = csv.sort_values(by=['FLOPS'])
            if args.cols is None:
                print(csv)
            else:
                disp_cols = [_ for _ in args.cols if _ in csv.columns]
                mismatch = set(args.cols).difference(disp_cols)
                if len(mismatch) > 0:
                    print(f"The following columns were not found and cannot be printed: {sorted(mismatch)}")
                print(csv[disp_cols])
        if args.unique_show:
            print(name)
            param_cols = [f'p{_}' for _ in range(10)]+['c0']
            no_dupes = csv.drop_duplicates(subset=param_cols)
            dupes = csv[csv.duplicated(subset=param_cols)]
            if args.sort_flops and 'FLOPS' in no_dupes.columns:
                no_dupes = no_dupes.sort_values(by=['FLOPS'])
            if args.cols is None:
                print(no_dupes)
            else:
                disp_cols = [_ for _ in args.cols if _ in csv.columns]
                mismatch = set(args.cols).difference(disp_cols)
                if len(mismatch) > 0:
                    print(f"The following columns were not found and cannot be printed: {sorted(mismatch)}")
                print(no_dupes[disp_cols])
            for (iterdx, row) in no_dupes.iterrows():
                # Get the dupes that match this row
                search_tup = tuple(row[param_cols].values)
                n_matching = (dupes[param_cols] == search_tup).sum(1)
                full_matches = np.where(n_matching == len(param_cols))[0]
                flops = np.append(row['FLOPS'], dupes.loc[dupes.index[full_matches], 'FLOPS'].values)
                if len(flops) == 1 and not args.unique_all:
                    continue
                print(f"{row.to_frame().T[param_cols]} appears {len(flops)} times in the records")
                print("\t"+f"Min FLOPS: {flops.min()}")
                print("\t"+f"Mean FLOPS: {flops.mean()}")
                print("\t"+f"Max FLOPS: {flops.max()}")
                print("\t"+f"Var FLOPS: {flops.std()}")
        if save is not None:
            csv.to_csv(save, index=False)
        if args.enumerate:
            import pprint
            for col in csv.columns:
                # Sort based on count descending
                options = np.asarray(sorted(set(csv[col])))
                counts = np.asarray([csv[col].to_list().count(k) for k in options])
                sort = np.argsort(-counts)
                col_dict = dict((k, c) for (k, c) in zip(options[sort], counts[sort]))
                #col_dict = dict((k,csv[col].to_list().count(k)) for k in sorted(set(csv[col])))
                pprint.pprint({col: col_dict}, sort_dicts=False)

def main(args=None):
    args = parse(args)
    pd.set_option('display.min_rows', args.pandas_display)
    pd.set_option('display.max_rows', args.pandas_display)
    csvs = []
    names = []
    for name in args.csv:
        try:
            csvs.append(pd.read_csv(name))
            names.append(name)
        except:
            print(f"Unable to load csv '{name}'")
    deinterpret(csvs, names, args)

if __name__ == '__main__':
    main()


