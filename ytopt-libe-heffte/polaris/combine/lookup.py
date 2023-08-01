import pandas as pd
import numpy as np
import argparse

def build(prs=None):
    if prs is not None:
        return prs
    prs = argparse.ArgumentParser()
    prs.add_argument('--tl-csv', nargs='+', default=None, help="TL CSV to lookup records from")
    prs.add_argument('--oracle-csv', nargs='+', default=None, help="Result CSVs to lookup")
    prs.add_argument('--tpn', type=int, default=None, help="Identify TPN for TL CSV")
    prs.add_argument('--rpn', type=int, default=None, help="Identify RPN for TL CSV")
    return prs

def parse(args=None, prs=None):
    prs = build(prs)
    if args is None:
        args = prs.parse_args()
    for attrname in ['tl_csv', 'oracle_csv']:
        localattr = getattr(args, attrname)
        if type(localattr) is str:
            setattr(args, attrname, [localattr])
    return args

def load(li):
    full = []
    for name in li:
        try:
            csv = pd.read_csv(name)
            csv.insert(len(csv.columns), 'fname', [name] * len(csv))
            full.append(csv)
        except:
            print(f"Could not open {name}. Skipping.")
    return pd.concat(full).reset_index(drop=False)

def tl_augment(df, args):
    if args.tpn is not None:
        df.insert(len(df.columns), 'threads_per_node', [args.tpn] * len(df))
    if args.rpn is not None:
        df.insert(len(df.columns), 'ranks_per_node', [args.rpn] * len(df))
    return df


def main(args=None):
    args = parse(args)
    oracle_data = load(args.oracle_csv)
    transfer_data = tl_augment(load(args.tl_csv), args)
    # Find matching records in oracle if they exist
    match_params = [f'p{_}' for _ in range(10)] + ['c0', 'threads_per_node', 'ranks_per_node', 'mpi_ranks']
    for index in transfer_data.index:
        match_values = tuple(transfer_data.loc[index,match_params].values)
        n_matching_cols = (oracle_data[match_params] == match_values).sum(1)
        full_match_idx = np.where(n_matching_cols == len(match_params))[0]
        matched = oracle_data.iloc[full_match_idx]
        if len(matched) > 0:
            import pdb
            pdb.set_trace()
            print(transfer_data.iloc[index])

if __name__ == '__main__':
    main()

