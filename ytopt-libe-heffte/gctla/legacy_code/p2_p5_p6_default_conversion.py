import pandas as pd, numpy as np
import argparse

def build():
    prs = argparse.ArgumentParser()
    prs.add_argument("--inputs", nargs="+", default=None, required=True,
                    help="CSVs to change defaults (in-place)")
    return prs

def parse(args=None, prs=None):
    if prs is None:
        prs = build()
    if args is None:
        args = prs.parse_args()
    return args

def replace(column_name, df):
    if column_name == 'p2':
        if df['c0'].values.tolist()[0] == 'cufft':
            return '-no-reorder'
        else:
            return '-reorder'
    elif column_name == 'p5':
        return '-pencils'
    elif column_name == 'p6':
        return '-r2c_dir 0'
    else:
        raise ValueError(f"No conversion for column '{column_name}'")

def main(args=None):
    args = parse(args)

    fix_keys = [f"p{x}" for x in [2,5,6]]
    for name in args.inputs:
        df = pd.read_csv(name)
        print(f"Load {len(df)} items from {name}")
        for key in fix_keys:
            default_idx = np.where(df[key].astype(str) == ' ')[0]
            print("\t"+f"Found {len(default_idx)} default ' ' values for {key} at indices: {default_idx}")
            if len(default_idx) > 0:
                replacement = replace(key, df)
                print("\t\t"+f"Replacement value: '{replacement}'")
                df.loc[default_idx, key] = [replacement] * len(default_idx)
        print(f"Re-saving to {name}")
        df.to_csv(name, index=False)

if __name__ == '__main__':
    main()

