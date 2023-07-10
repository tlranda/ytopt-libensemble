import argparse
import time
import pandas as pd
import numpy as np
from plopper import Plopper

def build(prs=None):
    if prs is None:
        prs = argparse.ArgumentParser()
    prs.add_argument("--input", nargs="+", required=True,
                     help="CSV files to read and directly evaluate configurations")
    prs.add_argument("--outdir", default="./",
                     help="Output directory used by Plopper (default: %(default)s)")
    prs.add_argument("--csv", default="results.csv",
                     help="Plopper's results are written to this file")
    return prs

def parse(prs=None, args=None):
    if prs is None:
        prs = build()
    if args is None:
        args = prs.parse_args()
    return args

def load(args):
    frames = []
    for name in args.input:
        try:
            frames.append(pd.read_csv(name))
        except:
            print(f"Failed to read {name}")
    return pd.concat(frames)

def evaluate_csvs(data, args):
    plopper_obj = Plopper("speed3d.sh", args.outdir)
    params = data.columns.tolist()
    results = []
    start_time = time.time()
    for idx, line in data.iterrows():
        line = line.tolist()
        # For OpenMC Plopper, we need to prepend this
        os.system("./processexe.pl exe.pl " +str(line[4])+ " " +str(line[5])+ " " +str(value[6]))
        os.environ["OMP_NUM_THREADS"] = str(value[4])
        result = plopper_obj.findRuntime(line, params, 1)
        res_time = time.time()
        results.append(line + [result, res_time-start_time])
    return results

def export(data, results, args):
    original_columns = data.columns.tolist()
    # We need to remove these values (if present) to avoid confusion
    drop_lookup = ['objective', 'predicted', 'elapsed_sec']
    found = []
    for lookup in drop_lookup:
        if lookup in original_columns:
            found.append(original_columns.index(lookup))
        else:
            found.append(None)
    for index in sorted(set(found).difference(set([None])), reverse=True):
        del original_columns[index]
        for record in results:
            del record[index]
    # Remaining columns + our two added values from the plopper running
    frame = pd.DataFrame(results, columns=original_columns+['objective','elapsed_sec'])
    frame.to_csv(args.csv, index=False)

def main(args=None):
    args = parse(args=args)
    data = load(args)
    results = evaluate_csvs(data, args)
    export(data, results, args)
    print(results)

if __name__ == '__main__':
    main()

