import pathlib
import argparse
import subprocess

def build(prs=None):
    if prs is not None:
        return prs
    prs = argparse.ArgumentParser()
    prs.add_argument("--tl-dir", help="Look for tl directories as subdirectories here")
    prs.add_argument("--oracle-dir", help="Look for oracle directories as subdirectories here")
    return prs

def parse(args=None, prs=None):
    prs = build(prs)
    if args is None:
        args = prs.parse_args()
    return args

def main(args=None):
    args = parse(args)
    tl_options = [_ for _ in pathlib.Path(args.tl_dir).iterdir()]
    oracle_options = [_ for _ in pathlib.Path(args.oracle_dir).iterdir() if 'Default' not in str(_) and _.is_dir()]
    oracle_lookup = {}
    for opt in oracle_options:
        portions = str(opt.stem).split('_')
        try:
            nodes = int(portions[1][:-1])
            app   = int(portions[2][:-1])
        except:
            print(opt, 'failed to translate; excluding')
        else:
            oracle_lookup[(nodes, app)] = opt
    for tl in tl_options:
        portions = str(tl.stem).split('_')
        nodes = int(portions[4][:-1])
        app   = int(portions[5][:-1])
        try:
            oracle = oracle_lookup[(nodes, app)]
        except:
            print(f"No oracle for {(nodes, app)}; skipping {tl}")
        else:
            call = f"python3 lookup.py --tl-csv {tl}/predicted_results.csv --oracle-csv {oracle}/manager_results.csv --tpn 256 --rpn 64".split()
            print(' '.join(call))
            subprocess.run(call)

if __name__ == '__main__':
    main()

