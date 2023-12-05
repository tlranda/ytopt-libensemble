import numpy as np
np.random.seed(1234)
import pandas as pd
import pathlib
import shutil
import argparse

def build_sequence(csv):
    # Collect key info for sequence from CSV -- raise error if we can't parse it deterministically
    mpi_ranks = sorted(set(csv['mpi_ranks']))
    if len(mpi_ranks) > 1:
        raise ValueError("Multiple MPI Rank kinds detected")
    else:
        mpi_ranks = mpi_ranks[0]
    ranks_per_node = sorted(set(csv['ranks_per_node']))
    if len(ranks_per_node) > 1:
        raise ValueError("Multiple ranks per node detected")
    else:
        ranks_per_node = ranks_per_node[0]
    num_nodes = mpi_ranks // ranks_per_node
    system_id = sorted(set(csv['machine_identifier']))
    if len(system_id) > 1:
        raise ValueError("Multiple systems detected")
    else:
        system_id = system_id[0]
    match system_id:
        case 'theta-knl':
            threads_per_node = 256
        case 'theta-gpu':
            threads_per_node = 128
        case 'polaris-gpu':
            threads_per_node = 64
        case 'cpu-polaris':
            threads_per_node = 64
        case _:
            raise ValueError(f"Unknown system id {_}")
    # Make sequence array
    max_depth = threads_per_node // ranks_per_node
    sequence = [2 ** _ for _ in range(1,10) if (2**_) <= max_depth]
    if len(sequence) >= 2:
        intermediates = []
        prevpow = sequence[1]
        for rawpow in sequence[2:]:
            if rawpow+prevpow >= max_depth:
                break
            intermediates.append(rawpow+prevpow)
            prevpow = rawpow
        sequence = sorted(intermediates + sequence)
    if np.log2(max_depth) - int(np.log2(max_depth)) > 0:
        sequence = sorted(sequence+[max_depth])
    if max_depth not in sequence:
        sequence = sorted(sequence+[max_depth])
    # Validate that the sequence list is correct
    present_sequence = sorted(set(csv['p9']))
    # Not every CSV will record all sequence values -- only an error if the CSV has a value that we couldn't reconstruct
    if len(set(present_sequence).difference(set(sequence))) > 0:
        raise ValueError(f"CSV Sequence includes: {present_sequence} -- Reconstructed Sequence: {sequence}")
    return sequence, threads_per_node

def validate(p9_sequence, converted, sequence):
    indexer = lambda x: int(x * len(sequence))
    for idx, (new, original) in enumerate(zip(converted, p9_sequence)):
        if original != sequence[indexer(new)]:
            raise ValueError(f"Failed to verify index {idx} (Original: {original} | New: {new})")

def convert(csv, sequence, threads_per_node):
    new_csv = csv.copy()
    # Document threads per node
    rpn_column = list(new_csv.columns).index('ranks_per_node')
    new_csv.insert(rpn_column, 'threads_per_node', [threads_per_node] * len(new_csv))
    # Convert p9 with noise based on the range values can take
    interval = 1/len(sequence)
    p9_sequence = new_csv['p9'].to_numpy()
    # Floors for each value
    baselines = np.array([_*interval for _ in range(len(sequence))])
    # Replace each p9_sequence value with its baseline value
    bases = baselines[np.searchsorted(sequence, p9_sequence)]
    # Create noise that stays within bound to next interval
    noise = interval * np.random.rand(len(p9_sequence))
    # Combine noise with the baselines
    converted = bases + noise
    # Guarantee that this converted sequence will re-convert correctly
    validate(p9_sequence, converted, sequence)
    new_csv['p9'] = converted
    return new_csv

def adjust_csvs(directory, deinterpret=False):
    # Collect files to fix
    fqueue = []
    for tldir in pathlib.Path(directory).iterdir():
        if not tldir.is_dir():
            continue
        for fpath in tldir.iterdir():
            if fpath.match("*manager_results*") and not fpath.match("*legacy*"):
                fqueue.append(fpath)
    for fpath in fqueue:
        legacy_path = pathlib.Path(str(fpath.parent)+"/manager_results_legacy.csv")
        if deinterpret and legacy_path.exists():
            shutil.copy2(legacy_path, fpath)
        csv = pd.read_csv(fpath)
        if csv['p9'].dtype in [np.float64, np.float32]:
            print(fpath, " is already converted")
            continue
        sequence, threads_per_node = build_sequence(csv)
        print(str(fpath.parent)+'/'+str(fpath.stem))
        print("sequence: ", sequence)
        new_csv = convert(csv, sequence, threads_per_node)
        # Backup old results before we clobber them
        shutil.copy2(fpath, str(fpath.parent)+"/manager_results_legacy.csv")
        new_csv.to_csv(fpath, index=False)

def build():
    prs = argparse.ArgumentParser()
    prs.add_argument("--dirs", required=True, nargs="+",
                     help="Directories to crawl for CSVs to fix (crawls subdirs of indicated dir for CSVs)")
    prs.add_argument("--deinterpret", action="store_true",
                     help="Find legacy files and replace them first")
    return prs

def parse(prs=None, args=None):
    if prs is None:
        prs = build()
    if args is None:
        args = prs.parse_args()
    return args

def main(args=None):
    args = parse(args=args)
    for directory in args.dirs:
        adjust_csvs(directory, deinterpret=args.deinterpret)

if __name__ == '__main__':
    main()

