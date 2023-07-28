import argparse
import pathlib
import pandas as pd
import numpy as np

def build():
    prs = argparse.ArgumentParser()
    prs.add_argument('--merge-from', '--from', nargs='+', default=None, help="Directories to grab results from")
    prs.add_argument('--merge-to', '--to', nargs='+', default=None, help="Directories to place results in (must be ONE or same number of arguments as --merge-from)")
    return prs

def parse(args=None, prs=None):
    if prs is None:
        prs = build()
    if args is None:
        args = prs.parse_args()

    # Fix argparse not treating nargs well
    for name in ['merge_from', 'merge_to']:
        local = getattr(args, name)
        if type(local) is str:
            setattr(args, name, [local])

    # Validate merging pattern
    if len(args.merge_to) == 1:
        args.merge_to = args.merge_to * len(args.merge_from)
    elif len(args.merge_to) > 1 and len(args.merge_to) != len(args.merge_from):
        UnclearMergingPattern = f"Merge to has {len(args.merge_to)} entries, unable to match to merge from with {len(args.merge_from)}. Merge to must be same length or just one entry."
        raise ValueError(UnclearMergingPattern)
    return args

def merge(_from, _to):
    fromdir = pathlib.Path(_from)
    todir = pathlib.Path(_to)

    # Direct copy without clobbering
    migrations = [fromdir.joinpath('ytopt.log'),
                  fromdir.joinpath('ensemble.log'), ]

    # Filter out things that do not exist
    migrations = [_ for _ in migrations if _.exists()]
    migrations.extend([_ for _ in fromdir.glob('qsub*.batch')])
    migrations.extend([_ for _ in fromdir.glob('run*.py')])
    for workerdir in fromdir.glob('worker*'):
        globbed = [_ for _ in workerdir.joinpath('tmp_files').glob('*.sh')]
        migrations.extend(globbed)
        globbed = [_ for _ in workerdir.joinpath('tmp_files').glob('*.log')]
        migrations.extend(globbed)

    # Create directories to migrate
    todir.mkdir(parents=True, exist_ok=True)
    directories = set([todir.joinpath(*_.parts[1:-1]) for _ in migrations])
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    for migrate_from in migrations:
        migrate_from.rename(todir.joinpath(pathlib.Path(*migrate_from.parts[1:])))

    # Actually combine files
    numpy_files = []
    for _dir in [fromdir, todir]:
        full = _dir.joinpath('full_H_array.npz')
        if full.exists():
            with open(full, 'rb') as f:
            numpy_files.append(np.load(f))
        for _ in _dir.glob('libE_history_at_abort_*.npy'):
            with open(_, 'rb') as f:
                numpy_files.append(np.load(f))
    if len(numpy_files) > 0:
        np_dtype = numpy_files[0].dtype.descr
        indices = [len(_) for _ in numpy_files]
        combined = np.empty(sum(indices), dtype=np_dtype)
        # The only column that doesn't really work to naively combine is sim_id, but that's ok to duplicate
        for idx, npf in enumerate(numpy_files):
            for name in np_dtype.names:
                combined[name][sum(indices[:idx]):] = npf[name]
        combined.save(todir.joinpath('combined.npz'))

    csv_files = {}
    for _dir in [fromdir, todir]:
        available_csvs = [_ for _ in _dir.glob('*results.csv')]
        unique_stems = set([_.stem for _ in available_csvs])
        for stem in unique_stems:
            if stem not in csv_files.keys():
                csv_files[stem] = []
        for csv in available_csvs:
            csv_files[csv.stem].append(csv)
    for stem in in csv_files.keys():
        loaded = pd.concat([pd.read_csv(_) for _ in csv_files[stem]])
        loaded.to_csv(todir.joinpath(stem+'.csv'), index=False)

def main(args=None):
    args = parse(args)
    for merge_from, merge_to in zip(args.merge_from, args.merge_to):
        merge(merge_from, merge_to)

if __name__ == '__main__':
    main()

