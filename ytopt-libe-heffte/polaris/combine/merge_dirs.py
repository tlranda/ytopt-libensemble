import argparse
import pathlib
import pandas as pd
import numpy as np
import signal
import pdb

args = None

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

def iterate_version(path, counter):
    return path.parent.joinpath(path.stem + f'_v{counter}' + path.suffix)

def version(path):
    if not path.exists():
        return path
    counter = 0
    while iterate_version(path, counter).exists():
        counter += 1
    return iterate_version(path, counter)

change_record = {}

def merge(_from, _to):
    global change_record
    fromdir = pathlib.Path(_from)
    todir = pathlib.Path(_to)
    change_record[todir] = []

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
    # Relative to fromdir prevents paths such as fromdir = a/b/c from recreating the a/b path within
    # the todir (which prevents different fromdir stems from properly merging within the todir)
    subdirs = set([todir.joinpath(*_.relative_to(fromdir).parts[:-1]) for _ in migrations if len(_.relative_to(fromdir).parts[:-1]) > 1])
    for directory in subdirs:
        directory.mkdir(parents=True, exist_ok=True)
    for migrate_from in migrations:
        proposal = version(todir.joinpath(pathlib.Path(*migrate_from.relative_to(fromdir).parts)))
        migrate_from.rename(proposal)
        change_record[todir].append(('migrate', str(proposal), [str(migrate_from)]))
        print(change_record[todir][-1])

    # Actually combine files
    numpy_files = []
    numpy_fnames = []
    for _dir in [todir, fromdir]:
        prior = _dir.joinpath('combined.npy')
        if prior.exists():
            with open(prior, 'rb') as f:
                numpy_files.append(np.load(f))
                numpy_fnames.append(prior)
            # Should only exist for todir, but we want it first and to ignore other copied
            # data
            continue
        full = _dir.joinpath('full_H_array.npz')
        if full.exists():
            with open(full, 'rb') as f:
                numpy_files.append(np.load(f))
                numpy_fnames.append(full)
        for _ in _dir.glob('libE_history_at_abort_*.npy'):
            with open(_, 'rb') as f:
                numpy_files.append(np.load(f))
                numpy_fnames.append(_)
    if len(numpy_files) > 0:
        # I think dtypes should be constant, but sometimes evidently not? So ensure the largest
        # Dtype is used and drop defining it if/when not present
        dtypes = [_.dtype for _ in numpy_files]
        np_dtype = dtypes[np.argmax([len(_) for _ in dtypes])]
        indices = [len(_) for _ in numpy_files]
        combined = np.empty(sum(indices), dtype=np_dtype.descr)
        # The only column that doesn't really work to naively combine is sim_id, but that's ok to duplicate
        for idx, npf in enumerate(numpy_files):
            for name in np_dtype.names:
                if name not in npf.dtype.names:
                    continue
                start_idx = sum(indices[:idx])
                end_idx = start_idx + indices[idx]
                combined[name][start_idx:end_idx] = npf[name]
        np_out = todir.joinpath('combined')
        np.save(np_out, combined)
        change_record[todir].append(('numpy_combine', str(np_out.with_suffix('.npy')), [str(_) for _ in numpy_fnames]))
        print(change_record[todir][-1])
        print("\t", f"With {len(combined)} records")

    csv_files = {}
    for _dir in [todir, fromdir]:
        available_csvs = [_ for _ in _dir.glob('*results.csv')]
        unique_stems = set([_.stem for _ in available_csvs]).difference(set(csv_files.keys()))
        for stem in unique_stems:
            csv_files[stem] = []
        for csv in available_csvs:
            csv_files[csv.stem].append(csv)
    for stem in csv_files.keys():
        loaded = pd.concat([pd.read_csv(_) for _ in csv_files[stem]])
        csv_out = todir.joinpath(stem+'.csv')
        loaded.to_csv(csv_out, index=False)
        change_record[todir].append(('csv_combine', str(csv_out), [str(_) for _ in csv_files[stem]]))
        print(change_record[todir][-1])
        print("\t", f"With {len(loaded)} records")

    # Any and all remaining files and directories
    queue = [fromdir]
    while len(queue) > 0:
        qlen = len(queue)
        for q in queue[:qlen]:
            for remain in q.iterdir():
                if remain.is_symlink():
                    continue
                elif remain.is_dir():
                    queue.append(remain)
                    continue
                else:
                    rel_path = pathlib.Path(*remain.relative_to(fromdir).parts)
                    dest = version(todir.joinpath(rel_path))
                    remain.rename(dest)
                    change_record[todir].append(('cleanup', str(dest), [str(remain)]))
                    print(change_record[todir][-1])
        # Trim processed portions of the queue
        queue = queue[qlen:]

def log_moves(signum=None, frame=None):
    global args
    global change_record
    for todir in change_record.keys():
        log = version(todir.joinpath('migration.log'))
        with open(log, 'w') as f:
            f.write("reason; destination_file; list_of_source_files\n")
            for record in change_record[todir]:
                f.write(";".join([str(_) for _ in record])+"\n")

def main(passed_args=None):
    global args
    if passed_args is None:
        passed_args = args
    args = parse(passed_args)
    signal.signal(signal.SIGINT, log_moves)
    signal.signal(signal.SIGTERM, log_moves)
    #try:
    for merge_from, merge_to in zip(args.merge_from, args.merge_to):
        merge(merge_from, merge_to)
    #except KeyboardInterrupt:
    #    log_moves(signum=signal.SIGINT)
    #    raise
    # Write change record
    log_moves()

if __name__ == '__main__':
    main()

