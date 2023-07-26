import argparse
import pathlib

def build():
    prs = argparse.ArgumentParser()
    prs.parse_args('--merge-from', '--from', nargs='+', default=None, help="Directories to grab results from")
    prs.parse_args('--merge-to', '--to', nargs='+', default=None, help="Directories to place results in (must be ONE or same number of arguments as --merge-from)")
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
                  fromdir.joinpath('ensemble.log'),
                  fromdir.glob('qsub_*.batch'),
                  fromdir.glob('run_*.py'),
                 ]
    for workerdir in fromdir.glob('worker*'):
        migrations.extend(fromdir.joinpath(workerdir).joinpath('tmp_files').glob('*.sh'))
        migrations.extend(fromdir.joinpath(workerdir).joinpath('tmp_files').glob('*.log'))
    for migrate_from in migrations:
        migrate(migrate_from, todir)
    # Actually combine files
    merges = [((_dir.joinpath('full_H_array.npz'), *_dir.glob('libE_history_at_abort_*.npy')) for _dir in [fromdir, todir]),
              ((_dir.joinpath('manager_results.csv'), _dir.joinpath('results.csv')) for _dir in [fromdir, todir]),
             ]
    for merge_from_options, merge_to_options in merges:
        combine(merge_from_options, merge_to_options)

def main(args=None):
    args = parse(args)
    for merge_from, merge_to in zip(args.merge_from, args.merge_to):
        merge(merge_from, merge_to)

if __name__ == '__main__':
    main()

