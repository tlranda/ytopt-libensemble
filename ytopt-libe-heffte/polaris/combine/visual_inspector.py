import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import warnings

def build(prs=None):
    if prs is None:
        prs = argparse.ArgumentParser()
    prs.add_argument("--directory", nargs="*", help="Directories to directly include in results aggregration (default: None)")
    prs.add_argument("--crawl-directory", nargs="*", help="Directories to crawl for subdirectory names (default: None)")
    prs.add_argument("--highlight", type=int, default=-1, help="Focus results on the nth (0-indexed) directory (default: ALL directories)")
    prs.add_argument("--timeout", type=float, default=20.0, help="Virtual timeout to include on graph (default: %(default)s)")
    prs.add_argument("--flops-only", action="store_true", help="Drop runtime from the plots")
    return prs

def parse(prs=None, args=None):
    if prs is None:
        prs = build()
    if args is None:
        args = prs.parse_args()
    if args.directory is None:
        args.directory = []
    if args.crawl_directory is not None:
        for dirname in args.crawl_directory:
            for found in os.listdir(dirname):
                args.directory.extend([name for name in found if os.path.isdir(os.path.join(dirname,name))])
    if args.highlight >= 0:
        args.directory = [args.directory[args.highlight]]
    args.directory = sorted(args.directory)
    return args

def load(args):
    frames = []
    for dirname in args.directory:
        try:
            frame = pd.read_csv(os.path.join(dirname, "manager_results.csv"))
        except FileNotFoundError:
            try:
                frame = pd.read_csv(os.path.join(dirname, "results.csv"))
            except FileNotFoundError:
                no_results = f"No results file found in {dirname} -- skipping"
                warnings.warn(no_results)
                continue
            else:
                print(f"No manager results -- substituting with worker results")
        # Invert flops metric for evaluation
        frame["FLOPS"] = -1 * frame["FLOPS"]
        # Fetch # workers for data preparation purposes
        max_workers = frame.iloc[0]['libE_workers']
        # Put ORIGINAL index in first
        frame = frame.reset_index(drop=False)
        # Reordering to ensure mathematical correctness
        frame = frame.sort_values(by=['elapsed_sec', 'libE_id', ])
        # Elapsed time is difference between INDIVIDUAL WORKER'S previous timesteps
        prev_worker_time = dict((v+2, 0) for v in range(max_workers))
        elapsed_diff = []
        for idx, record in frame.iterrows():
            diff = record['elapsed_sec'] - prev_worker_time[record['libE_id']]
            prev_worker_time[record['libE_id']] = record['elapsed_sec']
            elapsed_diff.append(diff)
        # Continuous diff is difference between ANY WORKER'S previous timestep
        continuous_diff = [0] + frame['elapsed_sec'].tolist()
        continuous_diff = [b-a for a,b in zip(continuous_diff[:-1], continuous_diff[1:])]
        frame.insert(0, 'elapsed_diff', elapsed_diff)
        frame.insert(0, 'continuous_diff', continuous_diff)
        frames.append(frame)
    return frames

def observations(frame, args):
    max_workers = frame.iloc[0]['libE_workers']
    # Make per-worker observations
    diff_sum = dict((v+1, sum(frame[frame['libE_id'] == v+2]['elapsed_diff'])) for v in range(max_workers))
    cdiff_sum = dict((v+1, sum(frame[frame['libE_id'] == v+2]['continuous_diff'])) for v in range(max_workers))
    # Sanity check the time spent per worker and the time spent on the job
    print("Time per worker")
    print(diff_sum)
    # Parallel sum
    print("Evaluations per worker")
    print(dict((v+1, frame['libE_id'].tolist().count(v+2)) for v in range(max_workers)))
    # Should match final elapsed time
    print("Total time across all workers (parallel)")
    print(sum(cdiff_sum.values()))

    # Propose a timeout setting
    print("Average time per worker")
    avg_time = dict((v+1, np.mean(frame[frame['libE_id'] == v+2]['elapsed_diff'])) for v in range(max_workers))
    print(avg_time)
    print("Average time overall")
    print(np.mean(list(avg_time.values())))
    timed_out = dict((v+1, np.where(frame[frame['libE_id'] == v+2]['elapsed_diff'] >= args.timeout)[0].tolist()) for v in range(max_workers))
    timeouts = sum(map(len, timed_out.values()))
    if timeouts == 0:
        print(f"Proposed timeout=={args.timeout} would not clip any evaluations")
    else:
        print(f"With timeout=={args.timeout}, {timeouts} evaluations would be removed (per-worker evaluation # shown):")
        drop_keys = [k for k in timed_out.keys() if len(timed_out[k]) == 0]
        for key in drop_keys:
            del timed_out[key]
        print(timed_out)

    # Check for signs the SCRIPT could be wrong
    deviations = dict((v+1, len(np.where(frame[frame['libE_id'] == v+2]['elapsed_diff'].to_numpy() < 0.0)[0])) for v in range(max_workers))
    if sum(deviations.values()) > 0:
        print("Weird deviations?")
        print(deviations)
    else:
        print("No unusual elapsed seconds timings")
    # Final check on frame contents for a glimpse of anything else that may be weird
    print(frame[['index','libE_id','elapsed_sec','elapsed_diff','continuous_diff','FLOPS']])
    print()

if __name__ == '__main__':
    args = parse()
    frames = load(args)
    for frame in frames:
        observations(frame, args)

    fig, ax = plt.subplots()
    if not args.flops_only:
        lines = []
        mpl_lines = []
        for frame in frames:
            #newline = sns.lineplot(frame, x='elapsed_sec', y='FLOPS', estimator=None, marker='.')
            newline = sns.lineplot(frame, x='index', y='elapsed_diff', estimator=None, marker='.')
            lines.append(newline)
            mpl_lines.extend(newline.lines)
        #ax.hlines(0.0, xmin=0, xmax=max([max(frame['elapsed_sec']) for frame in frames]))
        ax.hlines(0.0, xmin=0, xmax=max([max(frame['index']) for frame in frames]))
        ax.hlines(args.timeout, xmin=0, xmax=max([max(frame['index']) for frame in frames]))

    for frame, color in zip(frames, matplotlib.colors.TABLEAU_COLORS):
        sns.lineplot(frame, x='index', y='FLOPS', estimator=None, marker='+',
                     color=color, markeredgecolor=color, linestyle='--', linewidth=1)
        if args.flops_only:
            ax.hlines(frame['FLOPS'].to_numpy().mean(), xmin=0, xmax=max([max(frame['index']) for frame in frames]), color=color)

    #plt.legend(labels=nice_names, loc='best')
    plt.show()

