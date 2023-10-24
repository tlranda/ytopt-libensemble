import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import warnings
import re

def build(prs=None):
    if prs is None:
        prs = argparse.ArgumentParser()
    prs.add_argument("--directory", "--directories", nargs="*", help="Directories to directly include in results aggregration (default: None)")
    prs.add_argument("--rename", nargs="*", help="Rename each directory from --directory (cannot use with --crawl-directory)")
    prs.add_argument("--title", default=None, help="Set plot title")
    prs.add_argument("--no-resort", action="store_true", help="Enforce order of directories from command line (default: re-sorted)")
    prs.add_argument("--crawl-directory", "--crawl-directories", nargs="*", help="Directories to crawl for subdirectory names (default: None)")
    prs.add_argument("--highlight", type=int, default=-1, help="Focus results on the nth (0-indexed) directory (default: ALL directories)")
    prs.add_argument("--save", default=None, help="Save figure rather than display (to this path, if given)")
    prs.add_argument("--quantile", type=float, nargs='*', default=0.5, help="Quantile to indicate (per directory, if more than one specified; default: %(default)s)")
    prs.add_argument("--timeout", type=float, default=20.0, help="Virtual timeout to include on graph (default: %(default)s)")
    prs.add_argument("--drop-failures", action="store_true", help="Omit failures from the plot")
    prs.add_argument("--flops-only", action="store_true", help="Drop runtime from the plots")
    prs.add_argument("--monotonic", action="store_true", help="Reorder to have objective monotonically increasing")
    prs.add_argument("--stats", action="store_true", help="Calculate extended stats (text only)")
    prs.add_argument("--no-plots", action="store_true", help="Skip visuals")
    prs.add_argument("--normalize-y", action="store_true", help="Normalize y-axis values")
    prs.add_argument("--smart-names", action="store_true", help="Attempt to make better names for plot legends")
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
    if args.crawl_directory is not None and args.rename is not None:
        no_crawl_rename = "Cannot specify --crawl-directory and --rename at the same time. Use --directory instead"
        raise ValueError(no_crawl_rename)
    elif len(args.rename) > 0 and len(args.rename) != len(args.directory):
        mismatch_length = "Length of --directory and --rename arguments must be the same"
        raise ValueError(mismatch_length)
    elif len(args.rename) == 0:
        args.rename = [None] * len(args.directory)
    if args.highlight >= 0:
        args.directory = [args.directory[args.highlight]]
    if type(args.quantile) is float:
        args.quantile = [args.quantile]
    if len(args.quantile) < len(args.directory):
        args.quantile += [args.quantile[-1]] * (len(args.directory) - len(args.quantile))
    args.quantile = np.asarray(args.quantile)
    args.directory = np.asarray(args.directory)
    args.rename = np.asarray(args.rename)
    if not args.no_resort:
        directory_sort = np.argsort(args.directory)
        args.directory = args.directory[directory_sort]
        args.rename = args.rename[directory_sort]
        args.quantile = args.quantile[directory_sort]
    return args

def load(args):
    frames = []
    names = []
    for dirname, rename in zip(args.directory, args.rename):
        if not os.path.isdir(dirname):
            continue
        try:
            frame = pd.read_csv(os.path.join(dirname, "manager_results.csv"))
        except FileNotFoundError:
            try:
                frame = pd.read_csv(os.path.join(dirname, "results.csv"))
            except FileNotFoundError:
                # If there's only one CSV there to pick, use that one
                csvs = [_ for _ in os.listdir(dirname) if (len(_) > 4 and _[-4:] == b'.csv')]
                if len(csvs) == 1:
                    print(f"Did not find normal CSV name (manager_results.csv or results.csv), but found {csvs[0]}")
                    frame = pd.read_csv(os.path.join(dirname, csvs[0].decode()))
                else:
                    no_results = f"No results CSV file found in {dirname} -- skipping"
                    warnings.warn(no_results)
                    continue
            else:
                print(f"No manager results -- substituting with worker results")
        # Invert flops metric for evaluation
        frame["GFLOPS"] = -1 * frame["FLOPS"]
        # Fetch # workers for data preparation purposes
        max_workers = frame.iloc[0]['libE_workers']
        # Put ORIGINAL index in first
        frame = frame.reset_index(drop=False)
        # Reordering to ensure mathematical correctness
        frame = frame.sort_values(by=['elapsed_sec', 'libE_id', ])
        # Elapsed time is difference between INDIVIDUAL WORKER'S previous timesteps
        prev_worker_time = dict((v, 0) for v in sorted(set(frame['libE_id'].values)))
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
        # Maybe we don't want failures
        if args.drop_failures:
            frame = frame.loc[frame['GFLOPS'] > 0.0]
        # Normalization
        if args.normalize_y:
            frame['GFLOPS'] = (frame['GFLOPS'] - frame['GFLOPS'].min()) / (frame['GFLOPS'].max() - frame['GFLOPS'].min())
        frames.append(frame)
        if rename is None:
            names.append(dirname.rstrip('/').split('/')[-1] + f" ({len(frame)})")
        else:
            names.append(rename + f" ({len(frame)})")
    return frames, names

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
    print(frame[['index','libE_id','elapsed_sec','elapsed_diff','continuous_diff','GFLOPS']])
    print()

def smart_name(name):
    rname = None
    if "GPTune" in name:
        rname = "GPTune"
    elif "Weak_TL" in name:
        rname = "Gaussian Copula"
    elif re.match(r"Theta_[0-9]+n_[0-9]+a", name):
        rname = "Bayesian Optimization"
    matches = re.match(r".*\(([0-9]+)\)", name)
    if rname is not None and matches is not None:
        rname += f" ({matches.groups()[0]} samples)"
    return name if rname is None else rname

def smart_title(names):
    portions = [set(_[:_.rindex(' ')].split("_")) for _ in names]
    common = portions[0]
    for index in range(1,len(portions)):
        common = common.intersection(portions[index])
    common_nodes = int([_[:-1] for _ in common if _.endswith('n')][0])
    common_appsc = int([_[:-1] for _ in common if _.endswith('a') and _ != "Theta"][0])
    return f"heFFTe {common_nodes} Nodes compute {common_appsc}^3 FFT"

def visualizations(frames, names, args):
    fig, ax = plt.subplots()
    max_frame_index = max([max(frame['index']) for frame in frames])
    if not args.flops_only:
        for frame, name in zip(frames, names):
            #newline = sns.lineplot(frame, x='elapsed_sec', y='GFLOPS', estimator=None, marker='.')
            newline = sns.lineplot(frame, x='index', y='elapsed_diff', estimator=None, marker='.', label=name)
        #ax.hlines(0.0, xmin=0, xmax=max([max(frame['elapsed_sec']) for frame in frames]))
        ax.hlines(0.0, xmin=0, xmax=max_frame_index)
        ax.hlines(args.timeout, xmin=0, xmax=max_frame_index)

    flines = []
    for frame, color, name, idx in zip(frames, matplotlib.colors.TABLEAU_COLORS, names, np.arange(len(frames))):
        if args.monotonic:
            frame = frame.sort_values(by=['GFLOPS',]).reset_index(drop=True)
            frame['index'] = frame.index
        fline = sns.lineplot(data=frame, x='index', y='GFLOPS', estimator=None, marker='+', label=name,
                             color=color, markeredgecolor=color, linestyle='--', linewidth=1)
        if args.normalize_y:
            ax.set_ylabel("Normalized GFLOP/s")
        if args.flops_only:
            ax.hlines(frame['GFLOPS'].quantile(args.quantile[idx]), xmin=0, xmax=max_frame_index, color=color)

    leglines = [matplotlib.lines.Line2D([0],[0], color=c, linestyle='--', linewidth=1, marker='+', markeredgecolor=c)
                for f,c in zip(frames, matplotlib.colors.TABLEAU_COLORS)]
    #plt.legend(labels=names, loc='best')
    #print(names)
    ax.hlines(0.0, xmin=0, xmax=max_frame_index, color='black', zorder=-1)
    #plt.legend(leglines, names, loc='best')
    if args.smart_names:
        ax.set_title(smart_title(names))
        names = [smart_name(_) for _ in names]
    plt.legend(leglines, names, loc='center right', title="Data Source (#Samples)")
    if args.title is not None:
        plt.title(args.title)
    if args.save is None:
        print("Showing plots")
        plt.show()
    else:
        print(f"Saving plot to {args.save}")
        plt.savefig(args.save, dpi=400)


if __name__ == '__main__':
    args = parse()
    frames, names = load(args)
    print(names)
    if args.stats:
        for frame in frames:
            observations(frame, args)
    if not args.no_plots:
        visualizations(frames, names, args)

