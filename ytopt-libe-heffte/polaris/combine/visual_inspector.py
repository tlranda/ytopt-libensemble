import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import os

# Should be changed at some point
managed_files = sorted([_+"/manager_results.csv" for _ in os.listdir() if _.startswith('ensemble_ThetaLibEScaling')])
names = [os.path.dirname(_)[len("ensemble_ThetaLibEScaling_"):] for _ in managed_files]
nice_names = [f"{_.split('_')[0]} LibE Worker" for _ in names]

timeout = 20.0
#index = 3
#managed_files = [managed_files[index]]
#nice_names = [nice_names[index]]
print(managed_files, nice_names, timeout)

frames = []
for file, name in zip(managed_files, nice_names):
    frame = pd.read_csv(file)
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
    timed_out = dict((v+1, np.where(frame[frame['libE_id'] == v+2]['elapsed_diff'] >= timeout)[0].tolist()) for v in range(max_workers))
    timeouts = sum(map(len, timed_out.values()))
    if timeouts == 0:
        print(f"Proposed timeout=={timeout} would not clip any evaluations")
    else:
        print(f"With timeout=={timeout}, {timeouts} evaluations would be removed (per-worker evaluation # shown):")
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
    frames.append(frame)

fig, ax = plt.subplots()
lines = []
mpl_lines = []
for frame in frames:
    #newline = sns.lineplot(frame, x='elapsed_sec', y='FLOPS', estimator=None, marker='.')
    newline = sns.lineplot(frame, x='index', y='elapsed_diff', estimator=None, marker='.')
    lines.append(newline)
    mpl_lines.extend(newline.lines)
#ax.hlines(0.0, xmin=0, xmax=max([max(frame['elapsed_sec']) for frame in frames]))
ax.hlines(0.0, xmin=0, xmax=max([max(frame['index']) for frame in frames]))
ax.hlines(20.0, xmin=0, xmax=max([max(frame['index']) for frame in frames]))

for frame, color in zip(frames, matplotlib.colors.TABLEAU_COLORS):
    sns.lineplot(frame, x='index', y='FLOPS', estimator=None, marker='+',
                 color=color, markeredgecolor=color, linestyle='--', linewidth=1)

plt.legend(labels=nice_names, loc='best')
plt.show()

