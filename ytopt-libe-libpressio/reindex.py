import pandas as pd
import numpy as np

c1 = pd.read_csv("ensemble_Polaris_2r_mgardgpu_fa53488f/results.csv")
print(f"c1 loads {len(c1)} records")
c2 = pd.read_csv("ensemble_Polaris_2r_mgardgpu_974494b7/results.csv")
print(f"c2 loads {len(c2)} records")
# Prepare for elapsed_time updates
worker_times = {}
c1_diffs = []
for time, worker in zip(c1['elapsed_sec'], c1['libE_id']):
    if worker not in worker_times:
        worker_times[worker] = time
        c1_diffs.append(time)
    else:
        c1_diffs.append(time-worker_times[worker])
        worker_times[worker] = time
worker_times = {}
c2_diffs = []
for time, worker in zip(c2['elapsed_sec'], c2['libE_id']):
    if worker not in worker_times:
        worker_times[worker] = time
        c2_diffs.append(time)
    else:
        c2_diffs.append(time-worker_times[worker])
        worker_times[worker] = time
c1.insert(c1.columns.tolist().index('elapsed_sec')+1, 'duration', c1_diffs)
c2.insert(c2.columns.tolist().index('elapsed_sec')+1, 'duration', c2_diffs)

# Prepare for merge identification
subset = ['p0','p1','p2']
c1_v = c1[subset].to_numpy()
c2_v = c2[subset].to_numpy()

translate_c1_c2 = np.asarray([None] * len(c1))
translate_c2_c1 = np.asarray([None] * len(c2))

# Search
for src_idx in range(len(c1)):
    translation = np.where((tuple(c1_v[src_idx]) == c2_v).sum(1) == len(subset))[0]
    if len(translation) > 0:
        translate_c1_c2[src_idx] = translation[0]
for src_idx in range(len(c2)):
    translation = np.where((tuple(c2_v[src_idx]) == c1_v).sum(1) == len(subset))[0]
    if len(translation) > 0:
        translate_c2_c1[src_idx] = translation[0]
# Verify
for src_idx in range(len(c1)):
    if translate_c1_c2[src_idx] is not None:
        if translate_c2_c1[translate_c1_c2[src_idx]] != src_idx:
            print(f"Broken translation: c1->c2 via {src_idx}; c2->c1 {translate_c2_c1[translate_c1_c2[src_idx]]}")
for src_idx in range(len(c2)):
    if translate_c2_c1[src_idx] is not None:
        if translate_c1_c2[translate_c2_c1[src_idx]] != src_idx:
            print(f"Broken translation: c2->c1 via {src_idx}; c1->c2 {translate_c1_c2[translate_c2_c1[src_idx]]}")
# Condense
uniq_c1 = [i for i,v in enumerate(translate_c1_c2) if v is None]
uniq_c2 = [i for i,v in enumerate(translate_c2_c1) if v is None]
common = [i for i,v in enumerate(translate_c1_c2) if v is not None]
total = len(uniq_c1)+len(uniq_c2)+len(common)
pd.set_option('display.max_rows', total)
print(f"Total records: {total}")
u1 = c1.iloc[uniq_c1]
u2 = c2.iloc[uniq_c2]
print("C1 unique")
print(u1)
print("C2 unique")
print(u2)
print("Common")
cm = c1.iloc[common].copy()
cmrg = cm.copy()
translate = ['FOM', 'elapsed_sec', 'duration']
cm_translate = c2.iloc[translate_c1_c2[common]][translate]
for field in translate:
    cm.insert(cm.columns.tolist().index(field)+1, f'{field}_2', cm_translate[field].tolist())
# Common Merge
# Prefers minimal FOM to replace error with objective when possible, else minimal duration
cmrg_fom, cmrg_elapse, cmrg_duration = [], [], []
for record in cm.index:
    cand_fom = cm.loc[record,['FOM','FOM_2']]
    if len(set(cand_fom)) == 1:
        cand_dur = cm.loc[record,['duration','duration_2']]
        pref = np.argmin(cand_dur)
    else:
        pref = np.argmin(cand_fom)
    cmrg_fom.extend(cm.loc[record,['FOM' if pref == 0 else 'FOM_2']].values)
    cmrg_elapse.extend(cm.loc[record,['elapsed_sec' if pref == 0 else 'elapsed_sec_2']].values)
    cmrg_duration.extend(cm.loc[record,['duration' if pref == 0 else 'duration_2']].values)
cmrg['FOM'] = cmrg_fom
cmrg['elapsed_sec'] = cmrg_elapse
cmrg['duration'] = cmrg_duration
print(cmrg)
#print(cm[cm.columns.tolist()[:10]])

# Full Merge
# Order on elapsed sec
merged = pd.concat((u1,u2,cmrg)).sort_values(by='elapsed_sec').reset_index(drop=True)
merged.to_csv('manager_results.csv')

