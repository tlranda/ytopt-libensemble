# Shows log results and scripts used in a directory
# Ends with summary statistics (mean, stddev) of the entire directory versus the average per-log
# (This demonstrates the difference in variability of all evaluations versus repeating a single
# configuration -- to rule out the idea that configuration chances are less impactful on
# performance that system performance variability itself)

import pathlib
import numpy as np

# Must be run relative to the tmp_files directory you want to use
scripts = [_ for _ in pathlib.Path('tmp_files').iterdir() if _.suffix == '.sh']
# No special checks here to see if all files exist, it just assumes they do
logs = dict((k, [k.with_stem(k.stem+end).with_suffix('.log') for end in ['_0','_1','_2']]) for k in scripts)

# Grab the performance from logs (assumes they exist and are not erroneous)
def showperf(f):
	with open(f,'r') as l:
		pline = [_.rstrip().split(' ')[2:] for _ in l.readlines() if _.startswith('Performance')]
	if len(pline) == 1:
		print(f, " ".join(pline[0]))
		return float(pline[0][0])
	else:
		raise ValueError

# Shows the 3rd line (for speed3d_r2c.sh templates, this is effectively the configuration)
def showscript(f):
	with open(f,'r') as l:
		print(f, l.readlines()[2].rstrip())

# Iterate everything
per_key = dict((k,list()) for k in logs.keys())
overall = list()
for key in logs.keys():
	showscript(key)
	for log in logs[key]:
		per_key[key].append(showperf(log))
	overall.append(np.mean(per_key[key]))
	print(overall[-1], np.std(per_key[key]))

# Overall variability
print(np.mean(overall), np.std(overall))
# Variability per config
print(np.mean(np.mean(np.asarray(list(per_key.values())),axis=1)),
      np.mean(np.std(np.asarray(list(per_key.values())),axis=1)))
