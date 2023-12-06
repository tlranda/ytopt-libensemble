import subprocess
import shlex
import inspect
import time
import pathlib
import pandas as pd, numpy as np
import argparse

###
# This script allows you to run many libE jobs in parallel without oversubscribing a specific
# total node limit. HOWEVER, take caution that ytopt and libEnsemble logs WILL be a cluttered mess
# between parallel runs / scripts attempting to manipulate the files.
# Newer libEnsemble versions should permit creating the ensemble logs in the ensemble directory,
# and having that directory exist ahead of time to make symlinks etc and fully run the process within
# its ensemble directory (ie: no cross-contamination) but this isn't ready yet.
###


###
# CLASSES
###

# Convenience class that automatically sets class attributes from local space
class SetWhenDefined():
    def overrideSelfAttrs(self):
        SWD_Ignore = set(['self', 'args','kwargs', 'SWD_Ignore'])
        frame = inspect.currentframe().f_back
        flocals = frame.f_locals # Parent stack function local variables
        fcode = frame.f_code # Code object for parent stack function
        # LOCALS
        values = dict((k,v) for (k,v) in flocals.items() if k not in SWD_Ignore)
        # VARARGS
        if 'args' in flocals.keys() and len(flocals['args']) > 0:
            values.update({'varargs': flocals['args']})
        # KWARGS
        if 'kwargs' in flocals.keys():
            values.update(dict((k,v) for (k,v) in flocals['kwargs'].items() if k not in SWD_Ignore))
        # Get names of all arguments from your __init__ method
        specified_values = fcode.co_varnames[:fcode.co_argcount]
        override = set(specified_values).difference(SWD_Ignore)
        for attrname in override:
            # When the current value is None but there's a class default, pick the default
            if values[attrname] is None and hasattr(self, attrname):
                values[attrname] = getattr(self, attrname)
        # Apply values to attributes of this instance
        for (k,v) in values.items():
            setattr(self, k, v)

# Format the shell command based on own attributes and track # of occupied nodes by the job
class JobInfo(SetWhenDefined):
    basic_job_string = ""
    def __repr__(self):
        return self.basic_job_string.format(**self.__dict__)
    def shortName(self):
        return str(self.__dict__)
    def format(self):
        return shlex.split(str(self))
    def nodes(self):
        pass

class LibeJobInfo(JobInfo):
    basic_job_string = "python3 libEwrapper.py --mpi-ranks {n_ranks} --worker-timeout 300 "+\
                       "--application-scale {app_scale} --cpu-override 256 --cpu-ranks-per-node 64 "+\
                       "--ensemble-workers {workers} --max-evals {n_records} "+\
                       "--configure-environment craympi --machine-identifier theta-knl "+\
                       "--system theta --ens-dir-path Theta_Extend_{n_nodes}n_{app_scale}a "+\
                       "--resume {resume} --launch-job --display-results"
    def __init__(self, n_nodes, n_ranks, app_scale, workers, n_records, resume):
        self.overrideSelfAttrs()
    def nodes(self):
        return self.n_nodes * self.workers

class SleepJobInfo(JobInfo):
    basic_job_string = "sleep {duration}"
    def __init__(self, duration):
        self.overrideSelfAttrs()
    def nodes(self):
        return self.duration

# This class emulates a polling object for debug purposes (run real LibeJobs without executing them)
class pretend_subprocess:
    def __init__(self, command, max_polls = None):
        self.poll_count = 0
        self.max_polls = max_polls if max_polls is not None else np.random.randint(1,4)
        self.command = command
    def poll(self):
        self.poll_count += 1
        if self.poll_count >= self.max_polls:
            return 0
        else:
            return None
    def __repr__(self):
        return f"<Pretend Popen: returncode: 0 args: {self.command}>"


###
# PARSING
###

def build():
    prs = argparse.ArgumentParser()
    prs.add_argument('--max-nodes', type=int, required=True, help="Maximum nodes in parallel")
    prs.add_argument('--searched', type=str, nargs='+', required=True, help="Directories to include for expansion")
    prs.add_argument('--n-records', type=int, required=True, help="Number of records required for each directory")
    prs.add_argument('--ranks-per-node', type=int, default=64, help="MPI rank expansion factor (default: %(default)s)")
    prs.add_argument('--max-workers', type=int, default=4, help="Max LibE workers (will be reduced for larger node jobs automatically; default: %(default)s)")
    prs.add_argument('--sleep', type=int, default=60, help="Sleep period (in seconds) before checking to start up more jobs (default: %(default)s)")
    prs.add_argument('--demo', action='store_true', help="Operate in demo mode (jobs are `sleep` commands)")
    return prs

def parse(args=None, prs=None):
    if prs is None:
        prs = build()
    if args is None:
        args = prs.parse_args()
    # For demo purposes, `args.searched` correspond to integer times to sleep in a queue
    if args.demo:
        args.jobs = [SleepJobInfo(int(job)) for job in sorted(args.searched)]
    else:
        args.jobs = []
        for search in sorted(args.searched):
            search_path = pathlib.Path(search)
            dirname = str(search_path.stem)
            # Format: Theta_####n_####a
            system, n_nodes, app_scale = dirname.split('_')
            n_nodes = int(n_nodes[:-1])
            n_ranks = n_nodes * args.ranks_per_node
            app_scale = int(app_scale[:-1])
            manager_handle = search_path.joinpath('manager_results.csv')
            existing_records = pd.read_csv(manager_handle)
            # Jobs should only be created when record count is insufficient
            if len(existing_records) < args.n_records:
                workers = 1
                while workers < args.max_workers and n_nodes * workers < args.max_nodes:
                    workers += 1
                args.jobs.append(
                    LibeJobInfo(n_nodes, n_ranks, app_scale, workers, args.n_records, manager_handle)
                )
                print(f"{search_path} becomes job {args.jobs[-1].shortName()}")
            else:
                print(f"{search_path} satisfies requirement for {args.n_records} records with {len(existing_records)}")
    return args

def main(args=None):
    args = parse(args)
    nodes_in_flight = 0
    subprocess_queue = {}
    while len(args.jobs) > 0 or len(subprocess_queue.keys()) > 0:
        print(f"Remaining job queue: {[_.shortName() for _ in args.jobs]}")
        print(f"Check on {len(subprocess_queue.keys())} jobs")
        # Check for jobs that have finished
        returned = [_ for _ in subprocess_queue if _.poll() is not None]
        print(returned)
        for release in returned:
            print(f"Reclaim job {release} with {subprocess_queue[release]} nodes")
            nodes_in_flight -= subprocess_queue[release]
            del subprocess_queue[release]
        unused_queue = []
        for idx, job in enumerate(args.jobs):
            n_nodes = job.nodes()
            if n_nodes + nodes_in_flight > args.max_nodes:
                print(f"Job requests {n_nodes} nodes exceed max in flight -- {job.shortName()}")
                unused_queue.extend(args.jobs[idx:])
                break
            command_split = job.format()
            print(f"Launch job: {' '.join(command_split)}")
            if args.demo:
                new_sub = subprocess.Popen(command_split)
            else:
                # DEBUG: Don't run a real job, but this object represents key functionality for equivalence
                #new_sub = pretend_subprocess(command_split)
                new_sub = subprocess.Popen(command_split)
            subprocess_queue[new_sub] = job.nodes()
            nodes_in_flight += n_nodes
        args.jobs = unused_queue
        # VERY inactive poll between jobs -- maximize resources for everything else we don't need these to pop off immediately
        if len(subprocess_queue) > 0:
            print(f"Sleep for {args.sleep} seconds")
            time.sleep(args.sleep)
    print("ALL jobs done")

if __name__ == '__main__':
    main()

