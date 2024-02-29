import subprocess
import os
import shlex
import inspect
import time
import pathlib
import re
import argparse

import pandas as pd, numpy as np

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
        SWD_Ignore = set(['self','args','kwargs','SWD_Ignore'])
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

class NodeFileManager():
    def __init__(self):
        self.node_list = []
        if 'PBS_NODEFILE' in os.environ:
            with open(os.environ['PBS_NODEFILE'],'r') as nodefile:
                self.node_list = [_.rstrip() for _ in nodefile.readlines()]
        self.node_list = np.asarray(self.node_list)
        self.n_nodes = self.node_list.shape[0]
        self.allocated = np.full((self.n_nodes), False, dtype=bool)
        self.tags = dict()

    def reserve_nodes(self, n_nodes, tag):
        if type(n_nodes) is str:
            n_nodes = int(n_nodes)
        # Special case : No nodes are ever reserved
        if self.n_nodes == 0:
            self.tags[tag] = {'file': None, 'indices': []}
            return

        assert n_nodes <= (~self.allocated).sum(), f"Allocation request for {n_nodes} exceeds free capacity: {(~self.allocated).sum()}/{self.allocated.shape[0]}"
        # Select unallocated nodes to reserve
        allocation = np.nonzero(~self.allocated)[0][:n_nodes]
        self.allocated[allocation] = True
        # Create uniquely named allocation file
        allocation_name = pathlib.Path("nodelist.txt")
        i = 1
        while allocation_name.exists():
            allocation_name = allocation_name.with_stem(f"nodelist_{i}")
            i += 1
        allocation_name.touch()
        with open(allocation_name,'w') as nodefile:
            nodefile.write("\n".join(self.node_list[allocation]))
        # Save information for future reference under the tag
        self.tags[tag] = {'file': allocation_name,
                          'indices': allocation}

    def free_nodes(self, tag):
        assert tag in self.tags, f"Tag '{tag}' not found in list of known tags: {sorted(self.tags.keys())}"
        # De-allocate nodes and free the tag
        # Because we set indicies == empty list when not managing nodes, this operation doesn't need
        # to be guarded against that case
        self.allocated[self.tags[tag]['indices']] = False
        del self.tags[tag]

    def modify_job_string(self, job_string, tag):
        assert tag in self.tags, f"Tag '{tag}' not found in list of known tags: {sorted(self.tags.keys())}"
        if len(self.tags[tag]['indices']) > 0:
            job_string += f" --node-list-file {self.tags[tag]['file']}"
        return job_string
# GLOBAL MANAGER FOR NODE LISTS
NodeListMaker = NodeFileManager()

def NoOp(*args,**kwargs):
    pass

# Useful for pre-launch / post-launch commands that cannot be executed by a subshell
class PythonEval():
    def __init__(self, func=NoOp, args=(), kwargs={}, **other_attrs):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        for (attr, value) in other_attrs.items():
            setattr(self, attr, value)
    def __str__(self):
        return f"Calls {self.func} with:"+\
                "\n\targs:\n\t\t"+"\n\t\t".join([str(_) for _ in self.args])+\
                "\n\tkwargs:\n\t\t"+"\n\t\t".join([f"Key: '{k}'; Value: '{v}'" for (k,v) in self.kwargs.items()])
    def make_args(self, job):
        new_args = []
        for attr in self.args:
            if hasattr(attr, 'format') and callable(getattr(attr,'format')):
                attr = attr.format(**job.__dict__)
            new_args.append(attr)
        new_kwargs = {}
        for key, val in self.kwargs.items():
            if hasattr(val, 'format') and callable(getattr(val,'format')):
                val = val.format(**job.__dict__)
            new_kwargs[key] = val
        return new_args, new_kwargs
    def __call__(self, job):
        new_args, new_kwargs = self.make_args(job)
        self.func(*new_args, **new_kwargs)

# Format the shell command based on own attributes and track # of occupied nodes by the job
class JobInfo(SetWhenDefined):
    basic_job_string = ""
    prelaunch_tasks = []
    postlaunch_tasks = []
    finalize_tasks = []
    def __str__(self):
        return self.basic_job_string.format(**self.__dict__)
    def shortName(self):
        except_spec = dict((k,v) for (k,v) in self.__dict__.items() if not k.startswith('_') and k not in ['spec'])
        return str(except_spec)
    def specName(self):
        return self.spec
    def nodes(self):
        pass
    def process_cmd_queue(self, queue, identifier):
        n_queued = len(queue)
        if n_queued == 0:
            print(f"{identifier} -- Nothing to do!")
        for idx, cmd in enumerate(queue):
            if type(cmd) is str:
                command_split = shlex.split(cmd.format(**self.__dict__))
                print(f"{identifier} command {idx+1}/{n_queued}: {command_split}")
                new_sub = subprocess.Popen(command_split)
                retcode = new_sub.wait()
                if retcode != 0:
                    raise ValueError("Command failed")
            elif isinstance(cmd, PythonEval):
                print(f"{identifier} Python command {idx+1}/{n_queued}: {cmd}")
                cmd(self)
    def prelaunch(self):
        self.process_cmd_queue(self.prelaunch_tasks, "Pre-launch")
    def format(self):
        return shlex.split(str(self))
    def postlaunch(self):
        self.process_cmd_queue(self.postlaunch_tasks, "Post-launch")
    def finalize(self):
        self.process_cmd_queue(self.finalize_tasks, "Finalize")
    def update_tasks(self, prelaunch, postlaunch, finalize):
        if prelaunch is not None:
            self.prelaunch_tasks = prelaunch
        if postlaunch is not None:
            self.postlaunch_tasks = postlaunch
        if finalize is not None:
            self.finalize_tasks = finalize

class MutableJobInfo(JobInfo):
    def mutate_basic_job_string(self, _callable, *args, **kwargs):
        self.basic_job_string = _callable(*args, **kwargs)

# These tasks will be placed into the LibeJobInfo instances below, but have to wait for class instances to finalize
libE_dirname = "polarisSlingshot11_{n_nodes}n_{app_scale[0]}_{app_scale[1]}_{app_scale[2]}a"
libE_prelaunch_tasks = ["mkdir -p "+libE_dirname,
                   PythonEval(func=os.chdir,
                              args=(libE_dirname,)),
                   PythonEval(func=NodeListMaker.reserve_nodes,
                              args=("{n_nodes}", libE_dirname)),
                   None, # Will be replaced when instantiated
                   "ln -s ../wrapper_components/ wrapper_components",
                   "ln -s ../libEwrapper.py libEwrapper.py",
                   "touch __init__.py"]
libE_postlaunch_tasks = [PythonEval(func=os.chdir, args=("..",)),]
libE_finalize_tasks = [PythonEval(func=NodeListMaker.free_nodes, args=(libE_dirname,)),]
class LibeJobInfo(MutableJobInfo):
    basic_job_string = "python3 libEwrapper.py "+\
                        "--system polaris "+\
                        "--mpi-ranks {n_ranks} "+\
                        "--gpu-override 4 --gpu-enabled "+\
                        "--application-x {app_scale[0]} --application-y {app_scale[1]} --application-z {app_scale[2]} "+\
                        "--ensemble-workers {workers} --max-evals {n_records} "+\
                        "--configure-environment craympi "+\
                        "--machine-identifier polarisSlingshot11 "+\
                        "--ens-dir-path polarisSlingshot11_{n_nodes}n_{app_scale[0]}_{app_scale[1]}_{app_scale[2]}a "+\
                        "--launch-job --display-results"
    """
    # DEPRECATED Theta system : Extending existing results
    basic_job_string = "python3 libEwrapper.py "+\
                       "--system theta "+\
                       "--mpi-ranks {n_ranks} "+\
                       "--cpu-override 256 --cpu-ranks-per-node 64 "+\
                       "--application-scale {app_scale} "+\
                       "--ensemble-workers {workers} --max-evals {n_records} "+\
                       "--configure-environment craympi "+\
                       "--machine-identifier theta-knl "+\
                       "--ens-dir-path Theta_Extend_{n_nodes}n_{app_scale}a "+\
                       "--resume {resume} --launch-job --display-results"
    """
    # Polaris system : Collecting new datasets
    def __init__(self, spec, n_nodes, n_ranks, app_scale, workers, n_records, resume):
        self.overrideSelfAttrs()
    def nodes(self):
        return int(self.n_nodes) * self.workers

class SleepJobInfo(JobInfo):
    basic_job_string = "sleep {duration}"
    def __init__(self, spec, duration):
        self.overrideSelfAttrs()
    def nodes(self):
        return self.duration

# This class emulates a polling object for debug purposes (run real LibeJobs without executing them)
class pretend_subprocess:
    def __init__(self, command, max_polls = None):
        self.poll_count = 0
        self.max_polls = max_polls if max_polls is not None else np.random.randint(1,4)
        self.command = command
        self.returncode = 0
    def poll(self):
        self.poll_count += 1
        if self.poll_count >= self.max_polls:
            return 0
        else:
            return None
    def __str__(self):
        return f"<Pretend Popen max_polls: {self.max_polls} returncode: {self.returncode} args: {self.command}>"


###
# PARSING
###

def build():
    prs = argparse.ArgumentParser()
    prs.add_argument('--description', type=str, required=True, help="File with newline-delimited paths to forge or expand")
    prs.add_argument('--job-type', choices=['sleep','libE'], default='libE', help="Identify what kinds of jobs are described by --description (default: %(default)s)")
    prs.add_argument('--max-nodes', type=int, required=True, help="Maximum nodes in parallel")
    prs.add_argument('--n-records', type=int, required=True, help="Number of records required for each item in description list")
    prs.add_argument('--ranks-per-node', type=int, default=4, help="MPI rank expansion factor (default: %(default)s)")
    prs.add_argument('--max-workers', type=int, default=4, help="Max LibE workers (will be reduced for larger node jobs automatically; default: %(default)s)")
    prs.add_argument('--sleep', type=int, default=60, help="Sleep period (in seconds) before checking to start up more jobs (default: %(default)s)")
    prs.add_argument('--demo', action='store_true', help="Operate in demo mode (only echo jobs commands)")
    prs.add_argument('--pretend-polls', action='store_true', help="Demo mode pretends to poll jobs a small random number of times rather than just once per job (default: %(default)s)")
    return prs

def parse(args=None, prs=None):
    if prs is None:
        prs = build()
    if args is None:
        args = prs.parse_args()
    args.jobs = []
    with open(args.description, 'r') as spec:
        for line_idx, line in enumerate(spec.readlines()):
            line = line.lstrip().rstrip()
            if line == '' or line.startswith('#'):
                continue
            if args.job_type == 'sleep':
                args.jobs.append(
                    SleepJobInfo(line, int(line))
                )
            elif args.job_type == 'libE':
                search_path = pathlib.Path(line)
                dirname = str(search_path.stem)
                matches = re.match(r"(.*)_([0-9]+)n_([0-9]+)_([0-9]+)_([0-9]+)a", dirname).groups(0)
                system, n_nodes, *app_scale = matches
                # n_nodes convert to int so we can ensure job oversubscription doesn't occur
                int_nodes = int(n_nodes)
                # Reject if never runnable
                if int_nodes > args.max_nodes:
                    print(f"REJECT spec (Requested nodes {int_nodes} > Max nodes {args.max_nodes}) in {args.description}:{line_idx} \"{line}\"")
                    continue
                n_ranks = int_nodes * args.ranks_per_node
                # Check if there are any existing results to extend
                manager_handle = search_path.joinpath('manager_results.csv')
                if manager_handle.exists():
                    existing_records = pd.read_csv(manager_handle)
                    # Jobs should only be created when record count is insufficient
                    if len(existing_records) < args.n_records:
                        workers = 1
                        while workers < args.max_workers and n_nodes * workers < args.max_nodes:
                            workers += 1
                        template_job = LibeJobInfo(line, n_nodes, n_ranks, app_scale, workers, args.n_records, manager_handle)
                        # Need to adjust one prelaunch task here now that we're instantiated
                        libE_instanced_prelaunch_tasks = []
                        for task in libE_prelaunch_tasks:
                            if task is not None:
                                libE_instanced_prelaunch_tasks.append(task)
                            else:
                                libE_instanced_prelaunch_tasks.append(
                                  PythonEval(func=template_job.mutate_basic_job_string,
                                             args=(NodeListMaker.modify_job_string, template_job.basic_job_string, libE_dirname),
                                            )
                                )
                        template_job.update_tasks(libE_instanced_prelaunch_tasks,
                                                  libE_postlaunch_tasks,
                                                  libE_finalize_tasks)
                        args.jobs.append(template_job)
                    else:
                        continue
                else:
                    workers = 1
                    while workers < args.max_workers and int_nodes * workers < args.max_nodes:
                        workers += 1
                    template_job = LibeJobInfo(line, n_nodes, n_ranks, app_scale, workers, args.n_records, manager_handle)
                    # Need to adjust one prelaunch task here now that we're instantiated
                    libE_instanced_prelaunch_tasks = []
                    for task in libE_prelaunch_tasks:
                        if task is not None:
                            libE_instanced_prelaunch_tasks.append(task)
                        else:
                            libE_instanced_prelaunch_tasks.append(
                              PythonEval(func=template_job.mutate_basic_job_string,
                                         args=(NodeListMaker.modify_job_string, template_job.basic_job_string, libE_dirname),
                                        )
                            )
                    template_job.update_tasks(libE_instanced_prelaunch_tasks,
                                              libE_postlaunch_tasks,
                                              libE_finalize_tasks)
                    args.jobs.append(template_job)
            print(f"Job description: {line} --> Job: {args.jobs[-1].shortName().rstrip()}")
    return args

def execute_job(job_obj, args):
    print(f"Execute job: {job_obj.specName()}")
    # Pre job
    job_obj.prelaunch()
    # Actual job
    command_split = job_obj.format()
    print(f"Launch job: {' '.join(command_split)}")
    if args.demo:
        print("DEMO -- no job launch")
        new_sub = pretend_subprocess(command_split, None if args.pretend_polls else 1)
    else:
        fout = open("parallel_job.output","w")
        ferr = open("parallel_job.error","w")
        new_sub = subprocess.Popen(command_split, stdout=fout, stderr=ferr)
    # Post launch
    job_obj.postlaunch()
    # FINALIZE must be called when the process actually returns
    return new_sub

def main(args=None):
    args = parse(args)
    nodes_in_flight = 0
    subprocess_queue = {}
    print("Initial job queue:")
    print("\t"+"\n\t".join([_.specName() for _ in args.jobs[:-1]])+"\n\t"+args.jobs[-1].specName())
    while len(args.jobs) > 0 or len(subprocess_queue.keys()) > 0:
        print(f"Job queue length: {len(args.jobs)}")
        print(f"Check on {len(subprocess_queue.keys())} jobs")
        # Check for jobs that have finished
        returned = [_ for _ in subprocess_queue.keys() if _.poll() is not None]
        for release in returned:
            print(f"Reclaim job {subprocess_queue[release]['spec']} with {subprocess_queue[release]['nodes']} nodes (return code: {release.returncode})")
            nodes_in_flight -= subprocess_queue[release]['nodes']
            # Clean up any final operations needed for this job
            subprocess_queue[release]['job'].finalize()
            del subprocess_queue[release]
        unused_queue = []
        for idx, job in enumerate(args.jobs):
            n_nodes = job.nodes()
            if n_nodes + nodes_in_flight > args.max_nodes:
                print(f"Job requests {n_nodes} nodes exceed max in flight -- {job.specName()}")
                unused_queue.append(args.jobs[idx])
                continue
            new_sub = execute_job(job, args)
            subprocess_queue[new_sub] = {'spec': job.specName(), 'nodes': job.nodes(), 'job': job}
            nodes_in_flight += n_nodes
        args.jobs = unused_queue
        print("Running jobs:")
        print("\t"+"\n\t".join([f"({spq_dict['nodes']} nodes): {spq_dict['spec']}" for spq_dict in subprocess_queue.values()]))
        # VERY inactive poll between jobs -- maximize resources for everything else we don't need these to pop off immediately
        if len(subprocess_queue) > 0:
            print(f"Sleep for {args.sleep} seconds")
            time.sleep(args.sleep)
    print("ALL jobs done")

if __name__ == '__main__':
    main()

