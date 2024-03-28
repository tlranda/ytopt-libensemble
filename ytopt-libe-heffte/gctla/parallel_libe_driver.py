import subprocess
import os
import time
import pathlib
import re
import argparse

import pandas as pd, numpy as np

# Local imports
from driver_components.NodeFileManager import NodeFileManager
from driver_components.PythonEval import PythonEval
from driver_components.SleepJobInfo import SleepJobInfo
from driver_components.LibEYtoptJobInfo import YtoptJobInfo
from driver_components.LibEGCJobInfo import GCJobInfo
from driver_components.PretendSubprocess import PretendSubprocess
from driver_components.utils import pretty_function_print

###
# This script allows you to run many libE jobs in parallel without oversubscribing a specific
# total node limit. HOWEVER, take caution that ytopt and libEnsemble logs WILL be a cluttered mess
# between parallel runs / scripts attempting to manipulate the files.
# Newer libEnsemble versions should permit creating the ensemble logs in the ensemble directory,
# and having that directory exist ahead of time to make symlinks etc and fully run the process within
# its ensemble directory (ie: no cross-contamination) but this isn't ready yet.
###

DEBUG_LEVEL=0
# GLOBAL MANAGER FOR NODE LISTS
NodeListMaker = NodeFileManager()

def produce_libE_job(jobType, line, n_nodes, n_ranks, app_scale, manager_handle, args):
    # No less than 1 worker
    # No more than max_workers workers
    # Prefer most workers that can fit in a job
    workers = max(1,min(args.max_workers, args.max_nodes // int(n_nodes)))
    if jobType == 'ytopt':
        template_job = YtoptJobInfo(line, n_nodes, n_ranks, app_scale, workers, args.n_records, manager_handle)
        dirname = "polarisSlingshot11_{n_nodes}n_{app_scale[0]}_{app_scale[1]}_{app_scale[2]}a"
    elif jobType == 'gctla':
        template_job = GCJobInfo(line, n_nodes, n_ranks, app_scale, workers, args.n_records, manager_handle)
        dirname = "polarisSlingshot11_GCTLA_{n_nodes}n_{app_scale[0]}_{app_scale[1]}_{app_scale[2]}a"
    prelaunch_tasks = [
        "mkdir -p "+dirname,
        PythonEval(func=os.chdir, args=(dirname,), debug_level=DEBUG_LEVEL),
        PythonEval(func=NodeListMaker.reserve_nodes, args=("{n_nodes}x{workers}", dirname)),
        PythonEval(func=template_job.mutate_basic_job_string,
                   args=(NodeListMaker.modify_job_string, template_job.basic_job_string, dirname)),
        "ln -s ../wrapper_components/ wrapper_components",
        "ln -s ../libEwrapper.py libEwrapper.py",
        "touch __init__.py"
    ]
    if jobType == 'gctla':
        prelaunch_tasks.append("ln -s ../source_tasks/ source_tasks")
    postlaunch_tasks = [PythonEval(func=os.chdir, args=("..",), debug_level=DEBUG_LEVEL),]
    finalize_tasks = [PythonEval(func=NodeListMaker.free_nodes, args=(dirname,), debug_level=DEBUG_LEVEL),]
    template_job.update_tasks(prelaunch_tasks,
                              postlaunch_tasks,
                              finalize_tasks)
    return template_job

###
# PARSING
###

def build():
    prs = argparse.ArgumentParser()
    prs.add_argument('--description', type=str, required=True, help="File with newline-delimited paths to forge or expand")
    prs.add_argument('--job-type', choices=['sleep','ytopt','gctla'], default='ytopt', help="Identify what kinds of jobs are described by --description (default: %(default)s)")
    prs.add_argument('--max-nodes', type=int, default=None, help="Maximum nodes in parallel (required if PBS_NODEFILE is not set, shrinks to contents of PBS_NODEFILE if --max-nodes indicates more nodes than are listed there)")
    prs.add_argument('--ignore-PBSNODEFILE', action='store_true', help="Disables shrinking job to PBS_NODEFILE's limit (useful for debug; NOT INTENDED FOR NON-DEBUG use) (default: %(default)s)")
    prs.add_argument('--n-records', type=int, required=True, help="Number of records required for each item in description list")
    prs.add_argument('--ranks-per-node', type=int, default=4, help="MPI rank expansion factor (default: %(default)s)")
    prs.add_argument('--max-workers', type=int, default=4, help="Max LibE workers (will be reduced for larger node jobs automatically; default: %(default)s)")
    prs.add_argument('--sleep', type=int, default=60, help="Sleep period (in seconds) before checking to start up more jobs (default: %(default)s)")
    prs.add_argument('--debug-level', choices=['min','max'], default='min', help="Amount of debug output from this program (default: %(default)s)")
    prs.add_argument('--demo', action='store_true', help="Operate in demo mode (only echo jobs commands)")
    prs.add_argument('--pretend-polls', action='store_true', help="Demo mode pretends to poll jobs a small random number of times rather than just once per job (default: %(default)s)")
    return prs

def parse(args=None, prs=None):
    global DEBUG_LEVEL
    if prs is None:
        prs = build()
    if args is None:
        args = prs.parse_args()

    # Not all code has direct access to args, just use this global value
    if args.debug_level == 'max':
        DEBUG_LEVEL = 1

    # Max nodes may be set on the command line, but if PBS_NODEFILE is set in the environment,
    # the node manager will only be capable of allocating those nodes. Therefore, --max-nodes
    # can use any subset (including proper subset) of that many nodes, but cannot exceed it
    if not args.ignore_PBSNODEFILE and 'PBS_NODEFILE' in os.environ:
        with open(os.environ["PBS_NODEFILE"], "r") as f:
            nodefile_nodes = len(f.readlines())
            print(f"PBS_NODEFILE, '{os.environ['PBS_NODEFILE']}' indicates {nodefile_nodes} nodes")
        if args.max_nodes is None:
            args.max_nodes = nodefile_nodes
        elif args.max_nodes > nodefile_nodes:
            print(f"REDUCING --max-nodes={args.max_nodes} to nodefile's limit of {nodefile_nodes}")
            args.max_nodes = nodefile_nodes
        elif args.max_nodes < nodefile_nodes:
            NodeListMaker.limit_nodes(args.max_nodes)
    else:
        # Have to artificially update NodeListMaker or it will reject you on the spot
        needed_to_add_nodes = args.max_nodes - len(NodeListMaker.node_list)
        NodeListMaker.node_list = list(NodeListMaker.node_list)
        for fake_node in range(needed_to_add_nodes):
            NodeListMaker.node_list.append(f"FAKE_NODE_{fake_node}")
        NodeListMaker.node_list = np.asarray(NodeListMaker.node_list)
        NodeListMaker.n_nodes = NodeListMaker.node_list.shape[0]
        NodeListMaker.allocated = np.full((NodeListMaker.n_nodes), False, dtype=bool)

    args.jobs = []
    rejected_info = []
    with open(args.description, 'r') as spec:
        for line_idx, line in enumerate(spec.readlines()):
            line = line.lstrip().rstrip()
            if line == '' or line.startswith('#'):
                continue
            if args.job_type == 'sleep':
                args.jobs.append(
                    SleepJobInfo(line, int(line))
                )
            elif args.job_type == 'ytopt' or args.job_type == 'gctla':
                search_path = pathlib.Path(line)
                dirname = str(search_path.stem)
                matches = re.match(r"(.*)_([0-9]+)n_([0-9]+)_([0-9]+)_([0-9]+)a", dirname).groups(0)
                system_and_identifier, n_nodes, *app_scale = matches
                # n_nodes convert to int so we can ensure job oversubscription doesn't occur
                int_nodes = int(n_nodes)
                # Reject if never runnable
                if int_nodes > args.max_nodes:
                    # Add one so it lines up with 1-based line counting used by most editors
                    rejected_info.append((int_nodes, line_idx+1, line, f"(Requested nodes {int_nodes} > Max nodes {args.max_nodes})"))
                    continue
                n_ranks = int_nodes * args.ranks_per_node
                # Check if there are any existing results to extend
                manager_handle = search_path.joinpath('manager_results.csv')
                if manager_handle.exists():
                    existing_records = pd.read_csv(manager_handle)
                    # Jobs should only be created when record count is insufficient
                    if len(existing_records) < args.n_records:
                        args.jobs.append(produce_libE_job(args.job_type, line, n_nodes, n_ranks, app_scale, manager_handle, args))
                    else:
                        # Add one so it lines up with 1-based line counting used by most editors
                        rejected_info.append((int_nodes, line_idx+1, line, f"(Requested {args.n_records} records already satisfied by existing file with {len(existing_records)} records)"))
                        continue
                else:
                    args.jobs.append(produce_libE_job(args.job_type, line, n_nodes, n_ranks, app_scale, manager_handle, args))
            if DEBUG_LEVEL > 0:
                print(f"Job description: {line} --> Job: {args.jobs[-1].shortName().rstrip()}")
    for (int_nodes, line_idx, line, reason) in rejected_info:
        print(f"!! --REJECT-- spec {reason} in {args.description}:{line_idx} \"{line}\"")
    if DEBUG_LEVEL > 0:
        described_classes = set()
        for job in args.jobs:
            jobclass = job.__class__.__name__
            if jobclass not in described_classes:
                described_classes.add(jobclass)
                print(f"{jobclass} Jobs have the following responsibilities:")
                if len(job.prelaunch_tasks) > 0:
                    print("\tPre-Launch Tasks:",[pretty_function_print(_.func) if isinstance(_,PythonEval) else str(_) for _ in job.prelaunch_tasks])
                if len(job.postlaunch_tasks) > 0:
                    print("\tPost-Launch Tasks:",[pretty_function_print(_.func) if isinstance(_,PythonEval) else str(_) for _ in job.postlaunch_tasks])
                if len(job.finalize_tasks) > 0:
                    print("\tFinalize Tasks:",[pretty_function_print(_.func) if isinstance(_,PythonEval) else str(_) for _ in job.finalize_tasks])
    return args

def execute_job(job_obj, args):
    print(f"Execute job: {job_obj.specName()}")
    # Pre job
    job_obj.prelaunch()
    # Actual job
    command_split = job_obj.format()
    print(f"Launch job: {' '.join(command_split)}")
    if args.demo:
        print("!! DEMO -- no job launch")
        new_sub = PretendSubprocess(command_split, None if args.pretend_polls else 1)
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
            n_nodes = job.nodes
            if n_nodes + nodes_in_flight > args.max_nodes:
                if DEBUG_LEVEL > 0:
                    print(f"Job requests {n_nodes} nodes exceed max in flight -- {job.specName()}")
                unused_queue.append(args.jobs[idx])
                continue
            new_sub = execute_job(job, args)
            subprocess_queue[new_sub] = {'spec': job.specName(), 'nodes': job.nodes, 'job': job}
            nodes_in_flight += n_nodes
        args.jobs = unused_queue
        if DEBUG_LEVEL > 0:
            print("Running jobs:")
            print("\t"+"\n\t".join([f"({spq_dict['nodes']} nodes): {spq_dict['spec']}" for spq_dict in subprocess_queue.values()]))
        # VERY inactive poll between jobs -- maximize resources for everything else we don't need these to pop off immediately
        if len(subprocess_queue) > 0:
            if DEBUG_LEVEL > 0:
                print(f"Sleep for {args.sleep} seconds")
            time.sleep(args.sleep)
    print("ALL jobs done")

if __name__ == '__main__':
    main()

