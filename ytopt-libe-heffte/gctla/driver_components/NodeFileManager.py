import os
import pathlib
from math import prod

import numpy as np

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
        self.limited = False

    def __str__(self):
        compose = f"{self.__class__.__name__}[{sum(self.allocated)}/{self.n_nodes} Allocated]"
        compose += "\nTags:\n\t"+"\n\t".join([f"{k}: {v}" for (k,v) in self.tags.items()])
        return compose

    def limit_nodes(self, n_nodes):
        # Since this object does not control what happens to allocated nodes, reject any request
        # that could lead to work returning without the ability to de-allocate its nodes
        #
        # In the future, this constraint can be relaxed by re-sorting the list to ensure all active
        # allocations will return OK, but I am not implementing that yet as it should be unnecessary
        if sum(self.allocated[n_nodes:]) > 0:
            raise ValueError("Nodes are currently allocated beyond indicated limit's cutoff -- cannot fulfill request")
        # Ensure the limitation can be removed later on
        self.full_node_list = self.node_list
        self.full_n_nodes = self.n_nodes
        self.full_allocated = self.allocated
        # Update values to limit
        self.node_list = self.node_list.copy()[:n_nodes]
        self.n_nodes = n_nodes
        self.allocated = self.allocated.copy()[:n_nodes]
        self.limited = True

    def unlimit_nodes(self):
        if not self.limited:
            raise ValueError("Nodes are not currently limited")
        self.node_list = self.full_node_list
        self.n_nodes = self.full_n_nodes
        currently_allocated = [idx for (idx, val) in self.allocated if val]
        self.allocated = self.full_allocated
        # Ensure state remains consistent, it may have changed since limits were put on
        self.allocated[:] = False
        self.allocated[currently_allocated] = True
        self.limited = False

    def reserve_nodes(self, n_nodes, tag):
        if type(n_nodes) is str:
            # We may get an expression
            if 'x' in n_nodes:
                n_nodes = prod([int(_) for _ in n_nodes.split('x')])
            else:
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
        # Use .resolve to remove ambiguity when inspecting this data (working directories can change!)
        allocation_name = pathlib.Path("nodelist.txt").resolve()
        i = 1
        while allocation_name.exists():
            allocation_name = allocation_name.with_stem(f"nodelist_{i}").resolve()
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

