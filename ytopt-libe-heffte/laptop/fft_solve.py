import numpy as np
import itertools
import time
import timeit

class mss:
    def __init__(self, num_procs=64, FFT_SIZE=512):
        self.num_procs = num_procs
        self.FFT_SIZE = FFT_SIZE
        self.all_indexes = (self.FFT_SIZE, self.FFT_SIZE, self.FFT_SIZE)
        self.factors = [2**x for x in range(int(np.log2(self.num_procs)),-1,-1)]
        self.reset()

    def reset(self):
        self.best_grid = (1,1,self.num_procs)
        self.best_surface = self.surface(self.best_grid)

    def surface(self, grid):
        box_size = (np.asarray(self.all_indexes)/np.asarray(grid)).astype(int)
        return (box_size * np.roll(box_size, -1)).sum()

    def v1(self):
        topos = []
        for i in range(1,self.num_procs+1):
            if self.num_procs % i == 0:
                remainder = int(self.num_procs / float(i))
                for j in range(1, remainder+1):
                    candidate_grid = (i, j, int(remainder/j))
                    if np.prod(candidate_grid) != self.num_procs:
                        continue
                    topos.append(candidate_grid)
                    candidate_surface = self.surface(candidate_grid)
                    if candidate_surface < self.best_surface:
                        self.best_surface = candidate_surface
                        self.best_grid = candidate_grid
        topos = reversed(topos)

    def v2(self):
        topos = []
        for candidate_grid in itertools.product(self.factors,repeat=3):
            if np.prod(candidate_grid) != self.num_procs:
                continue
            topos.append(candidate_grid)
            candidate_surface = self.surface(candidate_grid)
            if candidate_surface < self.best_surface:
                self.best_surface = candidate_surface
                self.best_grid = candidate_grid


print(timeit.repeat(stmt='x.v1()', setup='x = mss()', repeat=5, number=100, globals=globals()))
print(timeit.repeat(stmt='x.v2()', setup='x = mss()', repeat=5, number=100, globals=globals()))
#print(f"Best solve <{best_grid[0]}, {best_grid[1]}, {best_grid[2]}> with surface {best_surface} in {t1-t0} seconds")

