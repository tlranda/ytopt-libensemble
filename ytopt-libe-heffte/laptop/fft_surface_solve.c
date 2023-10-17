#include <stdio.h>
#include <valarray>

int main(void) {
  int num_procs = 4096;

  const int FFT_SIZE = 1400;
  std::valarray<int> all_indexes = {FFT_SIZE, FFT_SIZE, FFT_SIZE};
  std::valarray<int> best_grid = {1,1, num_procs};

  auto surface = [&](std::valarray<int> const &proc_grid)->
    int{
      auto box_size = all_indexes / proc_grid;
      printf("\t\tBox size = <%d, %d, %d>\n", box_size[0], box_size[1], box_size[2]);
      printf("\t\tElements of product are <%d, %d, %d>\n", (box_size * box_size.cshift(1))[0], (box_size * box_size.cshift(1))[1], (box_size * box_size.cshift(1))[2]);
      return ( box_size * box_size.cshift(1) ).sum();
    };
  int best_surface = surface({1,1, num_procs});
  printf("Initial solve has surface: %d\n", best_surface);

  for (int i = 1; i <= num_procs; i++) {
    if (num_procs % i == 0) {
      printf("Solvable iteration: %d\n", i);
      int const remainder = num_procs / i;
      printf("\tRemainder iterations: %d\n", remainder);
      for (int j = 1; j <= remainder; j++) {
        std::valarray<int> candidate_grid = {i, j, remainder / j};
        int const candidate_surface = surface(candidate_grid);
        printf("\t\t\tSolve <%d, %d, %d> has surface %d\n", candidate_grid[0], candidate_grid[1], candidate_grid[2], candidate_surface);
        if (candidate_surface < best_surface) {
          best_surface = candidate_surface;
          best_grid = candidate_grid;
          printf("\t\t\t\tNEW BEST SOLVE <%d, %d, %d>!\n", i, j, remainder/j);
        }
      }
    }
  }
  printf("Final grid <%d, %d, %d> has surface %d\n", best_grid[0], best_grid[1], best_grid[2], best_surface);
}

