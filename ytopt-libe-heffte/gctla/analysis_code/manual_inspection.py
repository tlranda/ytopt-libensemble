import pandas as pd, numpy as np
import pathlib
import warnings
import itertools
from scipy.stats import entropy
import matplotlib.pyplot as plt
from sdv.single_table import GaussianCopulaSynthesizer as GaussianCopula
from sdv.metadata import SingleTableMetadata
from sdv.sampling.tabular import Condition
from sdv.constraints import ScalarRange
import argparse

compare_cols = ['p0']+[f'p1{d}' for d in 'xyz']+[f'p{_}' for _ in range(2,10)]

def make_dist(value_dict, data):
    # Use value dictionary to get distribution histogram for this dataset
    breakdown = {}
    common_denom = len(data)
    for key in compare_cols:
        values = value_dict[key]
        keydata = list(data[key])
        breakdown[key] = [keydata.count(val) / common_denom for val in values]
    return breakdown

def load_csv(name, drop_invalid=False, quantile=1.0, transform=None):
    data = pd.read_csv(name)
    # Always invert FLOPs for analysis of metric
    data['FLOPS'] *= -1
    if drop_invalid:
        data = data[data['FLOPS'] > 0]
    data.insert(len(data.columns), 'source', [name]*len(data))
    data = data.sort_values(by=['FLOPS']).reset_index(drop=True)
    # Quantile selection
    qlen = int(np.round(quantile * len(data), 0))
    data = data.iloc[0:qlen]
    # Transformation
    if transform is not None:
        for (key, info) in transform.items():
            condition_dict, replacement = info
            for (iiloc, compare) in condition_dict.items():
                if data.loc[iiloc] != compare:
                    match key:
                        case 'p7' | 'p8':
                            default, topo = minSurfaceSplit(data.loc[0,'p1x'],
                                                            data.loc[0,'p1y'],
                                                            data.loc[0,'p1z'],
                                                            data.loc[0,'mpi_ranks'])
                            mpi_ranks = data.loc[iiloc]
                            top_depth = int(np.log2(mpi_ranks))
                            np_replacement = np.asarray([[int(_) for _ in rep.split(' ')] for rep in replacement])
                            compare_mpi_ranks = np.prod(np_replacement[0,0])
                            max_depth = int(np.log2(compare_mpi_ranks))
                            new_topo = []
                            for t in data[key]:
                                xyz = np.asarray([np.log2(int(_))/top_depth for _ in t.split(' ')])
                                proj_xyz = 2 ** (xyz * max_depth)
                                distances = ((np_replacement - proj_xyz) ** 2).sum(axis=1)
                                best_match = np.argmin(distances)
                                new_topo.append(" ".join([str(_) for _ in np_replacement[best_match]]))
                            data.loc[:,key] = new_topo
                            break
                        case 'p9':
                            seq = sequence_builder(data.loc[0,'threads_per_node'],
                                                   data.loc[0,'ranks_per_node'])
                            new_seq = []
                            seq_len = len(seq)
                            compare_len = len(replacement)
                            for s in data[key]:
                                new_seq.append(replacement[min(compare_len,int(seq.index(s)/seq_len * compare_len))])
                            data.loc[:,key] = new_seq
                            break
                        case _:
                            raise ValueError(f"No transformation known for key '{key}'")
                    break
    return data

def GC_SDV(source_dist, quantile, n_nodes=1, ranks_per_node=64, problem_x=64, problem_y=64, problem_z=64):
    conditions = [Condition({'mpi_ranks': n_nodes * ranks_per_node,
                             'p1x': problem_x,
                             'p1y': problem_y,
                             'p1z': problem_z},
                            num_rows=200)]
    constraints = [{'constraint_class': 'ScalarRange', # App scale limits
                        'constraint_parameters': {
                            'column_name': 'p1x',
                            'low_value': 64,
                            'high_value': 2048,
                            'strict_boundaries': False,},
                        },
                   {'constraint_class': 'ScalarRange',
                        'constraint_parameters': {
                            'column_name': 'p1y',
                            'low_value': 64,
                            'high_value': 2048,
                            'strict_boundaries': False,},
                        },
                   {'constraint_class': 'ScalarRange',
                        'constraint_parameters': {
                            'column_name': 'p1z',
                            'low_value': 64,
                            'high_value': 2048,
                            'strict_boundaries': False,},
                        },
                   {'constraint_class': 'ScalarRange', # System scale limit
                        'constraint_parameters': {
                            'column_name': 'mpi_ranks',
                            'low_value': 1,
                            'high_value': 16384,
                            'strict_boundaries': False,},
                        },
                  ]
    train_data = source_dist[['c0',]+compare_cols+['mpi_ranks',]]
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(train_data)
    warnings.simplefilter('ignore')
    model = GaussianCopula(metadata, enforce_min_max_values=False)
    model.add_constraints(constraints=constraints)
    model.fit(train_data)
    warnings.simplefilter('default')
    # Conditional sampling
    cond_samples = model.sample_from_conditions(conditions)
    return cond_samples, model

def sequence_builder(threads_per_node, ranks_per_node):
    max_depth = threads_per_node // ranks_per_node
    sequence = []
    depth = 1
    while (seq := 2**depth) <= max_depth:
        sequence.append(seq)
        depth += 1
        if depth > 3 and (seq := sequence[-1]+sequence[-2]) < max_depth:
            sequence.append(seq)
    if max_depth not in sequence:
        sequence.append(max_depth)
    return sequence

# Minimum surface splitting solve is used as the default topology for FFT (picked by heFFTe when in-grid and/or out-grid topology == ' ')
def surface(fft_dims, grid):
    # Volume of FFT assigned to each process
    box_size = (np.asarray(fft_dims) / np.asarray(grid)).astype(int)
    # Sum of exchanged surface areas
    return (box_size * np.roll(box_size, -1)).sum()
def minSurfaceSplit(X, Y, Z, procs):
    fft_dims = (X, Y, Z)
    best_grid = (1, 1, procs)
    best_surface = surface(fft_dims, best_grid)
    best_grid = " ".join([str(_) for _ in best_grid])
    topologies = []
    # Consider other topologies that utilize all ranks
    for i in range(1, procs+1):
        if procs % i == 0:
            remainder = int(procs / float(i))
            for j in range(1, remainder+1):
                candidate_grid = (i, j, int(remainder/j))
                if np.prod(candidate_grid) != procs:
                    continue
                strtopology = " ".join([str(_) for _ in candidate_grid])
                topologies.append(strtopology)
                candidate_surface = surface(fft_dims, candidate_grid)
                if candidate_surface < best_surface:
                    best_surface = candidate_surface
                    best_grid = strtopology
    # Topologies are reversed such that the topology order is X-1-1 to 1-1-X
    # This matches previous version ordering
    return best_grid, list(reversed(topologies))

def build():
    prs = argparse.ArgumentParser()

    target = prs.add_argument_group('target')
    target.add_argument("--n-nodes", "--nodes", type=int, default=1,
                     help="Target number of nodes for Conditional Sampling (default: %(default)s)")
    target.add_argument("--ranks-per-node", "--rpn", type=int, default=64,
                     help="Number of ranks per node to determine MPI ranks based on # nodes (default: %(default)s)")
    target.add_argument("--problem-class", "--app-size", type=int, default=64,
                     help="Target cube-based application size to predict in Conditional Sampling (default: %(default)s)")
    target.add_argument("--problem-x", "--app-x", type=int, default=None,
                     help="Explicit X-dimension application size for Conditional Sampling (default: --problem-class/--app-size value)")
    target.add_argument("--problem-y", "--app-y", type=int, default=None,
                     help="Explicit Y-dimension application size for Conditional Sampling (default: --problem-class/--app-size value)")
    target.add_argument("--problem-z", "--app-z", type=int, default=None,
                     help="Explicit Z-dimension application size for Conditional Sampling (default: --problem-class/--app-size value)")

    transfer = prs.add_argument_group('transfer')
    transfer.add_argument("--comparison", required=True,
                     help="Data to use as ground truth for TL target")
    transfer.add_argument("--dataset", nargs="*", default=None, required=True,
                     help="Data to use to TRAIN transfer learning")
    transfer.add_argument("--drop", nargs="*", default=None,
                     help="Data to NOT include in transfer learning (anti-globbing) (default: always includes --comparison value)")
    transfer.add_argument("--fit-quantile", type=float, default=0.8,
                     help="Dataset filtered to top-quantile performance for GC fitting (default: %(default)s)")
    transfer.add_argument("--precomputed", default=None,
                     help="Data to use AS results from trained transfer learning (skip GC-fitting and conditional sampling in this runtime)")

    inspection = prs.add_argument_group("inspection")
    inspection.add_argument("--quantile", type=float, default=0.8,
                     help="Ground Truth quantile for TL to beat (deafult: %(default)s)")
    return prs

def parse(args=None, prs=None):
    if prs is None:
        prs = build()
    if args is None:
        args = prs.parse_args()
    # Set up dataset
    if args.drop is None:
        args.drop = []
    if args.comparison not in args.drop:
        args.drop.append(args.comparison)
    data = []
    for d in args.dataset:
        if d not in args.drop:
            data.append(d)
    args.dataset = data
    # Set problem class with explicit overrides
    for attrname in [f"problem_{d}" for d in "xyz"]:
        if getattr(args, attrname) is None:
            setattr(args, attrname, args.problem_class)
    return args

def main(args=None):
    args = parse(args)
    # Load data
    ground_truth = load_csv(args.comparison)
    print(f"Loaded {len(ground_truth)} values for ground truth distribution (From: {args.comparison})")
    # Filter to what we'd like to see things look like
    best = ground_truth[ground_truth['FLOPS'] >= ground_truth['FLOPS'].quantile(args.quantile)]
    print(f"Selected {len(best)} values for best distribution (quantile = {args.quantile})")

    # Determine available values
    tpn_lookup = (0,'threads_per_node')
    rpn_lookup = (0,'ranks_per_node')
    mpi_lookup = (0,'mpi_ranks')
    gt_seq = sequence_builder(ground_truth.loc[tpn_lookup],
                              ground_truth.loc[rpn_lookup])
    def_topo, gt_topo = minSurfaceSplit(args.problem_x,
                                        args.problem_y,
                                        args.problem_z,
                                        ground_truth.loc[mpi_lookup])
    transform_info = {'p7': ({mpi_lookup: ground_truth.loc[mpi_lookup]}, gt_topo),
                      'p8': ({mpi_lookup: ground_truth.loc[mpi_lookup]}, gt_topo),
                      'p9': ({tpn_lookup: ground_truth.loc[tpn_lookup],
                              rpn_lookup: ground_truth.loc[rpn_lookup]}, gt_seq),
                     }
    fft_dim_scales = sorted([2**_ for _ in range(6,11)] + [1400])
    value_dict = {'c0': ['fftw','cufft'],
                  'p0': ['double','float'],
                  'p1x': fft_dim_scales,
                  'p1y': fft_dim_scales,
                  'p1z': fft_dim_scales,
                  'p2': ['-no-reorder','-reorder',' '],
                  'p3': ['-a2a','-a2av',' '],
                  'p4': ['-p2p','-p2p_pl',' '],
                  'p5': ['-pencils','-slabs',' '],
                  'p6': ['-r2c_dir 0','-r2c_dir 1','-r2c_dir 2',' '],
                  'p7': gt_topo,
                  'p8': gt_topo,
                  'p9': gt_seq,
                  }

    source_dataset = []
    for fname in args.dataset:
        source_dataset.append(load_csv(fname, quantile=args.fit_quantile, transform=transform_info))
    source_dataset = pd.concat(source_dataset).reset_index(drop=True)
    print(f"Loaded {len(args.dataset)} files and {len(source_dataset)} values for TL distribution")
    if args.precomputed is None:
        # Set up TL model
        cond_samples, model = GC_SDV(source_dataset, args.fit_quantile, args.n_nodes, args.ranks_per_node, args.problem_x, args.problem_y, args.problem_z)
    else:
        cond_samples = load_csv(args.precomputed)
        print(f"Loaded precomputed file with {len(source_dataset)} values for TL distribution")

    # Make the comparable distributions
    true_dist = make_dist(value_dict, ground_truth)
    best_dist = make_dist(value_dict, best)
    tl_dist = make_dist(value_dict, source_dataset)
    cond_dist = make_dist(value_dict, cond_samples)

    # Prepare sub distributions for source data
    sub_dists = []
    source_names = []
    for (source_name, by_source) in source_dataset.groupby('source'):
        sub_dists.append(make_dist(value_dict, by_source))
        source_names.append(str(pathlib.Path(source_name).parent.stem))
    # Compare
    kl_div = np.zeros((len(compare_cols), 4))
    for idx, col in enumerate(compare_cols):
        for subidx, (name, a_dist, b_dist) in enumerate(zip(['TL vs Truth', 'TL vs Best', 'CS vs Truth', 'CS vs Best'],
                                                         [tl_dist, tl_dist, cond_dist, cond_dist],
                                                         [true_dist, best_dist, true_dist, best_dist])):
            kl_div[idx,subidx] = entropy(a_dist[col], b_dist[col])
            if not np.isfinite(kl_div[idx,subidx]):
                kl_div[idx,subidx] = entropy(b_dist[col], a_dist[col])
        td = np.asarray(true_dist[col])
        bs = np.asarray(best_dist[col])
        tld = np.asarray(tl_dist[col])
        csd = np.asarray(cond_dist[col])
        warnings.simplefilter('ignore')
        tld_truth_arr = np.asarray([_ for _ in td * np.log(td / tld) if np.isfinite(_)])
        csd_truth_arr = np.asarray([_ for _ in td * np.log(td / csd) if np.isfinite(_)])
        tld_best_arr = np.asarray([_ for _ in bs * np.log(bs / tld) if np.isfinite(_)])
        csd_best_arr = np.asarray([_ for _ in bs * np.log(bs / csd) if np.isfinite(_)])
        warnings.simplefilter('default')
        options = value_dict[col]
        print(col, "KL Div:\t", kl_div[idx])
        print("Options:\t",options)
        print("True Dist:\t", td, (td * len(ground_truth)).astype(int), sum((td * len(ground_truth)).astype(int)))
        print("Best Dist:\t", bs, (bs * len(best)).astype(int), sum((bs * len(best)).astype(int)))
        print("TL Data Dist:\t", tld, (tld * len(source_dataset)).astype(int), sum((tld * len(source_dataset)).astype(int)))
        print("Conditional Dist:\t", csd, (csd * len(cond_samples)).astype(int), sum((csd * len(cond_samples)).astype(int)))
        print("TL vs Truth Partial KL:\t", tld_truth_arr, tld_truth_arr.sum())
        print("CS vs Truth Partial KL:\t", csd_truth_arr, csd_truth_arr.sum())
        print("TL vs Best Partial KL:\t", tld_best_arr, tld_best_arr.sum())
        print("CS vs Best Partial KL:\t", csd_best_arr, csd_best_arr.sum())
        print()
        fig, ax = plt.subplots()
        ax.plot(np.arange(len(options)), td, label="True Dist", marker='.')
        ax.plot(np.arange(len(options)), bs, label="Best Dist", marker='+')
        ax.plot(np.arange(len(options)), tld, label="TL Dist", linestyle='--', marker='o')
        ax.plot(np.arange(len(options)), csd, label="CS Dist", linestyle='--', marker='*')
        ax.set_xticks(np.arange(len(options)))
        ax.set_xticklabels(options)
        ax.legend()
        ax.set_title(f"Distribution for {col} (TL Target: {args.n_nodes} Nodes x {args.problem_class} FFT)")
        fig.savefig(f"KL_Div_{col}.png", dpi=400)
        # Get Source attribution as stacked bar plot
        fig, ax = plt.subplots()
        heights = np.zeros((len(set(source_dataset['source'].values)), len(options)))
        for src_idx in range(len(sub_dists)):
            col_heights = np.asarray(sub_dists[src_idx][col])
            # Normalize and scale to TLD
            heights[src_idx] = ((col_heights/len(sub_dists)))*tld
        ticks = np.arange(len(options))
        for idx,name in enumerate(source_names):
            ax.bar(ticks, heights[idx],label=name,bottom=heights[:idx].sum(axis=0))
        ax.set_title(f"Source Contribution for {col} (Left out: {args.n_nodes} Nodes x {args.problem_class} FFT)")
        ax.set_xticks(ticks)
        ax.set_xticklabels(options)
        ax.legend()
        fig.savefig(f"Source_Dist_{col}.png", dpi=400)
    plt.show()

if __name__ == '__main__':
    main()

