#!/usr/bin/env python3
"""
Generate a synthetic dataset for EM (Expectation-Maximization) classification.

Usage examples:
    python data-generator.py --samples 1000 --features 2 --clusters 3 --output em_dataset.csv
    python data-generator.py -s 500 -f 2 -k 3 --means-list "-5,0 0,5 5,0" --std 0.5 0.7 0.4 -o mydata.csv --plot
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import numpy as np
from pathlib import Path
import re

# Limit the number of points per cluster to plot for clarity
PLOT_MAX_PER_CLUSTER = 100
PLOT_SAMPLE_SEED = 123

def get_repo_data_dir() -> Path:
    """
        Return the absolute path to the repository's data directory,
        regardless of the current working directory.
    """
    # parents[0]=.../data/datasetGeneration, parents[1]=.../data, parents[2]=.../Parallel-EM-Clustering
    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir

def resolve_to_data_dir(path_str: str) -> Path:
    """
        If path_str is absolute, return it unchanged.
        If relative, place the file (basename) inside the repo data directory.
    """
    p = Path(path_str)
    if p.is_absolute():
        return p
    return get_repo_data_dir() / p.name

def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate a synthetic dataset for EM classification (GMM clustering)."
    )
    parser.add_argument("-s", "--samples", type=int, default=1000, help="Number of samples (default: 1000)")
    parser.add_argument("-f", "--features", type=int, default=2, help="Number of features (default: 2)")
    parser.add_argument("-k", "--clusters", type=int, default=3, help="Number of clusters (default: 3)")
    # Accept repeated --means (one per cluster). Use equals for negatives: --means=-3,5
    parser.add_argument("--means", dest="means", action="append", metavar="MEAN", default=None,
                        help='Repeat per cluster, e.g. --means=2,-2 --means=0,1 --means=-3,5 (use "=" for negatives: --means=-3,5)')
    # Also accept a single string list (quoted), e.g. --means-list "2,-2 0,1 -3,5" or "--means-list 2,-2;0,1;-3,5"
    parser.add_argument("--means-list", dest="means_list", type=str, default=None,
                        help='Quoted list: "2,-2 0,1 -3,5" or "2,-2;0,1;-3,5"')
    parser.add_argument("--std", type=float, nargs="+", default=None,
                        help="List of cluster standard deviations (default: all 1.0)")
    parser.add_argument("-equal", "--equal_size", action="store_true",
                        help="Generate clusters of equal size (default: False)", default=False)
    parser.add_argument("-r", "--random_state", type=int, default=43, help="Random seed (default: 43)")
    parser.add_argument("-o", "--output", type=str, default="em_dataset.csv",
                        help="Output CSV filename (default: em_dataset.csv)")
    parser.add_argument("-m", "--metadata", type=str, default="em_metadata.txt",
                        help="Output metadata filename (default: em_metadata.txt)")
    parser.add_argument("--plot", action="store_true", help="Show a scatter plot of the dataset")
    return parser

def parse_means(means_flags, means_list_str, n_clusters, n_features):
    """
        Support:
          - Repeated flags: --means 2,-2 --means 0,1 --means -3,5
            (use equals for negatives: --means=-3,5)
          - Single list:   --means-list "2,-2 0,1 -3,5" or "2,-2;0,1;-3,5"
    """
    tokens = []
    if means_list_str:
        # split by whitespace
        tokens.extend([t for t in re.split(r"\s+", means_list_str.strip()) if t])

    if means_flags:
        # each flag is one "a,b" string; note negatives require --means=-3,5 form
        tokens.extend(means_flags)

    if not tokens:
        return None

    centers = []
    for mean_str in tokens:
        vals = [float(x) for x in mean_str.split(",")]
        if len(vals) != n_features:
            raise ValueError(f"Each mean must have {n_features} values. Got: {mean_str}")
        centers.append(vals)

    if len(centers) != n_clusters:
        raise ValueError(f"Number of means must equal number of clusters ({n_clusters}). Got {len(centers)}.")
    return centers

def compute_cluster_counts(total_samples: int, k: int, equal: bool, seed: int) -> list[int]:
    """
        Return a list of k per-cluster sample counts summing to total_samples.
        Guarantees each cluster has at least 1 sample (if total_samples >= k).
    """
    if total_samples < k:
        raise ValueError(f"Total samples ({total_samples}) must be >= number of clusters ({k})")
    
    # If we specified uniform cluster sizes, divide equally the number of samples
    if equal:
        base = total_samples // k
        counts = [base] * k
        # Distribute remainder
        for i in range(total_samples - base * k):
            counts[i] += 1
        return counts
    
    rng = np.random.default_rng(seed)
    # Dirichlet for proportions (alpha=1 => uniform over simplex)
    weights = rng.dirichlet(alpha=np.ones(k))
    counts = np.floor(weights * total_samples).astype(int)
    
    # Add one sample to any empty cluster
    counts[counts == 0] = 1
    
    # Compute how many samples we need to add/subtract to reach total_samples 
    diff = total_samples - counts.sum()
    # Adjust diff (positive: add; negative: subtract)
    while diff != 0:
        if diff > 0:
            i = rng.integers(0, k)
            counts[i] += 1
            diff -= 1
        else:  # diff < 0 we need to remove samples
            # choose cluster with count > 1
            idxs = np.where(counts > 1)[0]
            i = idxs[rng.integers(0, len(idxs))]
            counts[i] -= 1
            diff += 1
    return counts.tolist()

def generate_em_dataset(n_samples, n_features, n_clusters, cluster_std, centers, random_state):
    """
        Generate synthetic Gaussian cluster data.
        Returns:
            X: array of shape (n_samples, n_features) - generated samples
            y_true: array of shape (n_samples,) - true cluster labels
    """
    X, y_true = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=centers,
        cluster_std=cluster_std,
        random_state=random_state
    )
    return X, y_true

def make_plot(X, y_true, max_per_cluster=PLOT_MAX_PER_CLUSTER, sample_seed=PLOT_SAMPLE_SEED):
    # Setup random generator for reproducibility
    rng = np.random.default_rng(sample_seed)
    # Indices of all data points
    data_indices = np.arange(len(y_true))

    selected = []
    # For each cluster k, randomly select up to PLOT_MAX_PER_CLUSTER points
    for k in np.unique(y_true):
        # Indices of points in cluster k
        k_indices = data_indices[y_true == k]
        max_points_per_cluster = min(max_per_cluster, k_indices.size)

        # Select max_points_per_cluster random indices from k_indices
        selected.append(rng.choice(k_indices, size=max_points_per_cluster, replace=False))
    
    sel = np.concatenate(selected) if selected else data_indices
    plt.figure(figsize=(7, 5))
    plt.scatter(X[sel, 0], X[sel, 1], c=y_true[sel], cmap="viridis", s=30)
    plt.title("Generated Dataset for EM Classification (True Clusters)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

def longest_line_len(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return max((len(line.rstrip("\n")) for line in f), default=0)

def main():
    parser = build_parser()
    args = parser.parse_args()

    # If no std provided, default to 1.0 for all clusters
    if args.std is None:
        cluster_std = [1.0] * args.clusters
    else:
        # If one single std provided, use it for all clusters
        if len(args.std) == 1:
            cluster_std = [args.std[0]] * args.clusters
        elif len(args.std) == args.clusters:
            cluster_std = args.std
        else:
            raise ValueError("Provide either one std (applied to all clusters) or one per cluster.")

    # Parse means (supports --means-list or repeated --means)
    centers = parse_means(args.means, args.means_list, args.clusters, args.features)

    # Determine per-cluster counts
    per_cluster_counts = compute_cluster_counts(args.samples, args.clusters, args.equal_size, args.random_state)
    total_samples = sum(per_cluster_counts)

    # Generate data
    X, y_true = generate_em_dataset(
        n_samples=per_cluster_counts,
        n_features=args.features,
        n_clusters=args.clusters,
        cluster_std=cluster_std,
        centers=centers,
        random_state=args.random_state
    )

    # Resolve output paths to the repo's data directory
    output_path = resolve_to_data_dir(args.output)
    metadata_path = resolve_to_data_dir(args.metadata) if args.metadata else None

    # Define a DataFrame with all the generated features
    df = pd.DataFrame(X, columns=[f"feature{i+1}" for i in range(args.features)])
    # Add true labels to the DataFrame
    df["true_label"] = y_true
    # Save the dataset to CSV
    df.to_csv(output_path, index=False)

    # Print summary
    print(f"Dataset saved to '{output_path}'")
    print(f"   Total samples: {total_samples} (per cluster: {per_cluster_counts})")
    print(f"   Features: {args.features}, Clusters: {args.clusters}")
    if centers is not None:
        print(f"   Means: {centers}")
    print(f"   Std: {cluster_std}")
    # length of the longest line in the CSV
    longest = longest_line_len(output_path)
    print(f"   Longest CSV line: {longest} chars")

    # Save metadata
    if metadata_path:
        with open(metadata_path, "w") as meta_file:
            meta_file.write(f"samples: {total_samples}\n")
            meta_file.write(f"features: {args.features}\n")
            meta_file.write(f"clusters: {args.clusters}\n")
            meta_file.write(f"samples_per_cluster: {per_cluster_counts}\n")
            meta_file.write(f"means: {centers if centers is not None else 'generated'}\n")
            meta_file.write(f"std: {cluster_std}\n")
            meta_file.write(f"random_state: {args.random_state}\n")
            meta_file.write(f"max_line_size: {longest}\n")
        print(f"Metadata saved to '{metadata_path}'")

    # Subsample for plotting if too many points per cluster
    if args.plot and args.features == 2:
        make_plot(X, y_true)
    elif args.plot:
        print("Plotting is only supported for 2D datasets (--features 2).")

if __name__ == "__main__":
    main()