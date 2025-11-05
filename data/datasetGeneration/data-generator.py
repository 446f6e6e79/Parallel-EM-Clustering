#!/usr/bin/env python3
"""
Generate a synthetic dataset for EM (Expectation-Maximization) classification.

Usage examples:
    python data-generator.py --samples 1000 --features 2 --clusters 3 --output em_dataset.csv
    python data-generator.py -s 500 -f 2 -k 3 --means -5,0 0,5 5,0 --std 0.5 0.7 0.4 -o mydata.csv --plot
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import numpy as np

# Limit the number of points per cluster to plot for clarity
PLOT_MAX_PER_CLUSTER = 100
PLOT_SAMPLE_SEED = 123

def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate a synthetic dataset for EM classification (GMM clustering)."
    )
    parser.add_argument("-s", "--samples", type=int, default=1000, help="Number of samples (default: 1000)")
    parser.add_argument("-f", "--features", type=int, default=2, help="Number of features (default: 2)")
    parser.add_argument("-k", "--clusters", type=int, default=3, help="Number of clusters (default: 3)")
    parser.add_argument("--means", nargs="+", default=None,
                        help='List of cluster means, e.g. --means -5,0 0,5 5,0 (a mean per feature per cluster must be provided)')
    parser.add_argument("--std", type=float, nargs="+", default=None,
                        help="List of cluster standard deviations (default: all 1.0)")
    parser.add_argument("-r", "--random_state", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("-o", "--output", type=str, default="em_dataset.csv",
                        help="Output CSV filename (default: em_dataset.csv)")
    parser.add_argument("-m", "--metadata", type=str, default="em_metadata.txt",
                        help="Output metadata filename (default: em_metadata.txt)")
    
    parser.add_argument("--plot", action="store_true", help="Show a scatter plot of the dataset")
    return parser

def parse_means(means_list, n_clusters, n_features):
    """
        Parse means input from CLI: e.g. --means -5,0 0,5 5,0
    """
    if means_list is None:
        return None

    centers = []
    # For each mean string (e.g. -5,0 - one per cluster) parse the atomic parameters
    for mean_str in means_list:
        # Split the string into individual coordinates
        mean_values = [float(x) for x in mean_str.split(",")]
        # If for a cluster the number of means does not match the number of features, raise an error
        if len(mean_values) != n_features:
            raise ValueError(f"Each mean must have {n_features} values. Got: {mean_str}")
        centers.append(mean_values)
    # If the number of provided means does not match the number of clusters, raise an error
    if len(centers) != n_clusters:
        raise ValueError(f"Number of means must equal number of clusters ({n_clusters}).")
    return centers

def generate_em_dataset(n_samples, n_features, n_clusters, cluster_std, centers, random_state):
    """
        Generate synthetic Gaussian cluster data.
        Returns:
            X: array of shape (n_samples, n_features) - generated samples
            y_true: array of shape (n_samples,) - true cluster labels
    """
    # If centers not provided, pass integer n_clusters to make_blobs so it generates centers
    centers_arg = centers if centers is not None else n_clusters
    X, y_true = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=centers_arg,
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

    # Parse means
    centers = parse_means(args.means, args.clusters, args.features)

    # Generate data
    X, y_true = generate_em_dataset(
        n_samples=args.samples,
        n_features=args.features,
        n_clusters=args.clusters,
        cluster_std=cluster_std,
        centers=centers,
        random_state=args.random_state
    )

    # Define a DataFrame with all the generated features
    df = pd.DataFrame(X, columns=[f"feature{i+1}" for i in range(args.features)])
    # Add true labels to the DataFrame
    df["true_label"] = y_true
    # Save the dataset to CSV
    df.to_csv(args.output, index=False)
    
    print(f"Dataset saved to '{args.output}'")
    print(f"   Samples: {args.samples}, Features: {args.features}, Clusters: {args.clusters}")
    if centers is not None:
        print(f"   Means: {centers}")
    print(f"   Std: {cluster_std}")

    # Save metadata
    if args.metadata:
        with open(args.metadata, "w") as meta_file:
            meta_file.write(f"samples: {args.samples}\n")
            meta_file.write(f"features: {args.features}\n")
            meta_file.write(f"clusters: {args.clusters}\n")
            meta_file.write(f"means: {centers if centers is not None else 'generated'}\n")
            meta_file.write(f"std: {cluster_std}\n")
            meta_file.write(f"random_state: {args.random_state}\n")
        print(f"Metadata saved to '{args.metadata}'")

    # Subsample for plotting if too many points per cluster
    if args.plot and args.features == 2:
        make_plot(X, y_true)
    elif args.plot:
        print("Plotting is only supported for 2D datasets (--features 2).")

if __name__ == "__main__":
    main()