import pandas as pd
import numpy as np
import imageio
from utils import create_clustering_frame, remap_predicted, derive_cluster_mapping

def visualize_em(csv_path, output_gif, iterations=None):
    """
    Visualizes the progression of the EM clustering algorithm as a GIF.
    Args:
        csv_path (str): Path to the CSV file containing clustering data.
        output_gif (str): Path to save the output GIF.
        iterations (list, optional): Specific iterations to include in the GIF. If None, includes all iterations.
    """
    # Load the CSV data
    df = pd.read_csv(csv_path)
    
    # Use only last iteration to build mapping
    last_it = df['iteration'].max()
    df_last = df[df['iteration'] == last_it]
    pred_to_real, _, _ = derive_cluster_mapping(df_last)
    df['predicted_cluster'] = remap_predicted(df['predicted_cluster'].to_numpy(), pred_to_real)

    # Fix the axis limits, based on the data range. This ensures consistent axes across frames.
    x_min, x_max = df['feature_1'].min(), df['feature_1'].max()
    y_min, y_max = df['feature_2'].min(), df['feature_2'].max()
    
    PAD = 0.1   # 10% padding
    pad_x = PAD * (x_max - x_min)
    pad_y = PAD * (y_max - y_min)

    # Define limits with padding
    xlim = (x_min - pad_x, x_max + pad_x)
    ylim = (y_min - pad_y, y_max + pad_y)

    # Filter iterations if specified    
    if iterations is not None:
        df = df[df['iteration'].isin(iterations)]

    frames = []
    for it in sorted(df['iteration'].unique()):
        frames.append(create_clustering_frame(df, it, xlim=xlim, ylim=ylim, show_iteration=True))

    imageio.mimsave(output_gif, frames, fps=min(2, len(frames)))
    print(f"Saved animation to {output_gif}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize EM clustering progression as a GIF.")
    parser.add_argument("-i", "--csv", dest="csv_path", default="data/algorithm_results/debug.csv",
                        help="Path to the debug CSV file (default: data/algorithm_results/debug.csv)")
    parser.add_argument("-o", "--out", dest="output_gif", default="data/elaborated/em_visualization.gif",
                        help="Output GIF path (default: data/elaborated/em_visualization.gif)")
    args = parser.parse_args()
    visualize_em(args.csv_path, args.output_gif, iterations=None)