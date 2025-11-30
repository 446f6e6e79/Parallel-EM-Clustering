import pandas as pd
import os
import imageio
from .visualization_utils import create_clustering_frame_2d, create_clustering_frame_3d
from .EM_utils import *

def __visualize_2d_clustering(df, show_iteration=True, PAD=0.1):
    """
    Create a visualization frame for 2D clustering.
    Args:
        df (DataFrame): DataFrame containing clustering data of the iterations to visualize.
        show_iteration (bool, optional): Whether to display the iteration number as a title on the plot.
        PAD (float, optional): Padding factor for the plot limits.
    Returns:
        frames: List of numpy arrays representing the frames.
        plts: List of matplotlib figure objects.
    """
    # Fix the axis limits, based on the data range. This ensures consistent axes across frames.
    x_min, x_max = df['feature_1'].min(), df['feature_1'].max()
    y_min, y_max = df['feature_2'].min(), df['feature_2'].max()

    pad_x = PAD * (x_max - x_min)
    pad_y = PAD * (y_max - y_min)

    # Define limits with padding
    xlim = (x_min - pad_x, x_max + pad_x)
    ylim = (y_min - pad_y, y_max + pad_y)

    frames = []
    plts = []
    for it in sorted(df['iteration'].unique()):
        # Get the data for the specific iteration
        df_it = df[df['iteration'] == it]
        # Create the frame
        frame, plt = create_clustering_frame_2d(df_it,
                                                xlim=xlim, ylim=ylim,
                                                title=f"Iteration {it}" if show_iteration else "",
                                                show_errors=True)
        frames.append(frame)
        plts.append(plt)
    return frames, plts

#TODO: report this on the readme
def __visualize_3d_clustering(df, show_iteration=True, PAD=0.1):
    """
    Create a visualization frame for 3D clustering.
    Args:
        df (DataFrame): DataFrame containing clustering data of the iterations to visualize.
        show_iteration (bool, optional): Whether to display the iteration number as a title on the plot.
        PAD (float, optional): Padding factor for the plot limits.
    Returns:
        frames: List of numpy arrays representing the frames.
        plts: List of matplotlib figure objects.
    """
    x_min, x_max = df['feature_1'].min(), df['feature_1'].max()
    y_min, y_max = df['feature_2'].min(), df['feature_2'].max()
    z_min, z_max = df['feature_3'].min(), df['feature_3'].max()
    pad_x = PAD * (x_max - x_min)
    pad_y = PAD * (y_max - y_min)
    pad_z = PAD * (z_max - z_min)

    # Define limits with padding
    xlim = (x_min - pad_x, x_max + pad_x)
    ylim = (y_min - pad_y, y_max + pad_y)
    zlim = (z_min - pad_z, z_max + pad_z)

    frames = []
    plts = []

    for it in sorted(df['iteration'].unique()):
        # Get the data for the specific iteration
        df_it = df[df['iteration'] == it]
        # Create the frame
        frame, plt = create_clustering_frame_3d(df_it,
                                                xlim=xlim, ylim=ylim, zlim=zlim,
                                                title=f"Iteration {it}" if show_iteration else "",
                                                show_errors=True)
        frames.append(frame)
        plts.append(plt)
    return frames, plts

def visualize_dataset(csv_path, output_png=None):
    """
    Visualizes the initial dataset before clustering.
    Args:
        csv_path (str): Path to the CSV file containing the dataset.
        output_png (str, optional): Path to save the output PNG. If None, does not save.
    Returns:
        int: 0 on success.
    """   
    # See how many features we have
    df = pd.read_csv(csv_path)
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    if len(feature_cols) < 2 or len(feature_cols) > 3:
        raise ValueError(f"This visualization function only supports 2D and 3D visualizations. You have {len(feature_cols)}.")
    if len(feature_cols) == 2:
        print("Creating 2D dataset visualization...")
        # Create a simple scatter plot of the dataset
        frame, plot = create_clustering_frame_2d(df,
                                   xlim=(df['feature_1'].min()-1, df['feature_1'].max()+1),
                                   ylim=(df['feature_2'].min()-1, df['feature_2'].max()+1),
                                   title="Initial Dataset Visualization (2D)",
                                   colors=['gray'],
                                   show_ellipsoid=False,
                                   show_errors=False)   
    else:
        print("Creating 3D dataset visualization...")
        # Create a simple 3D scatter plot of the dataset
        frame, plot = create_clustering_frame_3d(df,
                                   xlim=(df['feature_1'].min()-1, df['feature_1'].max()+1),
                                   ylim=(df['feature_2'].min()-1, df['feature_2'].max()+1),
                                   zlim=(df['feature_3'].min()-1, df['feature_3'].max()+1),
                                   title="Initial Dataset Visualization (3D)",
                                   colors=['gray'],
                                   show_errors=False,
                                   show_ellipsoid=False)
    plot.gca().legend_.remove() if plot.gca().get_legend() else None
    if output_png is not None:
        plot.savefig(output_png)
        print(f"Saved initial dataset visualization to {output_png}")
    return 0    

def visualize_em(csv_path, output_gif, dims=None, iterations=None):
    """
    Visualizes the progression of the EM clustering algorithm as a GIF.
    Args:
        csv_path (str): Path to the CSV file containing clustering data.
        output_gif (str): Path to save the output GIF.
        iterations (list, optional): Specific iterations to include in the GIF. If None, includes all iterations.
    """
    # Check that the file exists
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
    
    # Load the CSV data
    df = pd.read_csv(csv_path)
    # See how many features we have
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    if len(feature_cols) < 2 or len(feature_cols) > 3:
        raise ValueError(f"This visualization function only supports 2D and 3D visualizations. You have {len(feature_cols)}.")

    # Use only last iteration to build mapping from predicted to real labels
    last_it = df['iteration'].max()
    df_last = df[df['iteration'] == last_it]
    pred_to_real, _, _ = derive_cluster_mapping(df_last)
    df['predicted_cluster'] = remap_predicted(df['predicted_cluster'].to_numpy(), pred_to_real)

    # Check if specific iterations are provided
    if iterations is not None:
        # Check if we passed only one int
        if isinstance(iterations, int):
            iterations = [iterations]
        df = df[df['iteration'].isin(iterations)]

    if len(feature_cols) == 2:
        print("Creating 2D clustering visualization...")
        frames, plts = __visualize_2d_clustering(df, output_gif)
    
    # We are in the 3D case
    else:
        print("Creating 3D clustering visualization...")
        frames, plts = __visualize_3d_clustering(df, output_gif)

    # If we specified a single iteration, just save that frame
    if iterations is not None and len(iterations) == 1:
        plts[0].savefig(output_gif.replace('.gif', '.png'))
        return
    # Otherwise, save the full animation
    imageio.mimsave(output_gif, frames, fps=min(2, len(frames)))
    print(f"Saved animation to {output_gif}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Visualize EM clustering progression as a GIF.")
    parser.add_argument("-i", "--csv", dest="csv_path", default="data/algorithm_results/debug.csv",
                        help="Path to the debug CSV file (default: data/algorithm_results/debug.csv)")
    parser.add_argument("-o", "--out", dest="output_gif", default="data/elaborated/em_visualization.gif",
                        help="Output GIF path (default: data/elaborated/em_visualization.gif)")
    parser.add_argument("--show-initial", dest="output_png", default=None, 
                        help="If provided, saves an initial dataset visualization to this PNG path.")
    parser.add_argument("--iterations", dest="iterations", nargs='+', type=int, default=None,
                        help="Specific iterations to include in the GIF, provide one or more iteration numbers (default: all iterations).")
    args = parser.parse_args()

    if args.output_png is not None:
        visualize_dataset(args.csv_path, args.output_png)
    else:
        visualize_em(args.csv_path, args.output_gif, iterations=args.iterations)