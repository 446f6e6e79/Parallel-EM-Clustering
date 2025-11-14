import pandas as pd
import numpy as np
import imageio
from utils import create_clustering_frame, remap_predicted, derive_cluster_mapping

def visualize_em(csv_path, output_gif="data/em_progress.gif"):
    # Load the CSV data
    df = pd.read_csv(csv_path)

    # Use only last iteration to build mapping
    last_it = df['iteration'].max()
    df_last = df[df['iteration'] == last_it]

    pred_to_real, real_to_pred, _ = derive_cluster_mapping(df_last)
    df['predicted_cluster'] = remap_predicted(df['predicted_cluster'].to_numpy(), pred_to_real)

    frames = []
    for it in sorted(df['iteration'].unique()):
        frames.append(create_clustering_frame(df, it))

    imageio.mimsave(output_gif, frames, fps=2)
    print(f"Saved animation to {output_gif}")

if __name__ == "__main__":
    visualize_em("data/debug.csv")