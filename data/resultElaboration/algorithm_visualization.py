import pandas as pd
import numpy as np
import imageio
from utils import create_clustering_frame

def visualize_em(csv_path, output_gif="data/em_progress.gif"):
    # Read CSV data
    df = pd.read_csv(csv_path)
    frames = []
    #TODO: add a method to convert predicted cluster labels to the true labels

    # For each iteration, create a frame
    for it in sorted(df['iteration'].unique()):
        frame = create_clustering_frame(df, it)
        frames.append(frame)

    # Write all the frames as a gif
    imageio.mimsave(output_gif, frames, fps=2)
    print(f"Saved animation to {output_gif}")

if __name__ == "__main__":
    visualize_em("data/debug.csv")