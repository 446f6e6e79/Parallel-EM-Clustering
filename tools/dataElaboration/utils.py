import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt

def compute_metrics(group):
    """
        Calculate speedup and efficiency of compute_time, e_step_time, m_step_time for a problem size group.
        The function gets the sequential execution time from the row where n_processes == 1
        Computes:
            - speedup as t1 / tN
            - efficiency as speedup / n_processes
    """
    # Get the sequential execution time (1 process)
    t1 = group[group['n_process'] == 1]['compute_time'].iloc[0]
    t1_e = group[group['n_process'] == 1]['e_step_time'].iloc[0]
    t1_m = group[group['n_process'] == 1]['m_step_time'].iloc[0]

    group = group.copy()    
    # Calculate speedup for compute_time, e_step_time, m_step_time
    group['speedup'] = t1 / group['compute_time']
    group['e_step_speedup'] = t1_e / group['e_step_time']
    group['m_step_speedup'] = t1_m / group['m_step_time']
    # Calculate efficiency
    group['efficiency'] = group['speedup'] / group['n_process']
    group['e_step_efficiency'] = group['e_step_speedup'] / group['n_process']
    group['m_step_efficiency'] = group['m_step_speedup'] / group['n_process']

    return group

def make_pivot(df, value, index='n_process', columns=['n_samples', 'n_features', 'n_clusters'],
               aggfunc='mean', round_decimals=3, fillna=None, process_order=None):
    """
        Create a pivot table from the DataFrame with specified parameters.
        Parameters:
            df: Input DataFrame
            value: Column name to aggregate
            index: Column to use as index (default 'n_processes')
            columns: Column to use as columns ( defined as the aggregation of n_samples, n_features, n_clusters as default)
            aggfunc: Aggregation function (default 'mean')
            round_decimals: Number of decimals to round the results (default 3)
            fillna: Value to fill NaNs (default None)
            process_order: List specifying the order of index values (default None, sorts unique index values)
        Returns:
            Pivot table as a DataFrame
    """

    # Determine process order if not provided
    if process_order is None:
        process_order = sorted(df[index].unique())

    # Pivot with multi-index columns
    pivot = df.pivot_table(index=index, columns=columns, values=value, aggfunc=aggfunc)

    if fillna is not None:
        pivot = pivot.fillna(fillna)
    if round_decimals is not None:
        pivot = pivot.round(round_decimals)

    pivot = pivot.reindex(process_order)
    return pivot

def format_scientific(size):
    """
        Format dataset size in scientific notation for labeling
    """
    if size >= 1e6:
        return f"{size/1e6:.1f}×10⁶"
    elif size >= 1e3:
        return f"{size/1e3:.0f}×10³"
    else:
        return f"{size:.0f}"

def plot_metrics(filtered_df, metric, fixed_parameters=None):
    """
        Create a Speedup figure using Plotly
        Parameters:
            filtered_df: DataFrame filtered for selected datasets
            metric: Metric to plot ('speedup' or 'efficiency')
            fixed_parameters: List of dataset columns to include in the label
    """
    if metric not in ['speedup', 'efficiency']:
        raise ValueError("Metric must be 'speedup' or 'efficiency'")
    
    if fixed_parameters is None:
        fixed_parameters = ['n_samples', 'n_features', 'n_clusters']

    # Color palette
    colors = px.colors.qualitative.Set1
    
    # Create Speedup figure
    fig = go.Figure()

    n_proc = np.sort(filtered_df['n_process'].unique())
    if metric == 'speedup':
        # Draw ideal speedup line
        fig.add_trace(go.Scatter(
            x=n_proc, y=n_proc,
            mode='lines',
            name='Ideal Speedup',
            line=dict(dash='dash', color='red', width=2)
        )) 
    else:
        # Draw acceptable efficiency line (constant = 0.70)
        fig.add_trace(go.Scatter(
            x=n_proc,
            y=[0.70] * len(n_proc),
            mode='lines',
            name='Acceptable Efficiency (0.70)',
            line=dict(dash='dot', color='red', width=2)
        ))

    filtered_df = filtered_df.copy()

    # Build a dataset label string for plotting.
    def build_label(row):
        parts = []
        for col in fixed_parameters:
            parts.append(f"{format_scientific(int(row[col]))}")
        return " - ".join(parts)
    
    filtered_df['dataset_label'] = filtered_df.apply(build_label, axis=1)
    
    unique_labels = filtered_df['dataset_label'].unique()

    # Plot data for each dataset
    for i, label in enumerate(unique_labels):
        subset = filtered_df[filtered_df['dataset_label'] == label]
        fig.add_trace(go.Scatter(
            x=subset['n_process'],
            y=subset[metric],
            mode='lines+markers',
            name=f"Dataset {label}",
            line=dict(width=3, color=colors[i % len(colors)]),
            marker=dict(size=8)
        ))

    # Layout settings
    if metric == 'speedup':
        fig.update_layout(
            title='Speedup Analysis',
            xaxis_title='Number of Processes (P)',
            yaxis_title='Speedup (T₁ / Tₚ)',
            template='plotly_white',
            font=dict(size=12),
            height=500,
            showlegend=True
        )
    else:
        fig.update_layout(
        title='Parallel Efficiency Analysis',
        xaxis_title='Number of Processes (P)',
        yaxis_title='Efficiency (Speedup / P)',
        template='plotly_white',
        font=dict(size=12),
        height=500,
        showlegend=True,
        yaxis=dict(range=[0, 1.05])
)
    # Return the figure to the caller
    return fig

def cluster_mapping(y_true, y_pred):
    """
    Return:
      - pred_to_real: dict mapping predicted_label -> real_label
      - real_to_pred: dict mapping real_label -> predicted_label
      - accuracy: permutation-invariant accuracy in [0,1]
    """
    labels_true = np.unique(y_true)
    labels_pred = np.unique(y_pred)
    # Use same label order on rows/cols (union), so we can map indices back to labels
    labels = np.unique(np.concatenate([labels_true, labels_pred]))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    row_ind, col_ind = linear_sum_assignment(-cm)

    pred_to_real = {labels[c]: labels[r] for r, c in zip(row_ind, col_ind) if labels[c] in set(labels_pred)}
    real_to_pred = {labels[r]: labels[c] for r, c in zip(row_ind, col_ind) if labels[r] in set(labels_true)}
    acc = (cm[row_ind, col_ind].sum() / cm.sum()) if cm.sum() > 0 else 0.0
    return pred_to_real, real_to_pred, acc

def clustering_accuracy(df):
    """
        Calculate clustering accuracy from a DataFrame containing 'predicted' and 'real' columns.
        Uses the Hungarian algorithm to find the best matching between predicted and real labels.
        Parameters:
            df: DataFrame containing the clustering results
        Returns:
            accuracy: Clustering accuracy as a float
    """
    y_pred = df['predicted_cluster'].to_numpy()
    y_true = df['real_cluster'].to_numpy()
    _, _, accuracy = cluster_mapping(y_true, y_pred)
    return accuracy

def plot_cov_ellipses(mean, cov, ax, color,
                      sigmas=(1, 2, 3),
                      alphas=None,
                      edge_alpha=0.9,
                      show_center=True,
                      center_kwargs=None):
    """
    Draw concentric covariance ellipses for multiple sigma levels.
        mean: 2D center
        cov:  2x2 covariance matrix
        sigmas: iterable of sigma radii (e.g. (1,2,3))
        alphas: iterable of face transparency (same length as sigmas). If None, auto.
        edge_alpha: transparency for edges
        show_center: draw an 'x' at center
    """
    sigmas = list(sigmas)
    # Largest first (so smaller ellipses draw on top)
    sigmas.sort(reverse=True)

    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    stds = np.sqrt(vals)

    # If alphas not provided, compute inversely proportional to sigmas
    # Outer ellipses more transparent
    if alphas is None:
        base = 0.35
        alphas = []
        for s in sigmas:
            # inverse proportional
            alphas.append(base / (0.6 + 0.6 * s))
    else:
        if len(alphas) != len(sigmas):
            raise ValueError("alphas length must match sigmas length")

    for s, a in zip(sigmas, alphas):
        width, height = 2 * s * stds  # 2*σ*s gives diameter for each axis
        ellipse = Ellipse(
            xy=mean,
            width=width,
            height=height,
            angle=theta,
            facecolor=color,
            edgecolor=color,
            lw=2,
            alpha=a
        )
        ec = ellipse.get_edgecolor()
        if isinstance(ec, (tuple, list)) and len(ec) == 3:
            ellipse.set_edgecolor((*ec, edge_alpha))
        ax.add_patch(ellipse)

    if show_center:
        if center_kwargs is None:
            center_kwargs = dict(marker='x', s=64, linewidths=2, color=color, zorder=6)
        ax.scatter([mean[0]], [mean[1]], **center_kwargs)

def create_clustering_frame(df, it, xlim, ylim, show_iteration=True):
    """
    Create a matplotlib figure for a specific iteration of clustering.
    Parameters:
        df: DataFrame containing clustering data
        it: Iteration number to visualize
        xlim: Tuple (xmin, xmax) for x-axis limits
        ylim: Tuple (ymin, ymax) for y-axis limits
    Returns:
        image: Numpy array representing the figure
    """
    df_it = df[df['iteration'] == it]
    fig, ax = plt.subplots(figsize=(6, 6))

    # Set axis limits
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    unique_clusters = sorted(df_it['predicted_cluster'].unique())
    color_map = {c: plt.cm.tab10(i % 10) for i, c in enumerate(unique_clusters)}

    for c in unique_clusters:
        data = df_it[df_it['predicted_cluster'] == c]
        color = color_map[c]
        ax.scatter(data['feature_1'], data['feature_2'], s=12, label=f'Cluster {c}', color=color, alpha=0.85)

        mean = [data['mu_k_1'].iloc[0], data['mu_k_2'].iloc[0]]
        cov = np.diag([data['sigma_k_1'].iloc[0], data['sigma_k_2'].iloc[0]])
        plot_cov_ellipses(mean, cov, ax, color, sigmas=(1,2,3))

        wrong = data[data['real_cluster'] != data['predicted_cluster']]
        if not wrong.empty:
            ax.scatter(wrong['feature_1'], wrong['feature_2'], marker='x', s=45, linewidths=2,
                       color=color, alpha=1.0, zorder=10)

    # Create legend, removing duplicates
    handles, labels = ax.get_legend_handles_labels()
    by_label = {}
    for h, l in zip(handles, labels):
        if l not in by_label:
            by_label[l] = h
    ax.legend(by_label.values(), by_label.keys(), frameon=False)
    # Add label for feature axes
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    
    if show_iteration:
        ax.set_title(f"Iteration {it}")
    plt.tight_layout()

    fig.canvas.draw()
    image = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
    plt.close(fig)
    return image


def derive_cluster_mapping(df):
    """
    Convenience wrapper using DataFrame columns 'predicted_cluster' and 'real_cluster'.
    """
    y_pred = df['predicted_cluster'].to_numpy()
    y_true = df['real_cluster'].to_numpy()
    return cluster_mapping(y_true, y_pred)

def remap_predicted(y_pred, pred_to_real):
    """
    Remap predicted labels into the real label space using the mapping.
    """
    return np.array([pred_to_real.get(p, p) for p in y_pred])