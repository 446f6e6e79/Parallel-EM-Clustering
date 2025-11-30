import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

def _get_palette():
    """Return a consistent color palette."""
    return plt.cm.tab10.colors

def create_line_figure(xlabel, ylabel, title, figsize=(10, 6), grid=True):
    """
    Create a matplotlib figure/axes with consistent styling for line plots.
    Returns (fig, ax, colors).
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    # Define color palette from the universal color map
    colors = _get_palette()

    # Font sizes
    title_fs = 18
    label_fs = 14
    tick_fs = 12

    ax.set_xlabel(xlabel, fontsize=label_fs)
    ax.set_ylabel(ylabel, fontsize=label_fs)
    ax.set_title(title, fontsize=title_fs)

    # Tick label size
    ax.tick_params(axis='both', labelsize=tick_fs)

    # Set the grid if requested
    if grid:
        ax.grid(True, linestyle='-', alpha=0.6)
    return fig, ax, colors

def styled_line_plot(x_list, y_list, labels,
                     xlabel, ylabel, title,
                     y_min=None, y_max=None,
                     figsize=(10, 6), grid=True, legend=True, 
                     lineStyles=['-'], markers=['o']):
    """
    Generic multi‑line plot with consistent style.

    Parameters:
        x_list : sequence of 1D arrays
        y_list : sequence of 1D arrays
        labels : sequence of str
        xlabel : str, x-axis label
        ylabel : str, y-axis label
        title : str, title of the plot
        grid : bool
        legend : bool
        lineStyles : list of str, line styles for each line
    Returns:
        fig, ax : matplotlib Figure and Axes objects
    """
    # Create the figure from the baseline
    fig, ax, colors = create_line_figure(xlabel, ylabel, title, figsize, grid)
    # Plot each line with consistent styling
    for i, (x, y, lab) in enumerate(zip(x_list, y_list, labels)):
        ax.plot(
            x,
            y,
            marker=markers[i % len(markers)],
            linestyle=lineStyles[i % len(lineStyles)],
            linewidth=2,
            markersize=6,
            color=colors[i % len(colors)],
            label=lab,
        )
     # Set fixed y-limits if requested
    if y_min is not None or y_max is not None:
        ax.set_ylim(y_min, y_max)
    # Add legend if requested
    if legend:
        ax.legend(frameon=True, fontsize=12)
    # Adjust layout
    fig.tight_layout()
    return fig, ax

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
        Plot speedup or efficiency using matplotlib.
        Parameters:
            filtered_df: DataFrame filtered for selected datasets
            metric: Metric to plot ('speedup' or 'efficiency')
            fixed_parameters: List of dataset columns to include in the label
    """
    if metric not in ['speedup', 'efficiency']:
        raise ValueError("Metric must be 'speedup' or 'efficiency'")

    if fixed_parameters is None:
        fixed_parameters = ['n_samples', 'n_features', 'n_clusters']

    df = filtered_df.copy()

    # Build a dataset label string for plotting
    def build_label(row):
        parts = [format_scientific(int(row[col])) for col in fixed_parameters]
        return " - ".join(parts)
    
    # Label each dataset size with its relative scientific notation 
    df['dataset_label'] = df.apply(build_label, axis=1)
    unique_labels = df['dataset_label'].unique()

    # Prepare data for generic line plot
    x_list, y_list, labels = [], [], []
    n_proc = np.sort(df['n_process'].unique())

    # For each dataset, add its data to the plot lists
    for label in unique_labels:
        subset = df[df['dataset_label'] == label].sort_values('n_process')
        x_list.append(subset['n_process'].to_numpy())
        y_list.append(subset[metric].to_numpy())
        labels.append(f"Dataset {label}")
    # Add labels and titles
    ylabel = 'Speedup (T₁/Tₚ)' if metric == 'speedup' else 'Efficiency (Speedup / P)'
    title = 'Speedup Analysis' if metric == 'speedup' else 'Parallel Efficiency Analysis'

    fig, ax = styled_line_plot(
        x_list,
        y_list,
        labels,
        xlabel='Number of Processes (P)',
        ylabel=ylabel,
        title=title,
        figsize=(10, 6),
        grid=True,
        legend=True,
        y_min=0, y_max=None
    )

    # Add the ideal lines for reference
    if metric == 'speedup':
        ax.plot(n_proc, n_proc, 'r--', label='Ideal Speedup', linewidth=2)
    else:
        ax.plot(
            n_proc,
            [0.7] * len(n_proc),
            'r:',
            label='Acceptable Efficiency (0.70)',
            linewidth=2,
        )

    # Re‑draw legend to include reference line
    ax.legend(frameon=True, fontsize=12)
    fig.tight_layout()
    return plt

def plot_heatmap_comparison(table1, table2):
    """
        Plot comparison between two pivot tables.
        Parameters:
            table1: First pivot table (e.g., efficiency_hybrid_table)
            table2: Second pivot table (e.g., efficiency_mpi_table)
    """
    # Plot each n_samples as a separate line
    diff_table = table1 - table2

    plt.figure(figsize=(14, 6))
    sns.heatmap(diff_table, annot=True, cmap='RdBu_r', center=0)
    plt.xlabel('Dataset (n_samples, n_features, n_clusters)')
    plt.ylabel('Number of Processes')
    plt.title('Hybrid Efficiency - MPI Efficiency')
    plt.show()

def plot_graph_comparison(table1, table2):
    """
        Plot comparison between two pivot tables as line graphs.
    """
    avg_efficiency_hybrid = table1.mean(axis=1)
    avg_efficiency_mpi = table2.mean(axis=1)
    # Prepare data for generic line plot. On the x size we have number of processes, on y size average efficiency
    x_list = [avg_efficiency_hybrid.index, avg_efficiency_mpi.index]
    y_list = [avg_efficiency_hybrid.values, avg_efficiency_mpi.values]
    labels = ['Hybrid', 'MPI']

    fig, ax = styled_line_plot(
        x_list,
        y_list,
        labels,
        xlabel='Number of Processes',
        ylabel='Average Efficiency',
        title='Average Efficiency per Number of Processes',
        figsize=(10, 6),
        grid=True,
        legend=True,
        y_min=0, y_max=None,
        lineStyles=['-', '--'], markers=['o', 'x']
    )
    fig.tight_layout()
    plt.show()

def plot_cov_ellipses_2d(mean, cov, ax, color,
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

def plot_cov_ellipsoids_3d(mean, cov, ax, color,
                           sigmas=(1, 2, 3),
                           alphas=None,
                           edge_alpha=0.9,
                           show_center=True,
                           center_kwargs=None):
    """
    Draw concentric covariance ellipsoids for multiple sigma levels.
    (Simplified for diagonal 3x3 covariance matrix, no rotation needed)

    Parameters:
        mean (np.array): 3D center (mu_x, mu_y, mu_z).
        cov (np.array): Diagonal 3x3 covariance matrix.
        ax (Axes3D): Matplotlib 3D axis object.
        color (str): Color for the surface.
        sigmas (tuple): Iterable of sigma radii (e.g., (1, 2, 3)).
        alphas (iterable): Iterable of face transparency (same length as sigmas). If None, auto.
        edge_alpha (float): Transparency for edges (unused in simple surface plot).
        show_center (bool): Draw an 'x' at center.
    """
    
    sigmas = list(sigmas)
    # Largest first (so smaller ellipsoids draw on top)
    sigmas.sort(reverse=True)

    # Check for diagonal covariance and extract standard deviations
    if cov.shape != (3, 3) or not np.allclose(cov - np.diag(np.diag(cov)), 0):
        print("Warning: Function is optimized for diagonal 3x3 covariance matrices.")
        
    # Get the square root of diagonal elements (variances) to get standard deviations
    stds = np.sqrt(np.diag(cov))
    
    mu_x, mu_y, mu_z = mean
    sigma_x, sigma_y, sigma_z = stds

    # If alphas not provided, compute inversely proportional to sigmas
    if alphas is None:
        base = 0.35
        alphas = []
        for s in sigmas:
            alphas.append(base / (0.6 + 0.6 * s))
    else:
        if len(alphas) != len(sigmas):
            raise ValueError("alphas length must match sigmas length")

    # 1. Create the base sphere surface points using spherical coordinates
    u = np.linspace(0, 2 * np.pi, 50) # Use 50 points for resolution
    v = np.linspace(0, np.pi, 50)
    
    # Generate coordinates for a unit sphere (radius 1)
    x_unit = np.outer(np.cos(u), np.sin(v))
    y_unit = np.outer(np.sin(u), np.sin(v))
    z_unit = np.outer(np.ones(np.size(u)), np.cos(v))

    # 2. Scale and plot the ellipsoids for each sigma level
    for s, a in zip(sigmas, alphas):
        # Scale the unit sphere by (s * sigma_i) for each axis
        # and shift by the mean (mu_i)
        x = s * sigma_x * x_unit + mu_x
        y = s * sigma_y * y_unit + mu_y
        z = s * sigma_z * z_unit + mu_z

        # Plot the 3D surface
        ax.plot_surface(
            x, y, z,
            rstride=4, cstride=4,
            color=color,
            alpha=a,
            edgecolor=color,
            linewidth=0.5 * edge_alpha,
            shade=False, # Disable shading for a cleaner look
            zorder=-1 # Draw below data points
        )
        
    # 3. Plot the center point
    if show_center:
        if center_kwargs is None:
            center_kwargs = dict(marker='o', s=45, linewidths=2, color='k', zorder=10)
        ax.scatter([mean[0]], [mean[1]], [mean[2]], **center_kwargs)

def create_clustering_frame_2d(df_it, xlim, ylim, title="", show_errors=True, show_ellipsoid=True, colors=None, legend=True):
    """
    Create a 2d plot for a specific iteration of clustering.
    Parameters:
        df_it: DataFrame containing clustering data for a specific iteration
        xlim: Tuple (xmin, xmax) for x-axis limits
        ylim: Tuple (ymin, ymax) for y-axis limits
        title: Title for the plot
        show_errors: Whether to highlight misclassified points
        show_ellipsoid: Whether to show covariance ellipses
        colors: List of colors to use for clusters (default None, uses consistent palette)
        legend: Whether to show legend
    Returns:
        image: Numpy array representing the figure
        fig: Matplotlib figure object
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(6, 6))

    # Set axis limits
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    unique_clusters = sorted(df_it['predicted_cluster'].unique())
    # Define color map, using the default consistent palette
    colors = _get_palette() if colors is None else colors
    color_map = {c: colors[i % len(colors)] for i, c in enumerate(unique_clusters)}

    # For each cluster, plot points and covariance ellipses
    for c in unique_clusters:
        # Select data for the current cluster
        data_c = df_it[df_it['predicted_cluster'] == c]
        color = color_map[c]
        
        # Plot the data points for cluster c
        ax.scatter(data_c['feature_1'], data_c['feature_2'], s=12, label=f'Cluster {c}', color=color, alpha=0.85)

        # Plot covariance ellipse
        mean = [data_c['mu_k_1'].iloc[0], data_c['mu_k_2'].iloc[0]]
        cov = np.diag([data_c['sigma_k_1'].iloc[0], data_c['sigma_k_2'].iloc[0]])
        if show_ellipsoid:
            plot_cov_ellipses_2d(mean, cov, ax, color, sigmas=(1,2,3))

        # If not showing errors, skip
        if not show_errors:
            continue
        # Highlight misclassified points with 'x' marker
        wrong = data_c[data_c['real_cluster'] != data_c['predicted_cluster']]
        if not wrong.empty:
            ax.scatter(wrong['feature_1'], wrong['feature_2'], marker='x', s=45, linewidths=2,
                       color=color, alpha=1.0, zorder=10)

    # Create legend, removing duplicates
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        by_label = {}
        for h, l in zip(handles, labels):
            if l not in by_label:
                by_label[l] = h
        ax.legend(by_label.values(), by_label.keys(), frameon=True, fontsize=12, loc='upper right')
    # Add label for feature axes
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    
    # Add iteration title if requested
    if title:
        ax.set_title(title, fontsize=16)
        
    plt.tight_layout()
    fig.canvas.draw()
    image = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
    return image, fig

def create_clustering_frame_3d(df_it, xlim, ylim, zlim, title="", show_errors=True, show_ellipsoid=True, colors=None, legend=True):
    """
    Create a 3d plot for a specific iteration of clustering.
    Parameters:
        df: DataFrame containing clustering data for a specific iteration
        xlim: Tuple (xmin, xmax) for x-axis limits
        ylim: Tuple (ymin, ymax) for y-axis limits
        zlim: Tuple (zmin, zmax) for z-axis limits
        title: Title for the plot
        show_errors: Whether to highlight misclassified points
        show_ellipsoid: Whether to show covariance ellipsoids
        colors: List of colors to use for clusters (default None, uses consistent palette)
    Returns:
        image: Numpy array representing the figure
        fig: Matplotlib figure object
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    # Set axis limit
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)

    unique_clusters = sorted(df_it['predicted_cluster'].unique())
    # Define color map, using the default consistent palette
    if colors is None:
        colors = _get_palette()
    color_map = {c: colors[i % len(colors)] for i, c in enumerate(unique_clusters)}

    for c in unique_clusters:
        # Select data for the current cluster
        data_c = df_it[df_it['predicted_cluster'] == c]
        color = color_map[c]
        
        # Plot the data points for cluster c
        ax.scatter(data_c['feature_1'], data_c['feature_2'], data_c['feature_3'], s=12, label=f'Cluster {c}', color=color, alpha=0.85)
        
        # Plot covariance ellipsoid
        mean = [data_c['mu_k_1'].iloc[0], data_c['mu_k_2'].iloc[0], data_c['mu_k_3'].iloc[0]]
        cov = np.diag([data_c['sigma_k_1'].iloc[0], data_c['sigma_k_2'].iloc[0], data_c['sigma_k_3'].iloc[0]])
        if show_ellipsoid:
            plot_cov_ellipsoids_3d(mean, cov, ax, color, sigmas=(1,2,3))

        # If not showing errors, skip
        if not show_errors:
            continue
        # Highlight misclassified points with 'x' marker
        wrong = data_c[data_c['real_cluster'] != data_c['predicted_cluster']]
        if not wrong.empty:
            ax.scatter(wrong['feature_1'], wrong['feature_2'], wrong['feature_3'], marker='x', s=45, linewidths=2,
                       color=color, alpha=1.0, zorder=10)
        
    if legend:
        # Create legend, removing duplicates
        handles, labels = ax.get_legend_handles_labels()
        by_label = {}
        for h, l in zip(handles, labels):
            if l not in by_label:
                by_label[l] = h
        ax.legend(by_label.values(), by_label.keys(), frameon=True, fontsize=12, loc='upper right')
   
    # Add label for feature axes
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_zlabel("Feature 3")
    
    # Add iteration title if requested
    if title:
        ax.set_title(title, fontsize=16)
        
    plt.tight_layout()
    fig.canvas.draw()
    image = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
    return image, fig
