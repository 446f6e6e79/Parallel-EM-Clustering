import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment

def compute_metrics(group):
    """
        Calculate speedup and efficiency for a problem size group
        The function gets the sequential execution time from the row where n_processes == 1
        Computes:
            - speedup as t1 / tN
            - efficiency as speedup / n_processes
    """
    # Get the sequential execution time (1 process)
    t1 = group[group['n_process'] == 1]['compute_time'].iloc[0]
    
    # Calculate speedup and efficiency
    group = group.copy()
    group['speedup'] = t1 / group['compute_time']
    group['efficiency'] = group['speedup'] / group['n_process']
    
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

def clustering_accuracy(csv_file):
    """
        Calculate clustering accuracy from a CSV file containing 'predicted' and 'real' columns.
        Uses the Hungarian algorithm to find the best matching between predicted and real labels.
        Parameters:
            csv_file: Path to the CSV file
        Returns:
            accuracy: Clustering accuracy as a float
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)
    y_pred = df['predicted_cluster'].to_numpy()
    y_true = df['real_cluster'].to_numpy()

    # Build confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Hungarian algorithm to maximize accuracy
    row_ind, col_ind = linear_sum_assignment(-cm)
    
    # Final accuracy calculation
    accuracy = cm[row_ind, col_ind].sum() / cm.sum() 
    return accuracy * 100