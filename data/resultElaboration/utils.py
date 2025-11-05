import plotly.graph_objects as go
import plotly.express as px
import numpy as np

def compute_metrics(group):
    """
        Calculate speedup and efficiency for a problem size group
        The function gets the sequential execution time from the row where n_processes == 1
        Computes:
            - speedup as t1 / tN
            - efficiency as speedup / n_processes
    """
    # Get the sequential execution time (1 process)
    t1 = group[group['n_processes'] == 1]['total_execution_time'].iloc[0]
    
    # Calculate speedup and efficiency
    group = group.copy()
    group['speedup'] = t1 / group['total_execution_time']
    group['efficiency'] = group['speedup'] / group['n_processes']
    
    return group

def make_pivot(df, value, index='n_processes', columns='dataset_size',
               aggfunc='mean', round_decimals=3, fillna=None, process_order=None):
    """
        Create a pivot table from the DataFrame with specified parameters.
        Parameters:
            df: Input DataFrame
            value: Column name to aggregate
            index: Column to use as index (default 'n_processes')
            columns: Column to use as columns (default 'dataset_size')
            aggfunc: Aggregation function (default 'mean')
            round_decimals: Number of decimals to round the results (default 3)
            fillna: Value to fill NaNs (default None)
            process_order: List specifying the order of index values (default None, sorts unique index values)
        Returns:
            Pivot table as a DataFrame
    """
    if process_order is None:
        process_order = sorted(df[index].unique())
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

def plot_metrics(filtered_df, metric):
    """
        Create a Speedup figure using Plotly
        Parameters:
            filtered_df: DataFrame filtered for selected datasets
            metric: Metric to plot ('speedup' or 'efficiency')
    """
    if metric not in ['speedup', 'efficiency']:
        raise ValueError("Metric must be 'speedup' or 'efficiency'")
    
    # Color palette
    colors = px.colors.qualitative.Set1
    
    # Create Speedup figure
    fig = go.Figure()

    n_proc = np.sort(filtered_df['n_processes'].unique())
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

    # Plot data for each dataset
    for i, size in enumerate(filtered_df['dataset_size'].unique()):
        subset = filtered_df[filtered_df['dataset_size'] == size]
        label = format_scientific(size)
        fig.add_trace(go.Scatter(
            x=subset['n_processes'],
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