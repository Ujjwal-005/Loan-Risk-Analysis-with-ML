# utils/visualizations.py
"""
Plotly visualization functions
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np


def plot_cluster_scatter_2d(cluster_id, user_point=None):
    """
    Create 2D scatter plot showing user's position among clusters

    Args:
        cluster_id: User's cluster ID
        user_point: Tuple of (x, y) coordinates for user's position

    Returns:
        plotly figure
    """
    # Generate dummy scatter data for visualization
    np.random.seed(42)
    n_samples_per_cluster = 200

    cluster_centers = {
        0: (8, 7),  # Low risk
        1: (5, 5),  # Moderate risk
        2: (3, 6),  # Growth potential
        3: (6, 3),  # High utilization
        4: (2, 2)  # High risk
    }

    colors = {
        0: '#10b981',  # Green
        1: '#fbbf24',  # Yellow
        2: '#60a5fa',  # Blue
        3: '#f97316',  # Orange
        4: '#ef4444'  # Red
    }

    # Generate data for all clusters
    data_points = []
    for cid, center in cluster_centers.items():
        x = np.random.randn(n_samples_per_cluster) * 1.2 + center[0]
        y = np.random.randn(n_samples_per_cluster) * 1.2 + center[1]
        for xi, yi in zip(x, y):
            data_points.append({
                'x': xi,
                'y': yi,
                'cluster': cid,
                'color': colors[cid]
            })

    df = pd.DataFrame(data_points)

    # Create figure
    fig = go.Figure()

    # Add scatter points for each cluster
    for cid in range(5):
        cluster_df = df[df['cluster'] == cid]
        fig.add_trace(go.Scatter(
            x=cluster_df['x'],
            y=cluster_df['y'],
            mode='markers',
            name=f'Cluster {cid}',
            marker=dict(
                size=6,
                color=colors[cid],
                opacity=0.6
            ),
            hovertemplate='Cluster %{fullData.name}<br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
        ))

    # Add user's position
    if user_point:
        fig.add_trace(go.Scatter(
            x=[user_point[0]],
            y=[user_point[1]],
            mode='markers',
            name='Your Position',
            marker=dict(
                size=20,
                color=colors[cluster_id],
                symbol='star',
                line=dict(color='white', width=2)
            ),
            hovertemplate='<b>You Are Here</b><br>Cluster: %{text}<extra></extra>',
            text=[f'{cluster_id}']
        ))

    fig.update_layout(
        title='Customer Segmentation Map',
        xaxis_title='Financial Health Score',
        yaxis_title='Credit Quality Score',
        height=500,
        showlegend=True,
        hovermode='closest',
        plot_bgcolor='rgba(240,240,240,0.5)'
    )

    return fig


def plot_cluster_comparison_radar(user_inputs, cluster_id, cluster_stats):
    """
    Create radar chart comparing user to cluster average

    Args:
        user_inputs: User's financial inputs
        cluster_id: User's cluster ID
        cluster_stats: Average statistics for the cluster

    Returns:
        plotly figure
    """
    # Normalize values for radar chart (0-100 scale)
    categories = ['Income', 'FICO Score', 'DTI (inv)', 'Loan Amount', 'Credit Util (inv)']

    # User values (normalized)
    user_values = [
        min(100, (user_inputs.get('annual_inc', 50000) / 1000)),
        (user_inputs.get('Fico_avg_val', 700) - 300) / 5.5,
        100 - (user_inputs.get('dti', 20) * 2.5),
        min(100, (user_inputs.get('loan_amnt', 10000) / 500)),
        100 - (user_inputs.get('all_util', 30) * 2.5)
    ]

    # Cluster average values (normalized)
    cluster_values = [
        min(100, (cluster_stats['avg_income'] / 1000)),
        (cluster_stats['avg_fico'] - 300) / 5.5,
        100 - (cluster_stats['avg_dti'] * 2.5),
        min(100, (cluster_stats['avg_loan_amount'] / 500)),
        100 - (50 * 2.5)  # Dummy credit util
    ]

    fig = go.Figure()

    # Add cluster average
    fig.add_trace(go.Scatterpolar(
        r=cluster_values,
        theta=categories,
        fill='toself',
        name=f'Cluster {cluster_id} Average',
        line=dict(color='lightblue')
    ))

    # Add user
    fig.add_trace(go.Scatterpolar(
        r=user_values,
        theta=categories,
        fill='toself',
        name='Your Profile',
        line=dict(color='red')
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=True,
        title='Your Profile vs Cluster Average',
        height=450
    )

    return fig


def plot_feature_comparison_bar(user_inputs, cluster_stats):
    """
    Bar chart comparing user values to cluster averages

    Args:
        user_inputs: User's inputs
        cluster_stats: Cluster statistics

    Returns:
        plotly figure
    """
    features = ['Annual Income', 'FICO Score', 'DTI Ratio', 'Loan Amount']
    user_vals = [
        user_inputs.get('annual_inc', 50000),
        user_inputs.get('Fico_avg_val', 700),
        user_inputs.get('dti', 20),
        user_inputs.get('loan_amnt', 10000)
    ]
    cluster_vals = [
        cluster_stats['avg_income'],
        cluster_stats['avg_fico'],
        cluster_stats['avg_dti'],
        cluster_stats['avg_loan_amount']
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=features,
        y=user_vals,
        name='Your Values',
        marker_color='indianred'
    ))
    fig.add_trace(go.Bar(
        x=features,
        y=cluster_vals,
        name='Cluster Average',
        marker_color='lightsalmon'
    ))

    fig.update_layout(
        title='Feature Comparison',
        barmode='group',
        height=400,
        yaxis_title='Value'
    )

    return fig


def plot_shap_waterfall(shap_values, base_value, feature_df):
    """
    Create SHAP waterfall plot

    Args:
        shap_values: SHAP values array
        base_value: Base prediction value
        feature_df: DataFrame with feature values

    Returns:
        plotly figure
    """
    # Get top 10 features by absolute SHAP value
    feature_names = feature_df.columns.tolist()
    shap_vals = shap_values[0] if len(shap_values.shape) > 1 else shap_values

    # Create dataframe for sorting
    shap_df = pd.DataFrame({
        'feature': feature_names,
        'shap_value': shap_vals,
        'feature_value': feature_df.iloc[0].values
    })
    shap_df['abs_shap'] = abs(shap_df['shap_value'])
    shap_df = shap_df.sort_values('abs_shap', ascending=False).head(10)

    # Create waterfall
    fig = go.Figure(go.Waterfall(
        name="SHAP",
        orientation="v",
        measure=["relative"] * len(shap_df) + ["total"],
        x=shap_df['feature'].tolist() + ["Final Prediction"],
        y=shap_df['shap_value'].tolist() + [base_value + shap_vals.sum()],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": "#ef4444"}},
        increasing={"marker": {"color": "#10b981"}},
        totals={"marker": {"color": "#3b82f6"}}
    ))

    fig.update_layout(
        title="SHAP Waterfall Plot - Feature Impact on Prediction",
        height=500,
        showlegend=False,
        xaxis_title="Features",
        yaxis_title="SHAP Value Impact"
    )

    return fig


def plot_shap_bar(shap_values, feature_df, top_n=10):
    """
    Create horizontal bar chart of SHAP values

    Args:
        shap_values: SHAP values array
        feature_df: DataFrame with feature values
        top_n: Number of top features to show

    Returns:
        plotly figure
    """
    feature_names = feature_df.columns.tolist()
    shap_vals = shap_values[0] if len(shap_values.shape) > 1 else shap_values

    # Create dataframe
    shap_df = pd.DataFrame({
        'feature': feature_names,
        'shap_value': shap_vals
    })
    shap_df['abs_shap'] = abs(shap_df['shap_value'])
    shap_df = shap_df.sort_values('abs_shap', ascending=True).tail(top_n)

    # Create colors based on positive/negative
    colors = ['#ef4444' if val > 0 else '#10b981' for val in shap_df['shap_value']]

    fig = go.Figure(go.Bar(
        x=shap_df['shap_value'],
        y=shap_df['feature'],
        orientation='h',
        marker=dict(color=colors),
        text=[f"{val:.3f}" for val in shap_df['shap_value']],
        textposition='outside'
    ))

    fig.update_layout(
        title=f'Top {top_n} Features by Impact',
        xaxis_title='SHAP Value (Impact on Prediction)',
        yaxis_title='Features',
        height=400,
        showlegend=False
    )

    return fig


def plot_cluster_heatmap():
    """
    Create heatmap showing feature averages across clusters

    Returns:
        plotly figure
    """
    # Dummy heatmap data
    features = ['Income', 'FICO', 'DTI', 'Loan Amt', 'Util %']
    clusters = ['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4']

    # Normalized values (0-100 scale)
    data = [
        [85, 90, 15, 75, 20],  # Cluster 0
        [65, 70, 35, 60, 45],  # Cluster 1
        [55, 65, 45, 50, 55],  # Cluster 2
        [48, 55, 60, 40, 70],  # Cluster 3
        [38, 45, 75, 30, 85]  # Cluster 4
    ]

    fig = go.Figure(data=go.Heatmap(
        z=data,
        x=features,
        y=clusters,
        colorscale='RdYlGn',
        reversescale=False,
        text=data,
        texttemplate='%{text}',
        textfont={"size": 12},
        colorbar=dict(title="Normalized Value")
    ))

    fig.update_layout(
        title='Cluster Characteristics Heatmap',
        xaxis_title='Financial Features',
        yaxis_title='Customer Segments',
        height=400
    )

    return fig