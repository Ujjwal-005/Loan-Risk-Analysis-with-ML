# pages/cluster.py
"""
Cluster Analysis Page - K-Means Customer Segmentation
"""

import sys
import custom_functions
sys.modules['__main__'].invert_column = custom_functions.invert_column
sys.modules['__main__'].FeatureSelector = custom_functions.FeatureSelector

import streamlit as st
import pandas as pd
from utils.models import predict_cluster, generate_cluster_statistics
from utils.visualizations import (
    plot_cluster_scatter_2d,
    plot_cluster_comparison_radar,
    plot_feature_comparison_bar,
    plot_cluster_heatmap
)
from utils.validators import validate_clustering_inputs
from utils.agents import show_agent_b_clustering
from utils.constants import CLUSTERING_FEATURES, CLUSTER_DESCRIPTIONS

# Page config
st.set_page_config(page_title="Cluster Analysis", page_icon="📊", layout="wide")

# Track page visit
if 'visited_pages' not in st.session_state:
    st.session_state.visited_pages = set()
st.session_state.visited_pages.add('cluster')

st.title("📊 Customer Segmentation Analysis")
st.markdown("### Discover Your Financial Profile Cluster")
st.write("Enter your financial information to see which customer segment you belong to.")

# Sidebar form
with st.sidebar:
    st.header("📝 Your Financial Profile")

    with st.form("cluster_form"):
        st.subheader("💰 Loan Details")

        inputs = {}

        # Loan amount
        inputs['loan_amnt'] = st.number_input(
            CLUSTERING_FEATURES['loan_amnt']['label'],
            min_value=1000.0,
            max_value=50000.0,
            value=10000.0,
            step=500.0,
            help=CLUSTERING_FEATURES['loan_amnt']['help']
        )

        # Term
        inputs['term'] = st.selectbox(
            CLUSTERING_FEATURES['term']['label'],
            options=CLUSTERING_FEATURES['term']['options'],
            help=CLUSTERING_FEATURES['term']['help']
        )

        # Sub grade
        inputs['sub_grade'] = st.selectbox(
            CLUSTERING_FEATURES['sub_grade']['label'],
            options=CLUSTERING_FEATURES['sub_grade']['options'],
            help=CLUSTERING_FEATURES['sub_grade']['help']
        )

        st.subheader("💼 Employment & Income")

        # Employment length
        inputs['emp_length'] = st.selectbox(
            CLUSTERING_FEATURES['emp_length']['label'],
            options=CLUSTERING_FEATURES['emp_length']['options'],
            help=CLUSTERING_FEATURES['emp_length']['help']
        )

        # Annual income
        inputs['annual_inc'] = st.number_input(
            CLUSTERING_FEATURES['annual_inc']['label'],
            min_value=10000.0,
            max_value=500000.0,
            value=60000.0,
            step=5000.0,
            help=CLUSTERING_FEATURES['annual_inc']['help']
        )

        # Verification status
        inputs['verification_status'] = st.selectbox(
            CLUSTERING_FEATURES['verification_status']['label'],
            options=CLUSTERING_FEATURES['verification_status']['options'],
            help=CLUSTERING_FEATURES['verification_status']['help']
        )

        st.subheader("🏠 Property & Purpose")

        # Home ownership
        inputs['home_ownership'] = st.selectbox(
            CLUSTERING_FEATURES['home_ownership']['label'],
            options=CLUSTERING_FEATURES['home_ownership']['options'],
            help=CLUSTERING_FEATURES['home_ownership']['help']
        )

        # Purpose
        inputs['purpose'] = st.selectbox(
            CLUSTERING_FEATURES['purpose']['label'],
            options=CLUSTERING_FEATURES['purpose']['options'],
            help=CLUSTERING_FEATURES['purpose']['help']
        )

        st.subheader("📈 Financial Metrics")

        # DTI
        inputs['dti'] = st.slider(
            CLUSTERING_FEATURES['dti']['label'],
            min_value=0.0,
            max_value=40.0,
            value=15.0,
            step=0.5,
            help=CLUSTERING_FEATURES['dti']['help']
        )

        # Delinquencies
        inputs['delinq_2yrs'] = st.selectbox(
            CLUSTERING_FEATURES['delinq_2yrs']['label'],
            options=CLUSTERING_FEATURES['delinq_2yrs']['options'],
            help=CLUSTERING_FEATURES['delinq_2yrs']['help']
        )

        # FICO
        inputs['Fico_avg_val'] = st.slider(
            CLUSTERING_FEATURES['Fico_avg_val']['label'],
            min_value=350,
            max_value=800,
            value=700,
            step=5,
            help=CLUSTERING_FEATURES['Fico_avg_val']['help']
        )

        # Public records
        inputs['pub_rec'] = st.selectbox(
            CLUSTERING_FEATURES['pub_rec']['label'],
            options=CLUSTERING_FEATURES['pub_rec']['options'],
            help=CLUSTERING_FEATURES['pub_rec']['help']
        )

        # Credit inquiries
        inputs['inq_last_6mths'] = st.selectbox(
            CLUSTERING_FEATURES['inq_last_6mths']['label'],
            options=CLUSTERING_FEATURES['inq_last_6mths']['options'],
            help=CLUSTERING_FEATURES['inq_last_6mths']['help']
        )

        st.subheader("💳 Credit Utilization")

        # Revolving balance
        inputs['revol_bal'] = st.number_input(
            CLUSTERING_FEATURES['revol_bal']['label'],
            min_value=0.0,
            max_value=100000.0,
            value=5000.0,
            step=500.0,
            help=CLUSTERING_FEATURES['revol_bal']['help']
        )

        # All utilization
        inputs['all_util'] = st.slider(
            CLUSTERING_FEATURES['all_util']['label'],
            min_value=0.0,
            max_value=40.0,
            value=25.0,
            step=1.0,
            help=CLUSTERING_FEATURES['all_util']['help']
        )

        # BC utilization
        inputs['bc_util'] = st.slider(
            CLUSTERING_FEATURES['bc_util']['label'],
            min_value=0.0,
            max_value=40.0,
            value=25.0,
            step=1.0,
            help=CLUSTERING_FEATURES['bc_util']['help']
        )

        st.subheader("📊 Additional Info")

        # Months since derogatory
        inputs['mths_since_last_major_derog'] = st.number_input(
            CLUSTERING_FEATURES['mths_since_last_major_derog']['label'],
            min_value=0,
            max_value=85,
            value=60,
            help=CLUSTERING_FEATURES['mths_since_last_major_derog']['help']
        )

        # Total current balance
        inputs['tot_cur_bal'] = st.number_input(
            CLUSTERING_FEATURES['tot_cur_bal']['label'],
            min_value=0.0,
            max_value=688045.0,
            value=50000.0,
            step=5000.0,
            help=CLUSTERING_FEATURES['tot_cur_bal']['help']
        )

        # Total installment balance
        inputs['total_bal_il'] = st.number_input(
            CLUSTERING_FEATURES['total_bal_il']['label'],
            min_value=0.0,
            value=20000.0,
            step=2000.0,
            help=CLUSTERING_FEATURES['total_bal_il']['help']
        )

        # Max bank card balance
        inputs['max_bal_bc'] = st.number_input(
            CLUSTERING_FEATURES['max_bal_bc']['label'],
            min_value=0.0,
            value=5000.0,
            step=500.0,
            help=CLUSTERING_FEATURES['max_bal_bc']['help']
        )

        # Mortgage accounts
        inputs['mort_acc'] = st.number_input(
            CLUSTERING_FEATURES['mort_acc']['label'],
            min_value=0,
            value=0,
            help=CLUSTERING_FEATURES['mort_acc']['help']
        )

        # Accounts past due
        inputs['num_accts_ever_120_pd'] = st.selectbox(
            CLUSTERING_FEATURES['num_accts_ever_120_pd']['label'],
            options=CLUSTERING_FEATURES['num_accts_ever_120_pd']['options'],
            help=CLUSTERING_FEATURES['num_accts_ever_120_pd']['help']
        )

        # Satisfactory accounts
        inputs['num_sats'] = st.number_input(
            CLUSTERING_FEATURES['num_sats']['label'],
            min_value=0,
            value=10,
            help=CLUSTERING_FEATURES['num_sats']['help']
        )

        # Percent BC > 75
        inputs['percent_bc_gt_75'] = st.slider(
            CLUSTERING_FEATURES['percent_bc_gt_75']['label'],
            min_value=0.0,
            max_value=100.0,
            value=25.0,
            step=1.0,
            help=CLUSTERING_FEATURES['percent_bc_gt_75']['help']
        )

        # Percent never delinquent
        inputs['pct_tl_nvr_dlq'] = st.slider(
            CLUSTERING_FEATURES['pct_tl_nvr_dlq']['label'],
            min_value=0.0,
            max_value=100.0,
            value=90.0,
            step=1.0,
            help=CLUSTERING_FEATURES['pct_tl_nvr_dlq']['help']
        )

        submit_button = st.form_submit_button("🔍 Analyze My Segment", use_container_width=True)

# Main content
if submit_button:
    # Validate inputs
    validated_inputs, messages = validate_clustering_inputs(inputs)

    # Show warnings/errors
    if validated_inputs is None:
        for error in messages:
            st.error(error)
    else:
        # Show warnings
        for warning in messages:
            st.warning(warning)

        # Predict cluster
        with st.spinner("🔄 Analyzing your financial profile..."):
            try:
                cluster_id = predict_cluster(validated_inputs)
                cluster_stats = generate_cluster_statistics(cluster_id)

                # Store in session state
                st.session_state.cluster_result = {
                    'cluster_id': cluster_id,
                    'user_inputs': validated_inputs,
                    'cluster_stats': cluster_stats
                }

                # Display results
                st.success("✅ Analysis Complete!")

                # Cluster info
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.markdown(f"### Your Segment: Cluster {cluster_id}")
                    st.info(CLUSTER_DESCRIPTIONS.get(cluster_id, "Customer segment identified"))

                with col2:
                    st.metric("Cluster ID", cluster_id)
                    st.metric("Default Rate", f"{cluster_stats['default_rate']:.1f}%")

                # Visualizations
                st.markdown("---")
                st.markdown("### 📍 Your Position in Customer Space")

                # Calculate user position (using FICO and DTI as proxies for x, y)
                user_x = (validated_inputs['Fico_avg_val'] - 600) / 50
                user_y = (40 - validated_inputs['dti']) / 5

                fig_scatter = plot_cluster_scatter_2d(cluster_id, (user_x, user_y))
                st.plotly_chart(fig_scatter, use_container_width=True)

                # Comparison charts
                st.markdown("### 📊 Profile Comparison")

                col1, col2 = st.columns(2)

                with col1:
                    fig_radar = plot_cluster_comparison_radar(validated_inputs, cluster_id, cluster_stats)
                    st.plotly_chart(fig_radar, use_container_width=True)

                with col2:
                    fig_bars = plot_feature_comparison_bar(validated_inputs, cluster_stats)
                    st.plotly_chart(fig_bars, use_container_width=True)

                # Heatmap
                st.markdown("### 🔥 Cluster Characteristics Heatmap")
                fig_heatmap = plot_cluster_heatmap()
                st.plotly_chart(fig_heatmap, use_container_width=True)

                # AI Agent
                show_agent_b_clustering(cluster_id, validated_inputs, cluster_stats)

            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                st.info("Please ensure model files are in the 'models/' folder")
                import traceback

                st.code(traceback.format_exc())

else:
    st.info("👈 Fill out the form in the sidebar and click 'Analyze My Segment' to get started")

    # Show preview
    st.markdown("### 🎯 What You'll Discover")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Customer Segment")
        st.write("Find out which of 5 distinct customer segments you belong to based on your financial profile")

    with col2:
        st.markdown("#### Visual Analysis")
        st.write("See interactive charts showing your position and how you compare to others")

    with col3:
        st.markdown("#### AI Guidance")
        st.write("Get personalized recommendations from our AI advisor to improve your profile")