# pages/learn_more.py
"""
Learn More Page - Model Performance, Feature Importance, and Information
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils.constants import CLUSTERING_FEATURES, CLASSIFICATION_FEATURES

# Page config
st.set_page_config(page_title="Learn More", page_icon="📈", layout="wide")

st.title("📈 Learn More About LoanInsight AI")
st.markdown("### Model Performance, Technology, and Information")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Model Performance", "🔑 Feature Dictionary", "🤖 Technology", "ℹ️ About"])

with tab1:
    st.header("Model Performance Metrics")
    st.write("Performance statistics for our machine learning models")

    # Placeholder metrics (replace with actual metrics)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎯 K-Means Clustering")

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Silhouette Score", "0.68")
        with col_b:
            st.metric("Davies-Bouldin", "0.42")
        with col_c:
            st.metric("Clusters", "5")

        st.markdown("""
        **Model Details:**
        - Algorithm: K-Means
        - Features: 24 financial indicators
        - Standardization: Yes (StandardScaler)
        - Initialization: k-means++

        **Cluster Characteristics:**
        - Cluster 0: Low risk, high income
        - Cluster 1: Moderate risk, balanced profile
        - Cluster 2: Growth potential segment
        - Cluster 3: High utilization
        - Cluster 4: High risk segment
        """)

    with col2:
        st.subheader("✅ XGBoost Classification")

        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.metric("Accuracy", "87.3%")
        with col_b:
            st.metric("Precision", "84.5%")
        with col_c:
            st.metric("Recall", "89.2%")
        with col_d:
            st.metric("F1 Score", "86.8%")

        st.markdown("""
        **Model Details:**
        - Algorithm: XGBoost (Gradient Boosting)
        - Features: 24 financial + engineered features
        - Max Depth: 6
        - Learning Rate: 0.1
        - Risk Threshold: 0.35

        **Training:**
        - Training Samples: ~80,000
        - Validation: 5-fold cross-validation
        - Class Balance: SMOTE applied
        """)

    st.markdown("---")

    # Feature importance chart
    st.subheader("🔑 Top 15 Most Important Features")

    # Dummy feature importance data
    features = [
        'Fico_avg_val', 'dti', 'annual_inc', 'int_rate', 'all_util',
        'sub_grade', 'total_bal_il', 'max_bal_bc', 'bc_util',
        'mths_since_recent_inq', 'num_actv_rev_tl', 'loan_amnt',
        'avg_cur_bal', 'installment', 'mo_sin_old_rev_tl_op'
    ]
    importance = [0.18, 0.15, 0.12, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04, 0.04, 0.03, 0.03, 0.02, 0.02, 0.01]

    fig = go.Figure(go.Bar(
        x=importance,
        y=features,
        orientation='h',
        marker=dict(
            color=importance,
            colorscale='Viridis',
            showscale=True
        )
    ))

    fig.update_layout(
        title='Feature Importance in XGBoost Model',
        xaxis_title='Importance Score',
        yaxis_title='Features',
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("📖 Feature Dictionary")
    st.write("Detailed explanations of all features used in our models")

    # Search box
    search = st.text_input("🔍 Search features", "")

    # Combine all features
    all_features = {}

    # Add clustering features
    for key, value in CLUSTERING_FEATURES.items():
        if 'label' in value:
            all_features[key] = {
                'Feature Name': key,
                'Label': value['label'],
                'Description': value.get('help', 'N/A'),
                'Type': value['type'],
                'Used In': 'Clustering'
            }

    # Add classification-specific features
    classification_only = ['acc_open_past_24mths', 'avg_cur_bal', 'installment',
                           'num_actv_rev_tl', 'mths_since_rcnt_il', 'mths_since_recent_inq',
                           'inq_last_12m', 'mo_sin_old_rev_tl_op', 'int_rate']

    for key in classification_only:
        if key in CLASSIFICATION_FEATURES:
            value = CLASSIFICATION_FEATURES[key]
            if 'label' in value:
                all_features[key] = {
                    'Feature Name': key,
                    'Label': value['label'],
                    'Description': value.get('help', 'N/A'),
                    'Type': value['type'],
                    'Used In': 'Classification'
                }

    # Create dataframe
    feature_df = pd.DataFrame(all_features.values())

    # Apply search filter
    if search:
        feature_df = feature_df[
            feature_df['Feature Name'].str.contains(search, case=False) |
            feature_df['Label'].str.contains(search, case=False) |
            feature_df['Description'].str.contains(search, case=False)
            ]

    # Display
    st.dataframe(feature_df, use_container_width=True, height=600)

    st.markdown("---")
    st.markdown("### 📌 Key Feature Categories")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **Credit Metrics**
        - FICO Score
        - Credit Utilization
        - Credit Inquiries
        - Delinquencies
        """)

    with col2:
        st.markdown("""
        **Financial Position**
        - Annual Income
        - Debt-to-Income Ratio
        - Account Balances
        - Monthly Installments
        """)

    with col3:
        st.markdown("""
        **Loan Details**
        - Loan Amount
        - Interest Rate
        - Loan Term
        - Loan Purpose
        """)

with tab3:
    st.header("🤖 Technology Stack")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎨 Frontend")
        st.markdown("""
        - **Streamlit**: Interactive web framework
        - **Plotly**: Dynamic visualizations
        - **Custom CSS**: Modern UI/UX design
        """)

        st.subheader("🧠 Machine Learning")
        st.markdown("""
        - **scikit-learn**: K-Means clustering
        - **XGBoost**: Gradient boosting classification
        - **SHAP**: Model interpretability
        - **Pandas/NumPy**: Data processing
        """)

    with col2:
        st.subheader("🤖 AI/LLM Integration")
        st.markdown("""
        - **LangChain**: LLM orchestration framework
        - **Groq API**: Primary LLM (Llama 3.1 70B)
        - **Google Gemini**: Fallback LLM
        - **Conversational Agents**: Context-aware chat
        """)

        st.subheader("📦 Deployment")
        st.markdown("""
        - **Python 3.10+**: Core language
        - **Pickle**: Model serialization
        - **python-dotenv**: Environment management
        """)

    st.markdown("---")
    st.subheader("🔄 Architecture Flow")

    st.markdown("""
```
    User Input → Validation → Preprocessing → Model Pipeline → Prediction
                                                                    ↓
    SHAP Analysis ← LLM Explanation ← Feature Importance ← Risk Score
                        ↓
            Conversational AI Agent
```
    """)

    st.markdown("---")
    st.subheader("🔍 Model Explainability")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **SHAP (SHapley Additive exPlanations)**

        SHAP values provide a unified measure of feature importance based on game theory:

        - Shows exact contribution of each feature
        - Works for any machine learning model
        - Provides both global and local explanations
        - Respects feature interactions

        **Benefits:**
        - Understand individual predictions
        - Identify bias in decisions
        - Build trust with transparency
        """)

    with col2:
        st.markdown("""
        **LLM-Generated Explanations**

        We use large language models to translate technical outputs into natural language:

        - Converts SHAP values to plain English
        - Provides personalized recommendations
        - Answers follow-up questions
        - Suggests actionable improvements

        **Implementation:**
        - Primary: Groq (Llama 3.1 70B)
        - Fallback: Google Gemini Pro
        - Context-aware responses
        - Conversation memory
        """)

with tab4:
    st.header("ℹ️ About LoanInsight AI")

    st.markdown("""
    ## 🎯 Project Overview

    LoanInsight AI is an educational demonstration of how machine learning and artificial intelligence 
    can be applied to financial services, specifically in the loan approval process.

    ### 🎓 Purpose

    This application serves as:
    - **Educational Tool**: Learn about ML in finance
    - **Technology Demo**: Showcase modern AI/ML stack
    - **Explainability Example**: Demonstrate transparent AI
    - **User Experience**: Show how to make AI accessible
    """)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 🌟 Key Features

        **Customer Segmentation**
        - K-Means clustering (5 segments)
        - Visual profile comparison
        - Cluster characteristics analysis

        **Loan Eligibility**
        - XGBoost risk prediction
        - SHAP-based explanations
        - Alternative loan suggestions

        **AI Assistance**
        - Conversational agents
        - Personalized advice
        - Natural language explanations
        """)

    with col2:
        st.markdown("""
        ### 🔒 Privacy & Ethics

        **Data Handling**
        - No permanent storage
        - Session-based only
        - No external data sharing

        **Ethical AI**
        - Transparent decisions
        - Explainable predictions
        - Bias awareness

        **Compliance**
        - Educational use only
        - Not for actual lending
        - Disclaimer provided
        """)

    st.markdown("---")

    st.warning("""
    ### ⚠️ Important Disclaimer

    **LoanInsight AI is for educational and demonstration purposes ONLY.**

    This application:
    - ❌ Should NOT be used for actual lending decisions
    - ❌ Does not constitute financial advice
    - ❌ Is not validated for production use
    - ❌ May not reflect real-world lending complexity

    **Before Production Use:**
    - ✅ Validate models with domain experts
    - ✅ Conduct fairness and bias audits
    - ✅ Ensure regulatory compliance
    - ✅ Implement proper security measures
    - ✅ Get legal review

    **Always consult qualified financial professionals for actual lending decisions.**
    """)

    st.markdown("---")

    st.markdown("""
    ### 🚀 Future Enhancements

    Potential improvements for this project:
    - Real-time model updates
    - A/B testing framework
    - Multi-model ensemble
    - Advanced bias detection
    - Regulatory compliance checks
    - User feedback loop
    - Performance monitoring dashboard
    """)

    st.markdown("---")

    st.info("""
    ### 💡 Contributing

    This is an educational project. If you'd like to contribute or learn more:
    - Study the codebase structure
    - Experiment with different models
    - Try alternative LLM providers
    - Improve the UI/UX
    - Add new visualizations
    """)