# streamlit_app.py
"""
LoanInsight AI - Main Entry Point
Multi-page Streamlit app for loan analysis with AI-powered insights
"""
# CRITICAL: Import custom functions FIRST
import custom_functions
import sys

# Register in __main__ for compatibility
sys.modules['__main__'].invert_column = custom_functions.invert_column

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="LoanInsight AI",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize LLMs on app start
from utils.llm_handler import initialize_llms

initialize_llms()

# Initialize session state
if 'visited_pages' not in st.session_state:
    st.session_state.visited_pages = set()

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #0f766e;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #14b8a6;
        text-align: center;
        margin-bottom: 2rem;
    }
    .description {
        font-size: 1.1rem;
        text-align: center;
        color: #64748b;
        margin-bottom: 3rem;
    }
    .feature-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        transition: transform 0.3s;
    }
    .feature-card:hover {
        transform: translateY(-5px);
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #14b8a6 0%, #0f766e 100%);
        color: white;
        border-radius: 10px;
        padding: 1rem;
        font-size: 1.1rem;
        border: none;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s;
        font-weight: 600;
    }
    .stButton>button:hover {
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">💰 LoanInsight AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">AI-Powered Loan Analysis & Customer Segmentation</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="description">Leverage machine learning to understand your financial profile, '
    'discover your customer segment, and check loan eligibility with AI-powered explanations.</div>',
    unsafe_allow_html=True
)
st.success("🚀 Welcome to LoanInsight AI - An end-to-end Machine Learning and Explainable AI project.")
# Feature cards
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📊 Cluster Analysis")
    st.markdown("""
    Discover which customer segment you belong to based on your financial profile.
    Get personalized insights and improvement suggestions from our AI advisor.
    """)
    if st.button("🚀 Explore Clusters", key="btn_cluster"):
        st.switch_page("pages/cluster.py")

with col2:
    st.markdown("### ✅ Loan Eligibility")
    st.markdown("""
    Check your loan eligibility with AI-powered risk assessment. Get detailed 
    SHAP explanations and chat with our AI for personalized advice.
    """)
    if st.button("🚀 Check Eligibility", key="btn_eligibility"):
        st.switch_page("pages/eligibility.py")

with col3:
    st.markdown("### 📈 Learn More")
    st.markdown("""
    Explore model performance, feature importance, and understand the technology 
    behind our AI-powered loan analysis system.
    """)
    if st.button("🚀 Learn More", key="btn_learn"):
        st.switch_page("pages/learn_more.py")

# How it works section
st.markdown("---")
st.markdown("## 🔍 How It Works")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("#### 1️⃣ Enter Data")
    st.markdown("Provide your financial information through our intuitive forms")

with col2:
    st.markdown("#### 2️⃣ AI Analysis")
    st.markdown("Our ML models analyze your data using K-Means and XGBoost")

with col3:
    st.markdown("#### 3️⃣ Get Results")
    st.markdown("Receive instant predictions with visual explanations")

with col4:
    st.markdown("#### 4️⃣ Chat with AI")
    st.markdown("Ask questions and get personalized advice from our AI advisor")

# Key features
st.markdown("---")
st.markdown("## ✨ Key Features")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🤖 AI-Powered Insights")
    st.markdown("""
    - **Groq LLM Integration** with Gemini fallback
    - Conversational AI advisors on each page
    - Natural language explanations of complex decisions
    - Personalized improvement suggestions
    """)

    st.markdown("### 📊 Advanced Analytics")
    st.markdown("""
    - K-Means clustering for customer segmentation
    - XGBoost classification for loan prediction
    - SHAP values for model explainability
    - Interactive Plotly visualizations
    """)

with col2:
    st.markdown("### 💡 Transparent Decisions")
    st.markdown("""
    - Understand exactly why decisions were made
    - See which features impact your score most
    - Compare yourself to cluster averages
    - Get specific recommendations for improvement
    """)

    st.markdown("### 🔒 Privacy & Security")
    st.markdown("""
    - No data stored permanently
    - Session-based analysis only
    - Educational tool for demonstration
    - Not for actual lending decisions
    """)

# Technology stack
st.markdown("---")
st.markdown("## 🛠️ Technology Stack")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("**Frontend**")
    st.markdown("- Streamlit")
    st.markdown("- Plotly")
    st.markdown("- Custom CSS")

with col2:
    st.markdown("**ML Models**")
    st.markdown("- scikit-learn")
    st.markdown("- XGBoost")
    st.markdown("- SHAP")

with col3:
    st.markdown("**AI/LLM**")
    st.markdown("- LangChain")
    st.markdown("- Groq API")
    st.markdown("- Google Gemini")

with col4:
    st.markdown("**Data Processing**")
    st.markdown("- Pandas")
    st.markdown("- NumPy")
    st.markdown("- Python 3.10+")

# Disclaimer
st.markdown("---")
st.warning("""
⚠️ **Educational Tool Disclaimer**

LoanInsight AI is designed for educational and demonstration purposes only. This application:
- Should NOT be used for actual lending decisions
- Does not constitute financial advice
- Uses simplified models for demonstration
- Requires proper validation before production use
- Must comply with all applicable lending regulations

Always consult qualified financial professionals for actual lending decisions.
""")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #94a3b8; font-size: 0.9rem;'>
    Built by Ujjwal Singh • Streamlit • XGBoost • Groq AI • Gemini AI
</div>
""", unsafe_allow_html=True)