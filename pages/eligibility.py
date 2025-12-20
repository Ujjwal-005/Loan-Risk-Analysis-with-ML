# pages/eligibility.py
"""
Loan Eligibility Page - XGBoost Classification with SHAP Explanations
"""

import sys
import custom_functions
sys.modules['__main__'].invert_column = custom_functions.invert_column
sys.modules['__main__'].FeatureSelector = custom_functions.FeatureSelector


import streamlit as st
import pandas as pd
from utils.models import predict_loan_eligibility, get_shap_values
from utils.visualizations import plot_shap_waterfall, plot_shap_bar
from utils.validators import validate_classification_inputs
from utils.agents import show_agent_c_eligibility
from utils.constants import CLASSIFICATION_FEATURES

# Page config
st.set_page_config(page_title="Loan Eligibility", page_icon="✅", layout="wide")

# Track page visit
if 'visited_pages' not in st.session_state:
    st.session_state.visited_pages = set()
st.session_state.visited_pages.add('eligibility')

st.title("✅ Loan Eligibility Check")
st.markdown("### AI-Powered Risk Assessment")
st.write("Enter your information to check loan eligibility with detailed AI explanations.")

# Sidebar form
with st.sidebar:
    st.header("📋 Loan Application")

    with st.form("eligibility_form"):
        st.subheader("💰 Loan Details")

        inputs = {}

        # Loan amount
        inputs['loan_amnt'] = st.number_input(
            "Loan Amount ($)",
            min_value=1000.0,
            max_value=50000.0,
            value=10000.0,
            step=500.0
        )

        # Interest rate
        inputs['int_rate'] = st.slider(
            "Interest Rate (%)",
            min_value=0.0,
            max_value=60.0,
            value=12.0,
            step=0.5
        )

        # Installment
        inputs['installment'] = st.number_input(
            "Monthly Installment ($)",
            min_value=0.0,
            value=300.0,
            step=10.0
        )

        # Sub grade
        sub_grade_options = [
            'A1', 'A2', 'A3', 'A4', 'A5',
            'B1', 'B2', 'B3', 'B4', 'B5',
            'C1', 'C2', 'C3', 'C4', 'C5',
            'D1', 'D2', 'D3', 'D4', 'D5',
            'E1', 'E2', 'E3', 'E4', 'E5',
            'F1', 'F2', 'F3', 'F4', 'F5',
            'G1', 'G2', 'G3', 'G4', 'G5'
        ]
        inputs['sub_grade'] = st.selectbox("Credit Sub-Grade", options=sub_grade_options)

        st.subheader("💼 Employment & Income")

        # Employment length
        inputs['emp_length'] = st.selectbox(
            "Employment Length (years)",
            options=['<1', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10+']
        )

        # Annual income
        inputs['annual_inc'] = st.number_input(
            "Annual Income ($)",
            min_value=10000.0,
            max_value=500000.0,
            value=60000.0,
            step=5000.0
        )

        # Verification status
        inputs['verification_status'] = st.selectbox(
            "Income Verification",
            options=['Verified', 'Not Verified']
        )

        st.subheader("🏠 Property & Purpose")

        # Home ownership
        inputs['home_ownership'] = st.selectbox(
            "Home Ownership",
            options=['MORTGAGE', 'RENT', 'OWN', 'NO_PERMANENT_ADDRESS']
        )

        # Purpose
        inputs['purpose'] = st.selectbox(
            "Loan Purpose",
            options=['debt_consolidation', 'credit_card', 'home_improvement',
                     'major_purchase', 'medical', 'others']
        )

        st.subheader("📈 Financial Metrics")

        # DTI
        inputs['dti'] = st.slider(
            "Debt-to-Income Ratio (%)",
            min_value=0.0,
            max_value=40.0,
            value=15.0,
            step=0.5
        )

        # Delinquencies
        inputs['delinq_2yrs'] = st.selectbox(
            "Delinquencies (Last 2 Years)",
            options=['No', 'Yes']
        )

        # FICO
        inputs['Fico_avg_val'] = st.slider(
            "FICO Credit Score",
            min_value=350,
            max_value=800,
            value=700,
            step=5
        )

        # Credit inquiries
        inputs['inq_last_6mths'] = st.selectbox(
            "Credit Inquiries (Last 6 Months)",
            options=['0', '1', '2', '2+']
        )

        inputs['inq_last_12m'] = st.number_input(
            "Credit Inquiries (Last 12 Months)",
            min_value=0,
            value=1
        )

        st.subheader("💳 Credit Profile")

        # All utilization
        inputs['all_util'] = st.slider(
            "Overall Credit Utilization (%)",
            min_value=0.0,
            max_value=40.0,
            value=25.0,
            step=1.0
        )

        # Total installment balance
        inputs['total_bal_il'] = st.number_input(
            "Total Installment Balance ($)",
            min_value=0.0,
            value=20000.0,
            step=2000.0
        )

        # Max bank card balance
        inputs['max_bal_bc'] = st.number_input(
            "Max Bank Card Balance ($)",
            min_value=0.0,
            value=5000.0,
            step=500.0
        )

        st.subheader("📊 Account History")

        # Accounts opened
        inputs['acc_open_past_24mths'] = st.number_input(
            "Accounts Opened (Last 24 Months)",
            min_value=0,
            value=2
        )

        # Active revolving accounts
        inputs['num_actv_rev_tl'] = st.number_input(
            "Active Revolving Accounts",
            min_value=0,
            value=5
        )

        # Average current balance
        inputs['avg_cur_bal'] = st.number_input(
            "Average Current Balance ($)",
            min_value=0.0,
            value=10000.0,
            step=1000.0
        )

        # Months since recent installment
        inputs['mths_since_rcnt_il'] = st.number_input(
            "Months Since Recent Installment",
            min_value=0,
            value=12
        )

        # Months since recent inquiry
        inputs['mths_since_recent_inq'] = st.number_input(
            "Months Since Recent Inquiry",
            min_value=0,
            value=6
        )

        # Months since oldest revolving account
        inputs['mo_sin_old_rev_tl_op'] = st.number_input(
            "Months Since Oldest Revolving Account",
            min_value=0,
            value=120
        )

        submit_button = st.form_submit_button("✅ Check Eligibility", use_container_width=True)

# Main content
if submit_button:
    # Validate inputs
    validated_inputs, messages = validate_classification_inputs(inputs)

    # Show warnings/errors
    if validated_inputs is None:
        for error in messages:
            st.error(error)
        st.info("Please fix the errors above and try again.")
    else:
        # Show warnings
        for warning in messages:
            st.warning(warning)

        # Predict eligibility
        with st.spinner("🔄 Analyzing your application..."):
            try:
                # Add debug info
                st.write("DEBUG: Validated inputs:", validated_inputs)

                result = predict_loan_eligibility(validated_inputs)

                # Check if result is valid
                if result is None:
                    st.error("❌ Prediction returned None. Check console for errors.")
                else:
                    risk_proba, is_approved = result

                    # Store in session state
                    st.session_state.eligibility_result = {
                        'is_approved': is_approved,
                        'risk_score': risk_proba,
                        'user_inputs': validated_inputs
                    }

                # Display decision
                st.markdown("---")

                if is_approved:
                    st.markdown("""
                    <div style='background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); 
                                padding: 2rem; border-radius: 15px; border-left: 5px solid #10b981;'>
                        <h1 style='color: #065f46; margin: 0;'>✅ APPROVED</h1>
                        <p style='color: #047857; font-size: 1.2rem; margin-top: 0.5rem;'>
                            Congratulations! Your loan application meets our approval criteria.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.balloons()

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Risk Score", f"{risk_proba:.3f}", delta="Low Risk", delta_color="inverse")

                    with col2:
                        # Calculate EMI
                        loan_amt = validated_inputs.get('loan_amnt', 10000)
                        months = 36  # Default term
                        int_rate = validated_inputs.get('int_rate', 12) / 100 / 12
                        if int_rate > 0:
                            emi = loan_amt * int_rate * (1 + int_rate) ** months / ((1 + int_rate) ** months - 1)
                        else:
                            emi = loan_amt / months
                        st.metric("Estimated Monthly EMI", f"${emi:.2f}")

                    with col3:
                        confidence = (1 - risk_proba) * 100
                        st.metric("Confidence", f"{confidence:.1f}%")

                else:
                    st.markdown("""
                    <div style='background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%); 
                                padding: 2rem; border-radius: 15px; border-left: 5px solid #ef4444;'>
                        <h1 style='color: #991b1b; margin: 0;'>❌ NOT APPROVED</h1>
                        <p style='color: #b91c1c; font-size: 1.2rem; margin-top: 0.5rem;'>
                            Your application shows elevated risk factors. See explanation below.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("Risk Score", f"{risk_proba:.3f}", delta="High Risk")

                    with col2:
                        st.metric("Approval Threshold", f"{RISK_THRESHOLD:.3f}")

                st.markdown("---")

                # SHAP Explanation
                st.markdown("### 🔍 AI Explanation - Why This Decision?")
                st.write("Our model analyzed your profile. Here's what influenced the decision:")

                with st.spinner("Calculating feature impacts..."):
                    shap_values, base_value, feature_df = get_shap_values(validated_inputs)

                    # Create SHAP dataframe for display
                    shap_df = pd.DataFrame({
                        'Feature': feature_df.columns,
                        'Your Value': feature_df.iloc[0].values,
                        'SHAP Impact': shap_values[0],
                        'Direction': ['⬆ Increases Risk' if v > 0 else '⬇ Decreases Risk' for v in shap_values[0]]
                    })
                    shap_df['Abs Impact'] = abs(shap_df['SHAP Impact'])
                    shap_df = shap_df.sort_values('Abs Impact', ascending=False)

                # Visualizations
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("#### Top Contributing Factors")
                    fig_bar = plot_shap_bar(shap_values, feature_df, top_n=10)
                    st.plotly_chart(fig_bar, use_container_width=True)

                with col2:
                    st.markdown("#### Detailed Impact Analysis")
                    fig_waterfall = plot_shap_waterfall(shap_values, base_value, feature_df)
                    st.plotly_chart(fig_waterfall, use_container_width=True)

                # SHAP table
                st.markdown("#### 📋 Feature Impact Table")
                display_df = shap_df[['Feature', 'Your Value', 'SHAP Impact', 'Direction']].head(15)
                st.dataframe(display_df, use_container_width=True, height=400)

                # Comfortable loan amount suggestion
                if not is_approved:
                    st.markdown("---")
                    st.markdown("### 💡 Alternative Loan Amount Suggestion")

                    # Calculate safer amount
                    annual_inc = validated_inputs.get('annual_inc', 60000)
                    monthly_inc = annual_inc / 12
                    safe_emi_pct = 0.20  # 20% of monthly income
                    safe_emi = monthly_inc * safe_emi_pct

                    # Calculate loan amount from EMI (36 months, 12% interest)
                    months = 36
                    interest_rate = 0.12 / 12
                    safe_loan = safe_emi * ((1 + interest_rate) ** months - 1) / (
                                interest_rate * (1 + interest_rate) ** months)

                    col1, col2 = st.columns(2)

                    with col1:
                        st.success(f"""
                        **Recommended Loan Amount:** ${safe_loan:,.0f}

                        Based on your annual income of ${annual_inc:,.0f}, a more comfortable 
                        loan amount would be around **${safe_loan:,.0f}** with an approximate 
                        EMI of **${safe_emi:.2f}** per month.

                        This represents about **20%** of your monthly income, keeping you in a 
                        safe financial position.
                        """)

                    with col2:
                        st.info("""
                        **What You Can Do Next:**

                        • **Reduce loan amount** to the suggested level
                        • **Improve FICO score** by making on-time payments
                        • **Lower DTI ratio** by paying down existing debts
                        • **Reduce credit utilization** to below 30%
                        • **Wait 6 months** and reapply with improved profile
                        """)

                    # Quick action buttons
                    st.markdown("### 🎯 Quick Actions")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        if st.button("🔄 Try Suggested Amount", use_container_width=True):
                            st.info(f"Tip: Resubmit the form with loan amount: ${safe_loan:,.0f}")

                    with col2:
                        if st.button("📈 Improve FICO Tips", use_container_width=True):
                            st.info(
                                "• Pay all bills on time\n• Keep credit utilization below 30%\n• Don't close old credit cards\n• Dispute any errors on credit report")

                    with col3:
                        if st.button("💰 Lower DTI Tips", use_container_width=True):
                            st.info(
                                "• Pay down high-interest debt first\n• Avoid taking new loans\n• Consider debt consolidation\n• Increase income if possible")

                # AI Agent
                show_agent_c_eligibility(is_approved, risk_proba, shap_df, validated_inputs)

            except Exception as e:
                st.error(f"❌ Error during analysis: {str(e)}")
                st.info("Please ensure model files are in the 'models/' folder")
                import traceback

                st.code(traceback.format_exc())

else:
    st.info("👈 Complete the form in the sidebar and click 'Check Eligibility' to analyze your application")

    # Preview section
    st.markdown("### 🎯 How Our AI Assessment Works")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 1️⃣ Data Analysis")
        st.write("We analyze 20+ financial factors from your profile using advanced ML algorithms")

    with col2:
        st.markdown("#### 2️⃣ Risk Prediction")
        st.write("XGBoost model evaluates your default risk score with high accuracy")

    with col3:
        st.markdown("#### 3️⃣ AI Explanation")
        st.write("SHAP values show exactly which features influenced the decision and by how much")

    st.markdown("---")
    st.markdown("### ✨ What Makes Us Different")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **🔍 Full Transparency**
        - See exactly why decisions are made
        - Understand which factors help or hurt
        - No black box algorithms
        """)

        st.markdown("""
        **🤖 AI-Powered Advice**
        - Chat with our AI advisor
        - Get personalized recommendations
        - Ask questions about your application
        """)

    with col2:
        st.markdown("""
        **📊 Visual Explanations**
        - Interactive SHAP charts
        - Feature impact analysis
        - Easy-to-understand visualizations
        """)

        st.markdown("""
        **💡 Alternative Solutions**
        - Suggested loan amounts
        - Improvement strategies
        - Reapplication guidance
        """)