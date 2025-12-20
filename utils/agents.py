# utils/agents.py
"""
Conversational agents for different pages
"""

import streamlit as st
from utils.llm_handler import (
    chat_with_agent,
    generate_improvement_suggestions,
    explain_loan_decision,
    suggest_alternative_loan_amount
)


def initialize_chat_history(agent_key):
    """Initialize chat history for an agent in session state"""
    if agent_key not in st.session_state:
        st.session_state[agent_key] = []


def show_agent_b_clustering(cluster_id, user_inputs, cluster_stats):
    """
    Agent B: Clustering page conversational assistant

    Args:
        cluster_id: User's cluster ID
        user_inputs: User's financial inputs
        cluster_stats: Cluster statistics
    """
    st.markdown("---")
    st.markdown("### 💬 Chat with Your Financial Advisor")
    st.markdown("Ask questions about your cluster or how to improve your financial profile.")

    # Initialize chat history
    initialize_chat_history('agent_b_chat')

    # Generate initial insights button
    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("🎯 Get Improvement Suggestions", use_container_width=True):
            with st.spinner("Generating personalized suggestions..."):
                suggestions, model = generate_improvement_suggestions(
                    user_inputs,
                    cluster_id,
                    cluster_stats
                )

                # Add to chat history
                st.session_state.agent_b_chat.append({
                    'role': 'assistant',
                    'content': suggestions
                })
                st.rerun()

    with col2:
        if st.button("🔄 Clear Chat", use_container_width=True):
            st.session_state.agent_b_chat = []
            st.rerun()

    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.agent_b_chat:
            if message['role'] == 'user':
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown(f"**Advisor:** {message['content']}")

    # Chat input
    user_question = st.text_input(
        "Ask a question:",
        placeholder="e.g., How can I move to a better cluster?",
        key="agent_b_input"
    )

    if st.button("Send", key="agent_b_send"):
        if user_question:
            # Add user message to history
            st.session_state.agent_b_chat.append({
                'role': 'user',
                'content': user_question
            })

            # Get context
            context = {
                'cluster_id': cluster_id,
                'user_inputs': user_inputs,
                'cluster_stats': cluster_stats
            }

            # Get LLM response
            with st.spinner("Thinking..."):
                response, model = chat_with_agent(
                    user_question,
                    context,
                    st.session_state.agent_b_chat[:-1]  # Exclude the question we just added
                )

            # Add assistant response to history
            st.session_state.agent_b_chat.append({
                'role': 'assistant',
                'content': response
            })

            st.rerun()


def show_agent_c_eligibility(is_approved, risk_score, shap_df, user_inputs):
    """
    Agent C: Eligibility page conversational assistant

    Args:
        is_approved: Whether loan was approved
        risk_score: Risk probability
        shap_df: SHAP analysis dataframe
        user_inputs: User's financial inputs
    """
    st.markdown("---")
    st.markdown("### 💬 Chat About Your Loan Decision")

    # Initialize chat history
    initialize_chat_history('agent_c_chat')

    # Quick action buttons
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📝 Explain Decision", use_container_width=True):
            with st.spinner("Generating explanation..."):
                explanation, model = explain_loan_decision(
                    is_approved,
                    risk_score,
                    shap_df,
                    user_inputs
                )

                st.session_state.agent_c_chat.append({
                    'role': 'assistant',
                    'content': explanation
                })
                st.rerun()

    with col2:
        if not is_approved:
            if st.button("💡 Suggest Alternative", use_container_width=True):
                with st.spinner("Calculating alternatives..."):
                    alternative, model = suggest_alternative_loan_amount(
                        user_inputs,
                        risk_score
                    )

                    st.session_state.agent_c_chat.append({
                        'role': 'assistant',
                        'content': alternative
                    })
                    st.rerun()

    with col3:
        if st.button("🔄 Clear Chat", use_container_width=True):
            st.session_state.agent_c_chat = []
            st.rerun()

    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.agent_c_chat:
            if message['role'] == 'user':
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown(f"**AI Advisor:** {message['content']}")

    # Chat input
    user_question = st.text_input(
        "Ask a question:",
        placeholder="e.g., What can I do to get approved?",
        key="agent_c_input"
    )

    if st.button("Send", key="agent_c_send"):
        if user_question:
            # Add user message
            st.session_state.agent_c_chat.append({
                'role': 'user',
                'content': user_question
            })

            # Get context
            context = {
                'is_approved': is_approved,
                'risk_score': risk_score,
                'user_inputs': user_inputs
            }

            # Get LLM response
            with st.spinner("Thinking..."):
                response, model = chat_with_agent(
                    user_question,
                    context,
                    st.session_state.agent_c_chat[:-1]
                )

            # Add response
            st.session_state.agent_c_chat.append({
                'role': 'assistant',
                'content': response
            })

            st.rerun()