# utils/llm_handler.py
"""
LLM integration with Groq (primary) and Gemini (fallback)
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Load environment variables
load_dotenv()

# Global LLM instances
_primary_llm = None
_fallback_llm = None


def initialize_llms():
    """Initialize LLM instances with fallback"""
    global _primary_llm, _fallback_llm

    # Primary: Groq
    groq_api_key = os.getenv('GROQ_API_KEY')
    if groq_api_key:
        try:
            _primary_llm = ChatGroq(
                model="llama-3.1-70b-versatile",  # or "mixtral-8x7b-32768"
                temperature=0.7,
                groq_api_key=groq_api_key,
                max_tokens=2000
            )
            print("✅ Groq LLM initialized")
        except Exception as e:
            print(f"⚠️ Groq initialization failed: {e}")

    # Fallback: Gemini
    google_api_key = os.getenv('GOOGLE_API_KEY')
    if google_api_key:
        try:
            _fallback_llm = ChatGoogleGenerativeAI(
                model="gemini-pro",
                temperature=0.7,
                google_api_key=google_api_key,
                max_output_tokens=2000
            )
            print("✅ Gemini LLM initialized as fallback")
        except Exception as e:
            print(f"⚠️ Gemini initialization failed: {e}")


def get_llm_with_fallback():
    """
    Get LLM with automatic fallback from Groq to Gemini

    Returns:
        LLM instance
    """
    if _primary_llm is None and _fallback_llm is None:
        initialize_llms()

    # Try primary first
    if _primary_llm is not None:
        return _primary_llm, "Groq"

    # Fallback to Gemini
    if _fallback_llm is not None:
        return _fallback_llm, "Gemini"

    raise Exception("No LLM available. Please check your API keys in .env file")


def call_llm(system_prompt, user_message, chat_history=None):
    """
    Call LLM with automatic fallback

    Args:
        system_prompt: System instruction for the LLM
        user_message: User's message
        chat_history: List of previous messages (optional)

    Returns:
        tuple: (response_text, model_used)
    """
    llm, model_name = get_llm_with_fallback()

    # Build messages
    messages = [SystemMessage(content=system_prompt)]

    # Add chat history if available
    if chat_history:
        for msg in chat_history:
            if msg['role'] == 'user':
                messages.append(HumanMessage(content=msg['content']))
            elif msg['role'] == 'assistant':
                messages.append(AIMessage(content=msg['content']))

    # Add current user message
    messages.append(HumanMessage(content=user_message))

    try:
        # Try primary LLM
        response = llm.invoke(messages)
        return response.content, model_name

    except Exception as e:
        print(f"⚠️ Primary LLM ({model_name}) failed: {e}")

        # Try fallback
        if model_name == "Groq" and _fallback_llm is not None:
            try:
                print("🔄 Switching to Gemini fallback...")
                response = _fallback_llm.invoke(messages)
                return response.content, "Gemini (Fallback)"
            except Exception as fallback_error:
                print(f"❌ Fallback LLM also failed: {fallback_error}")
                return "I'm having trouble connecting to the AI service. Please try again later.", "Error"

        return "I'm having trouble processing your request. Please try again.", "Error"


def stream_llm_response(system_prompt, user_message, chat_history=None):
    """
    Stream LLM response for real-time display (optional for better UX)

    Args:
        system_prompt: System instruction
        user_message: User's message
        chat_history: Previous messages

    Yields:
        str: Chunks of response text
    """
    llm, model_name = get_llm_with_fallback()

    # Build messages
    messages = [SystemMessage(content=system_prompt)]

    if chat_history:
        for msg in chat_history:
            if msg['role'] == 'user':
                messages.append(HumanMessage(content=msg['content']))
            elif msg['role'] == 'assistant':
                messages.append(AIMessage(content=msg['content']))

    messages.append(HumanMessage(content=user_message))

    try:
        for chunk in llm.stream(messages):
            if hasattr(chunk, 'content'):
                yield chunk.content
    except Exception as e:
        print(f"Streaming error: {e}")
        yield "Error streaming response."


def generate_improvement_suggestions(user_inputs, cluster_id, cluster_stats):
    """
    Generate personalized improvement suggestions using LLM

    Args:
        user_inputs: User's financial data
        cluster_id: Assigned cluster
        cluster_stats: Cluster statistics

    Returns:
        str: LLM-generated suggestions
    """
    from utils.prompts import get_clustering_improvement_prompt

    system_prompt = get_clustering_improvement_prompt()

    user_message = f"""
    User Profile:
    - Cluster: {cluster_id}
    - Annual Income: ${user_inputs.get('annual_inc', 0):,.0f}
    - FICO Score: {user_inputs.get('Fico_avg_val', 0)}
    - DTI Ratio: {user_inputs.get('dti', 0):.1f}%
    - Loan Amount: ${user_inputs.get('loan_amnt', 0):,.0f}
    - Credit Utilization: {user_inputs.get('all_util', 0):.1f}%
    - Delinquencies: {user_inputs.get('delinq_2yrs', 'No')}

    Cluster Average:
    - Income: ${cluster_stats['avg_income']:,.0f}
    - FICO: {cluster_stats['avg_fico']}
    - DTI: {cluster_stats['avg_dti']:.1f}%
    - Default Rate: {cluster_stats['default_rate']:.1f}%

    Provide 3-5 specific, actionable recommendations to improve their financial profile.
    """

    response, model = call_llm(system_prompt, user_message)
    return response, model


def explain_loan_decision(is_approved, risk_score, shap_df, user_inputs):
    """
    Generate natural language explanation for loan decision

    Args:
        is_approved: Whether loan was approved
        risk_score: Risk probability score
        shap_df: DataFrame with SHAP values
        user_inputs: User's inputs

    Returns:
        str: LLM-generated explanation
    """
    from utils.prompts import get_loan_explanation_prompt

    system_prompt = get_loan_explanation_prompt()

    # Get top influencing features
    top_features = shap_df.head(5)
    features_text = "\n".join([
        f"- {row['Feature']}: {row['Your Value']} (Impact: {row['SHAP Impact']:.3f})"
        for _, row in top_features.iterrows()
    ])

    decision = "APPROVED" if is_approved else "NOT APPROVED"

    user_message = f"""
    Loan Decision: {decision}
    Risk Score: {risk_score:.3f}
    Threshold: 0.35

    Top Contributing Features:
    {features_text}

    User Profile Summary:
    - Income: ${user_inputs.get('annual_inc', 0):,.0f}
    - FICO: {user_inputs.get('Fico_avg_val', 0)}
    - DTI: {user_inputs.get('dti', 0):.1f}%
    - Loan Amount: ${user_inputs.get('loan_amnt', 0):,.0f}

    Explain this decision in 2-3 paragraphs in simple, friendly language.
    """

    response, model = call_llm(system_prompt, user_message)
    return response, model


def suggest_alternative_loan_amount(user_inputs, current_risk_score):
    """
    Suggest a safer loan amount using LLM

    Args:
        user_inputs: User's financial data
        current_risk_score: Current risk score

    Returns:
        str: LLM-generated suggestion
    """
    from utils.prompts import get_loan_alternative_prompt

    system_prompt = get_loan_alternative_prompt()

    user_message = f"""
    Current Application:
    - Loan Amount: ${user_inputs.get('loan_amnt', 0):,.0f}
    - Annual Income: ${user_inputs.get('annual_inc', 0):,.0f}
    - Risk Score: {current_risk_score:.3f}
    - DTI: {user_inputs.get('dti', 0):.1f}%
    - FICO: {user_inputs.get('Fico_avg_val', 0)}

    The loan was rejected. Suggest:
    1. A safer loan amount (specific number)
    2. Why this amount is better
    3. Expected monthly payment
    4. Alternative options
    """

    response, model = call_llm(system_prompt, user_message)
    return response, model


def chat_with_agent(user_question, context, chat_history=None):
    """
    General chat function for conversational agents

    Args:
        user_question: User's question
        context: Context about current analysis (cluster/loan decision)
        chat_history: Previous conversation messages

    Returns:
        str: LLM response
    """
    from utils.prompts import get_agent_chat_prompt

    system_prompt = get_agent_chat_prompt(context)

    response, model = call_llm(system_prompt, user_question, chat_history)
    return response, model