# utils/models.py
"""
Model loading and preprocessing functions
"""

# CRITICAL: Import custom functions BEFORE loading pickles
import sys
import custom_functions

# Register in __main__ so pickle can find it
sys.modules['__main__'].invert_column = custom_functions.invert_column
sys.modules['__main__'].FeatureSelector = custom_functions.FeatureSelector

import pandas as pd
import numpy as np
import joblib
import os
from utils.constants import CLUSTERING_FEATURES, CLASSIFICATION_FEATURES

# Global cache for loaded models
_kmeans_pipeline = None
_xgb_pipeline = None
_shap_explainer = None


def load_kmeans_pipeline():
    """Load K-Means clustering pipeline from pickle file"""
    global _kmeans_pipeline

    if _kmeans_pipeline is None:
        model_path = 'models/loan_kmeans_model_pipeline.pkl'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        _kmeans_pipeline = joblib.load(model_path)
        print("✅ K-Means pipeline loaded successfully")

    return _kmeans_pipeline


def load_xgb_pipeline():
    """Load XGBoost classification pipeline from pickle file"""
    global _xgb_pipeline

    if _xgb_pipeline is None:
        model_path = 'models/loan_xgb_model_pipeline.pkl'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        _xgb_pipeline = joblib.load(model_path)
        print("✅ XGBoost pipeline loaded successfully")

    return _xgb_pipeline


def load_shap_explainer():
    """Load SHAP explainer from pickle file"""
    global _shap_explainer

    if _shap_explainer is None:
        model_path = 'models/shap_explainer.pkl'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"SHAP explainer file not found: {model_path}")

        _shap_explainer = joblib.load(model_path)
        print("✅ SHAP explainer loaded successfully")

    return _shap_explainer


def preprocess_clustering_input(validated_inputs):
    """
    Convert validated inputs to DataFrame for clustering pipeline
    MUST match exact dtypes from training

    Args:
        validated_inputs: Dictionary of validated user inputs

    Returns:
        pd.DataFrame: Single-row dataframe ready for model prediction
    """
    # Create a dictionary with proper dtypes
    data = {
        'loan_amnt': float(validated_inputs['loan_amnt']),
        'term': str(validated_inputs['term']),  # object
        'sub_grade': str(validated_inputs['sub_grade']),  # object
        'emp_length': str(validated_inputs['emp_length']),  # object
        'home_ownership': str(validated_inputs['home_ownership']),  # object
        'annual_inc': float(validated_inputs['annual_inc']),
        'verification_status': str(validated_inputs['verification_status']),  # object
        'purpose': str(validated_inputs['purpose']),  # object
        'dti': float(validated_inputs['dti']),
        'delinq_2yrs': validated_inputs['delinq_2yrs'],  # Will convert to category
        'Fico_avg_val': float(validated_inputs['Fico_avg_val']),
        'pub_rec': validated_inputs['pub_rec'],  # Will convert to category
        'inq_last_6mths': validated_inputs['inq_last_6mths'],  # Will convert to category
        'revol_bal': float(validated_inputs['revol_bal']),
        'mths_since_last_major_derog': float(validated_inputs['mths_since_last_major_derog']),
        'tot_cur_bal': float(validated_inputs['tot_cur_bal']),
        'total_bal_il': float(validated_inputs['total_bal_il']),
        'max_bal_bc': float(validated_inputs['max_bal_bc']),
        'all_util': float(validated_inputs['all_util']),
        'bc_util': float(validated_inputs['bc_util']),
        'mort_acc': float(validated_inputs['mort_acc']),
        'num_accts_ever_120_pd': validated_inputs['num_accts_ever_120_pd'],  # Will convert to category
        'num_sats': float(validated_inputs['num_sats']),
        'percent_bc_gt_75': float(validated_inputs['percent_bc_gt_75']),
        'pct_tl_nvr_dlq': float(validated_inputs['pct_tl_nvr_dlq'])
    }

    # Create DataFrame
    df = pd.DataFrame([data])

    # Convert to category dtype where needed
    df['delinq_2yrs'] = pd.Categorical(df['delinq_2yrs'])
    df['pub_rec'] = pd.Categorical(df['pub_rec'])
    df['inq_last_6mths'] = pd.Categorical(df['inq_last_6mths'])

    # Note: num_accts_ever_120_pd should remain as category
    df['num_accts_ever_120_pd'] = pd.Categorical(df['num_accts_ever_120_pd'])

    return df


def preprocess_classification_input(validated_inputs):
    """
    Convert validated inputs to DataFrame for classification pipeline
    Pipeline expects ALL 72 columns with proper dtypes (no 'object' allowed by XGBoost)

    Args:
        validated_inputs: Dictionary of validated user inputs

    Returns:
        pd.DataFrame: Single-row dataframe with all 72 columns and correct dtypes
    """

    # Helper function to safely get values
    def safe_get(key, default=0.0):
        return validated_inputs.get(key, default)

    # Helper to convert to float safely
    def to_float(value):
        if isinstance(value, str):
            return float(value.replace('+', '').replace('Yes', '1').replace('No', '0'))
        return float(value)

    # Create DataFrame with ALL 72 columns from training data
    # CRITICAL: All numeric values must be float, all categorical must be category
    data = {
        # Collected from form - ensure float dtype
        'loan_amnt': to_float(safe_get('loan_amnt', 10000.0)),
        'term': str(safe_get('term', '36')),  # Will be categorical
        'int_rate': to_float(safe_get('int_rate', 12.0)),
        'installment': to_float(safe_get('installment', 300.0)),
        'sub_grade': str(safe_get('sub_grade', 'C3')),  # Will be categorical
        'emp_length': str(safe_get('emp_length', '5')),  # Will be categorical
        'home_ownership': str(safe_get('home_ownership', 'RENT')),  # Will be categorical
        'annual_inc': to_float(safe_get('annual_inc', 60000.0)),
        'verification_status': str(safe_get('verification_status', 'Not Verified')),  # Will be categorical
        'purpose': str(safe_get('purpose', 'debt_consolidation')),  # Will be categorical
        'dti': to_float(safe_get('dti', 15.0)),
        'delinq_2yrs': to_float(1.0 if safe_get('delinq_2yrs', 'No') == 'Yes' else 0.0),
        'inq_last_6mths': to_float(safe_get('inq_last_6mths', '0').replace('+', '')),
        'Fico_avg_val': to_float(safe_get('Fico_avg_val', 700.0)),
        'total_bal_il': to_float(safe_get('total_bal_il', 20000.0)),
        'max_bal_bc': to_float(safe_get('max_bal_bc', 5000.0)),
        'all_util': to_float(safe_get('all_util', 25.0)),
        'acc_open_past_24mths': to_float(safe_get('acc_open_past_24mths', 2.0)),
        'avg_cur_bal': to_float(safe_get('avg_cur_bal', 10000.0)),
        'num_actv_rev_tl': to_float(safe_get('num_actv_rev_tl', 5.0)),
        'mths_since_rcnt_il': to_float(safe_get('mths_since_rcnt_il', 12.0)),
        'mths_since_recent_inq': to_float(safe_get('mths_since_recent_inq', 6.0)),
        'inq_last_12m': to_float(safe_get('inq_last_12m', 1.0)),
        'mo_sin_old_rev_tl_op': to_float(safe_get('mo_sin_old_rev_tl_op', 120.0)),

        # Not collected - use reasonable defaults (all must be float)
        'open_acc': 10.0,
        'pub_rec': 0.0,
        'revol_bal': 5000.0,
        'revol_util': 30.0,
        'collections_12_mths_ex_med': 0.0,
        'mths_since_last_major_derog': 60.0,
        'acc_now_delinq': 0.0,
        'tot_cur_bal': 50000.0,
        'open_acc_6m': 1.0,
        'open_act_il': 3.0,
        'open_il_12m': 1.0,
        'open_il_24m': 2.0,
        'il_util': 50.0,
        'open_rv_12m': 2.0,
        'open_rv_24m': 3.0,
        'total_rev_hi_lim': 20000.0,
        'inq_fi': 1.0,
        'total_cu_tl': 5.0,
        'bc_open_to_buy': 3000.0,
        'bc_util': 40.0,
        'chargeoff_within_12_mths': 0.0,
        'delinq_amnt': 0.0,
        'mo_sin_old_il_acct': 100.0,
        'mo_sin_rcnt_rev_tl_op': 10.0,
        'mo_sin_rcnt_tl': 8.0,
        'mort_acc': 0.0,
        'mths_since_recent_bc': 15.0,
        'mths_since_recent_bc_dlq': 60.0,
        'mths_since_recent_revol_delinq': 60.0,
        'num_accts_ever_120_pd': 0.0,
        'num_actv_bc_tl': 3.0,
        'num_bc_sats': 4.0,
        'num_bc_tl': 5.0,
        'num_il_tl': 6.0,
        'num_op_rev_tl': 7.0,
        'num_rev_accts': 10.0,
        'num_sats': 10.0,
        'num_tl_120dpd_2m': 0.0,
        'num_tl_30dpd': 0.0,
        'num_tl_90g_dpd_24m': 0.0,
        'num_tl_op_past_12m': 2.0,
        'pct_tl_nvr_dlq': 90.0,
        'percent_bc_gt_75': 20.0,
        'pub_rec_bankruptcies': 0.0,
        'tax_liens': 0.0,
        'total_bal_ex_mort': 30000.0,
        'total_bc_limit': 15000.0,
        'inactive_acc': 2.0
    }

    # Create DataFrame
    df = pd.DataFrame([data])

    # Explicitly set dtypes for categorical columns (based on your training data)
    # These are the ones that were 'category' in training
    category_float_columns = [
        'collections_12_mths_ex_med', 'acc_now_delinq', 'chargeoff_within_12_mths',
        'num_tl_120dpd_2m', 'num_tl_30dpd', 'num_tl_90g_dpd_24m',
        'pub_rec_bankruptcies', 'tax_liens', 'delinq_2yrs', 'pub_rec', 'inq_last_6mths'
    ]

    # Convert these to category dtype
    for col in category_float_columns:
        if col in df.columns:
            df[col] = pd.Categorical(df[col].astype(float))

    # Object columns that should remain as object (for pipeline encoders)
    object_columns = ['term', 'sub_grade', 'emp_length', 'home_ownership',
                      'verification_status', 'purpose']

    # All other columns should be float
    for col in df.columns:
        if col not in category_float_columns and col not in object_columns:
            df[col] = df[col].astype(float)

    return df


def predict_cluster(validated_inputs):
    """
    Predict cluster for user inputs

    Args:
        validated_inputs: Dictionary of validated inputs

    Returns:
        int: Cluster ID (0-4)
    """
    pipeline = load_kmeans_pipeline()
    df = preprocess_clustering_input(validated_inputs)
    cluster_id = pipeline.predict(df)[0]
    return int(cluster_id)


def predict_loan_eligibility(validated_inputs):
    """
    Predict loan approval probability

    Args:
        validated_inputs: Dictionary of validated inputs

    Returns:
        tuple: (probability_of_default, is_approved)
    """
    try:
        pipeline = load_xgb_pipeline()
        df = preprocess_classification_input(validated_inputs)

        # DEBUG: Print dtypes to see what's wrong
        print("\n=== DEBUG: Input DataFrame ===")
        print(f"Shape: {df.shape}")
        print(f"\nDtypes:\n{df.dtypes}")
        print(f"\nFirst row:\n{df.iloc[0]}")

        # Get probability of default (class 1)
        proba = pipeline.predict_proba(df)[0, 1]

        from utils.constants import RISK_THRESHOLD
        is_approved = proba < RISK_THRESHOLD

        return float(proba), is_approved

    except Exception as e:
        print(f"\n=== ERROR in predict_loan_eligibility ===")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        # Re-raise so we can see it
        raise


def get_shap_values(validated_inputs):
    """
    Calculate SHAP values for classification prediction

    Args:
        validated_inputs: Dictionary of validated inputs

    Returns:
        tuple: (shap_values_array, base_value, feature_dataframe)
    """
    explainer = load_shap_explainer()
    df = preprocess_classification_input(validated_inputs)

    # Calculate SHAP values
    shap_values = explainer.shap_values(df)

    # Handle different SHAP output formats
    if isinstance(shap_values, list):
        # Binary classification returns list of arrays
        shap_values = shap_values[1]  # Get positive class

    # Get base value
    base_value = explainer.expected_value
    if isinstance(base_value, np.ndarray):
        base_value = base_value[1]

    return shap_values, base_value, df


def generate_cluster_statistics(cluster_id):
    """
    Generate dummy statistics for cluster comparison
    This should be replaced with actual statistics from training data

    Args:
        cluster_id: The cluster ID (0-4)

    Returns:
        dict: Statistics for the cluster
    """
    # Placeholder statistics - replace with actual data
    cluster_stats = {
        0: {
            'avg_income': 85000,
            'avg_fico': 740,
            'avg_dti': 12,
            'avg_loan_amount': 15000,
            'default_rate': 2.5
        },
        1: {
            'avg_income': 65000,
            'avg_fico': 690,
            'avg_dti': 18,
            'avg_loan_amount': 12000,
            'default_rate': 8.5
        },
        2: {
            'avg_income': 55000,
            'avg_fico': 670,
            'avg_dti': 22,
            'avg_loan_amount': 10000,
            'default_rate': 12.0
        },
        3: {
            'avg_income': 48000,
            'avg_fico': 640,
            'avg_dti': 28,
            'avg_loan_amount': 8000,
            'default_rate': 18.5
        },
        4: {
            'avg_income': 38000,
            'avg_fico': 600,
            'avg_dti': 35,
            'avg_loan_amount': 6000,
            'default_rate': 28.0
        }
    }

    return cluster_stats.get(cluster_id, cluster_stats[1])