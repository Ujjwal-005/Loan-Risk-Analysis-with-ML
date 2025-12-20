# utils/constants.py
"""
Constants and configuration for Loan_class_inator
"""

# Risk threshold for loan approval
RISK_THRESHOLD = 0.35

# ============================================================================
# CLUSTERING FEATURE CONFIGURATION
# ============================================================================
CLUSTERING_FEATURES = {
    'loan_amnt': {
        'type': 'numeric',
        'min': 0,
        'max': None,
        'allow_negative': False,
        'label': 'Loan Amount ($)',
        'help': 'The amount you want to borrow'
    },
    'term': {
        'type': 'categorical',
        'options': [36, 60],
        'label': 'Loan Term (months)',
        'help': 'Choose 36 or 60 months'
    },
    'sub_grade': {
        'type': 'categorical',
        'options': [
            'A1', 'A2', 'A3', 'A4', 'A5',
            'B1', 'B2', 'B3', 'B4', 'B5',
            'C1', 'C2', 'C3', 'C4', 'C5',
            'D1', 'D2', 'D3', 'D4', 'D5',
            'E1', 'E2', 'E3', 'E4', 'E5',
            'F1', 'F2', 'F3', 'F4', 'F5',
            'G1', 'G2', 'G3', 'G4', 'G5'
        ],
        'label': 'Credit Sub-Grade',
        'help': 'Your assigned credit grade'
    },
    'emp_length': {
        'type': 'categorical',
        'options': ['<1', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10+'],
        'label': 'Employment Length (years)',
        'help': 'Years at current job'
    },
    'home_ownership': {
        'type': 'categorical',
        'options': ['MORTGAGE', 'RENT', 'OWN', 'NO_PERMANENT_ADDRESS'],
        'label': 'Home Ownership Status',
        'help': 'Your current living situation'
    },
    'annual_inc': {
        'type': 'numeric',
        'min': 0,
        'max': None,
        'allow_negative': False,
        'label': 'Annual Income ($)',
        'help': 'Your yearly income'
    },
    'verification_status': {
        'type': 'categorical',
        'options': ['Verified', 'Not Verified'],
        'label': 'Income Verification',
        'help': 'Has your income been verified?'
    },
    'purpose': {
        'type': 'categorical',
        'options': ['debt_consolidation', 'credit_card', 'home_improvement',
                    'major_purchase', 'medical', 'others'],
        'label': 'Loan Purpose',
        'help': 'What will you use the loan for?'
    },
    'dti': {
        'type': 'numeric',
        'min': 0,
        'max': 40,
        'allow_negative': False,
        'label': 'Debt-to-Income Ratio (%)',
        'help': 'Monthly debt payments / monthly income'
    },
    'delinq_2yrs': {
        'type': 'categorical',
        'options': ['No', 'Yes'],
        'label': 'Delinquencies in Last 2 Years',
        'help': 'Any late payments in past 2 years?'
    },
    'Fico_avg_val': {
        'type': 'numeric',
        'min': 350,
        'max': 800,
        'allow_negative': False,
        'label': 'FICO Credit Score',
        'help': 'Your credit score (350-800)'
    },
    'pub_rec': {
        'type': 'categorical',
        'options': ['0', '1', '1+'],
        'label': 'Public Records',
        'help': 'Number of derogatory public records'
    },
    'inq_last_6mths': {
        'type': 'categorical',
        'options': ['0', '1', '2', '2+'],
        'label': 'Credit Inquiries (Last 6 Months)',
        'help': 'How many times was your credit checked?'
    },
    'revol_bal': {
        'type': 'numeric',
        'min': 0,
        'max': 100000,
        'cap_max': 100000,
        'allow_negative': False,
        'label': 'Revolving Balance ($)',
        'help': 'Total credit card balance'
    },
    'mths_since_last_major_derog': {
        'type': 'numeric',
        'min': 0,
        'max': 85,
        'cap_min': 999,  # Values below 0 capped to 999
        'cap_max': 85,   # Values above 85 capped to 85
        'allow_negative': True,  # Allow but will cap
        'label': 'Months Since Last Derogatory',
        'help': 'Time since last major negative mark'
    },
    'tot_cur_bal': {
        'type': 'numeric',
        'min': 0,
        'max': 688045,
        'cap_max': 688045,
        'allow_negative': False,
        'label': 'Total Current Balance ($)',
        'help': 'Total balance across all accounts'
    },
    'total_bal_il': {
        'type': 'numeric',
        'min': 0,
        'max': None,
        'allow_negative': False,
        'label': 'Total Installment Balance ($)',
        'help': 'Balance on installment loans'
    },
    'max_bal_bc': {
        'type': 'numeric',
        'min': 0,
        'max': None,
        'allow_negative': False,
        'label': 'Max Bank Card Balance ($)',
        'help': 'Highest credit card balance'
    },
    'all_util': {
        'type': 'numeric',
        'min': 0,
        'max': 40,
        'allow_negative': False,
        'label': 'Overall Credit Utilization (%)',
        'help': 'Percentage of available credit used'
    },
    'bc_util': {
        'type': 'numeric',
        'min': 0,
        'max': 40,
        'allow_negative': False,
        'label': 'Bank Card Utilization (%)',
        'help': 'Credit card utilization percentage'
    },
    'mort_acc': {
        'type': 'numeric',
        'min': 0,
        'max': None,
        'allow_negative': False,
        'label': 'Mortgage Accounts',
        'help': 'Number of mortgage accounts'
    },
    'num_accts_ever_120_pd': {
        'type': 'categorical',
        'options': ['0', '1', '1+'],
        'label': 'Accounts 120+ Days Past Due',
        'help': 'Severely delinquent accounts'
    },
    'num_sats': {
        'type': 'numeric',
        'min': 0,
        'max': None,
        'allow_negative': False,
        'label': 'Satisfactory Accounts',
        'help': 'Number of accounts in good standing'
    },
    'percent_bc_gt_75': {
        'type': 'numeric',
        'min': 0,
        'max': 100,
        'allow_negative': False,
        'label': '% Bank Cards > 75% Limit',
        'help': 'Percentage of cards near limit'
    },
    'pct_tl_nvr_dlq': {
        'type': 'numeric',
        'min': 0,
        'max': 100,
        'allow_negative': False,
        'label': '% Trades Never Delinquent',
        'help': 'Percentage with perfect payment history'
    }
}

# ============================================================================
# CLASSIFICATION FEATURE CONFIGURATION
# ============================================================================
CLASSIFICATION_FEATURES = {
    'sub_grade': CLUSTERING_FEATURES['sub_grade'],
    'home_ownership_MORTGAGE': {
        'type': 'encoded',
        'source': 'home_ownership',
        'match_value': 'MORTGAGE'
    },
    'verification_status_Not Verified': {
        'type': 'encoded',
        'source': 'verification_status',
        'match_value': 'Not Verified'
    },
    'home_ownership_RENT': {
        'type': 'encoded',
        'source': 'home_ownership',
        'match_value': 'RENT'
    },
    'acc_open_past_24mths': {
        'type': 'numeric',
        'min': 0,
        'allow_negative': False,
        'label': 'Accounts Opened (Last 24 Months)',
        'help': 'New accounts in past 2 years'
    },
    'loan_amnt': CLUSTERING_FEATURES['loan_amnt'],
    'emp_length': CLUSTERING_FEATURES['emp_length'],
    'avg_cur_bal': {
        'type': 'numeric',
        'min': 0,
        'allow_negative': False,
        'label': 'Average Current Balance ($)',
        'help': 'Average balance across accounts'
    },
    'installment': {
        'type': 'numeric',
        'min': 0,
        'allow_negative': False,
        'label': 'Monthly Installment ($)',
        'help': 'Monthly loan payment amount'
    },
    'num_actv_rev_tl': {
        'type': 'numeric',
        'min': 0,
        'allow_negative': False,
        'label': 'Active Revolving Accounts',
        'help': 'Number of active credit cards'
    },
    'dti': CLUSTERING_FEATURES['dti'],
    'mths_since_rcnt_il': {
        'type': 'numeric',
        'min': 0,
        'allow_negative': False,
        'label': 'Months Since Recent Installment',
        'help': 'Time since last installment loan'
    },
    'Fico_avg_val': CLUSTERING_FEATURES['Fico_avg_val'],
    'mths_since_recent_inq': {
        'type': 'numeric',
        'min': 0,
        'allow_negative': False,
        'label': 'Months Since Recent Inquiry',
        'help': 'Time since credit was last checked'
    },
    'annual_inc': CLUSTERING_FEATURES['annual_inc'],
    'delinq_2yrs': CLUSTERING_FEATURES['delinq_2yrs'],
    'total_bal_il': CLUSTERING_FEATURES['total_bal_il'],
    'inq_last_6mths': CLUSTERING_FEATURES['inq_last_6mths'],
    'max_bal_bc': CLUSTERING_FEATURES['max_bal_bc'],
    'inq_last_12m': {
        'type': 'numeric',
        'min': 0,
        'allow_negative': False,
        'label': 'Credit Inquiries (Last 12 Months)',
        'help': 'Number of credit checks in past year'
    },
    'mo_sin_old_rev_tl_op': {
        'type': 'numeric',
        'min': 0,
        'allow_negative': False,
        'label': 'Months Since Oldest Revolving Account',
        'help': 'Age of oldest credit card'
    },
    'all_util': CLUSTERING_FEATURES['all_util'],
    'purpose_others': {
        'type': 'encoded',
        'source': 'purpose',
        'match_value': 'others'
    },
    'int_rate': {
        'type': 'numeric',
        'min': 0,
        'max': 60,
        'allow_negative': False,
        'label': 'Interest Rate (%)',
        'help': 'Loan interest rate (0-60%)'
    }
}

# ============================================================================
# CLUSTER DESCRIPTIONS
# ============================================================================
CLUSTER_DESCRIPTIONS = {
    0: "🟢 **Low Risk Segment**: High income, excellent credit scores, low debt utilization. Strong financial health with minimal default risk.",
    1: "🟡 **Moderate Risk Segment**: Average income and credit scores with balanced debt levels. Standard lending terms typically apply.",
    2: "🟡 **Growth Potential Segment**: Newer borrowers with shorter credit history but stable income. Shows promising trajectory.",
    3: "🟠 **High Utilization Segment**: Adequate income but elevated credit utilization. May benefit from debt consolidation.",
    4: "🔴 **High Risk Segment**: Lower credit scores, higher delinquencies, elevated DTI ratios. Requires careful assessment and higher rates."
}