# utils/validators.py
"""
Input validation functions with specific rules
"""

from utils.constants import CLUSTERING_FEATURES, CLASSIFICATION_FEATURES


def validate_numeric_field(value, field_config, field_name):
    """
    Validate a numeric field based on its configuration

    Returns: (validated_value, warning_message or None)
    """
    warnings = []

    # Check if negative when not allowed
    if not field_config.get('allow_negative', False) and value < 0:
        return None, f"❌ {field_config['label']} cannot be negative"

    # Apply min cap
    if 'min' in field_config and field_config['min'] is not None:
        if value < field_config['min']:
            if 'cap_min' in field_config:
                value = field_config['cap_min']
                warnings.append(f"⚠️ {field_config['label']} adjusted to minimum: {value}")
            else:
                return None, f"❌ {field_config['label']} must be at least {field_config['min']}"

    # Apply max cap
    if 'max' in field_config and field_config['max'] is not None:
        if value > field_config['max']:
            if 'cap_max' in field_config:
                value = field_config['cap_max']
                warnings.append(f"⚠️ {field_config['label']} capped to maximum: {value}")
            else:
                return None, f"❌ {field_config['label']} cannot exceed {field_config['max']}"

    return value, warnings[0] if warnings else None


def validate_categorical_field(value, field_config, field_name):
    """
    Validate a categorical field

    Returns: (is_valid, warning_message or None)
    """
    if value not in field_config['options']:
        valid_options = ', '.join(map(str, field_config['options']))
        return False, f"❌ {field_config['label']} must be one of: {valid_options}"

    return True, None


def validate_clustering_inputs(inputs):
    """
    Validate all clustering inputs

    Args:
        inputs: Dictionary of user inputs

    Returns:
        tuple: (validated_inputs, list of warnings)
    """
    validated = {}
    warnings = []
    errors = []

    for field_name, field_config in CLUSTERING_FEATURES.items():
        if field_name not in inputs:
            errors.append(f"❌ Missing required field: {field_config['label']}")
            continue

        value = inputs[field_name]

        if field_config['type'] == 'numeric':
            validated_value, warning = validate_numeric_field(value, field_config, field_name)
            if validated_value is None:
                errors.append(warning)
            else:
                validated[field_name] = validated_value
                if warning:
                    warnings.append(warning)

        elif field_config['type'] == 'categorical':
            is_valid, warning = validate_categorical_field(value, field_config, field_name)
            if not is_valid:
                errors.append(warning)
            else:
                validated[field_name] = value

    # Additional business logic warnings
    if 'dti' in validated and validated['dti'] > 35:
        warnings.append("⚠️ High DTI ratio (>35%) may indicate financial stress")

    if 'Fico_avg_val' in validated and validated['Fico_avg_val'] < 600:
        warnings.append("⚠️ FICO score below 600 significantly impacts loan approval")

    if 'all_util' in validated and validated['all_util'] > 30:
        warnings.append("⚠️ Credit utilization above 30% is considered high")

    if errors:
        return None, errors

    return validated, warnings


def validate_classification_inputs(inputs):
    """
    Validate all classification inputs

    Args:
        inputs: Dictionary of user inputs (raw form inputs)

    Returns:
        tuple: (validated_inputs, list of warnings)
    """
    validated = {}
    warnings = []
    errors = []

    # List of raw fields that need to be validated
    raw_fields = {
        'sub_grade': {'type': 'categorical', 'options': [
            'A1', 'A2', 'A3', 'A4', 'A5', 'B1', 'B2', 'B3', 'B4', 'B5',
            'C1', 'C2', 'C3', 'C4', 'C5', 'D1', 'D2', 'D3', 'D4', 'D5',
            'E1', 'E2', 'E3', 'E4', 'E5', 'F1', 'F2', 'F3', 'F4', 'F5',
            'G1', 'G2', 'G3', 'G4', 'G5']},
        'home_ownership': {'type': 'categorical', 'options': ['MORTGAGE', 'RENT', 'OWN', 'NO_PERMANENT_ADDRESS']},
        'verification_status': {'type': 'categorical', 'options': ['Verified', 'Not Verified']},
        'purpose': {'type': 'categorical', 'options': ['debt_consolidation', 'credit_card', 'home_improvement',
                                                       'major_purchase', 'medical', 'others']},
        'emp_length': {'type': 'categorical', 'options': ['<1', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10+']},
        'delinq_2yrs': {'type': 'categorical', 'options': ['No', 'Yes']},
        'inq_last_6mths': {'type': 'categorical', 'options': ['0', '1', '2', '2+']},
        'loan_amnt': {'type': 'numeric', 'min': 0, 'allow_negative': False},
        'int_rate': {'type': 'numeric', 'min': 0, 'max': 60, 'allow_negative': False},
        'installment': {'type': 'numeric', 'min': 0, 'allow_negative': False},
        'annual_inc': {'type': 'numeric', 'min': 0, 'allow_negative': False},
        'dti': {'type': 'numeric', 'min': 0, 'max': 40, 'allow_negative': False},
        'Fico_avg_val': {'type': 'numeric', 'min': 350, 'max': 800, 'allow_negative': False},
        'total_bal_il': {'type': 'numeric', 'min': 0, 'allow_negative': False},
        'max_bal_bc': {'type': 'numeric', 'min': 0, 'allow_negative': False},
        'all_util': {'type': 'numeric', 'min': 0, 'max': 40, 'allow_negative': False},
        'acc_open_past_24mths': {'type': 'numeric', 'min': 0, 'allow_negative': False},
        'avg_cur_bal': {'type': 'numeric', 'min': 0, 'allow_negative': False},
        'num_actv_rev_tl': {'type': 'numeric', 'min': 0, 'allow_negative': False},
        'mths_since_rcnt_il': {'type': 'numeric', 'min': 0, 'allow_negative': False},
        'mths_since_recent_inq': {'type': 'numeric', 'min': 0, 'allow_negative': False},
        'inq_last_12m': {'type': 'numeric', 'min': 0, 'allow_negative': False},
        'mo_sin_old_rev_tl_op': {'type': 'numeric', 'min': 0, 'allow_negative': False},
    }

    # Validate all fields
    for field_name, field_config in raw_fields.items():
        if field_name not in inputs:
            errors.append(f"❌ Missing required field: {field_name}")
            continue

        value = inputs[field_name]

        if field_config['type'] == 'numeric':
            validated_value, warning = validate_numeric_field(value, field_config, field_name)
            if validated_value is None:
                errors.append(warning)
            else:
                validated[field_name] = validated_value
                if warning:
                    warnings.append(warning)

        elif field_config['type'] == 'categorical':
            is_valid, warning = validate_categorical_field(value, field_config, field_name)
            if not is_valid:
                errors.append(warning)
            else:
                validated[field_name] = value

    # Additional warnings
    if 'dti' in validated and validated['dti'] > 35:
        warnings.append("⚠️ High DTI ratio may reduce approval chances")

    if 'Fico_avg_val' in validated and validated['Fico_avg_val'] < 620:
        warnings.append("⚠️ FICO score below 620 may result in higher interest rates or denial")

    if errors:
        return None, errors

    return validated, warnings


def convert_binary_to_int(value):
    """Convert Yes/No to 1/0"""
    if value in ['Yes', 'yes', 'YES', 1, '1', True]:
        return 1
    return 0


def convert_categorical_to_encoded(value, options):
    """Convert categorical value to numeric encoding"""
    if value in options:
        return options.index(value)
    return 0