# resave_pickles.py
"""
Re-save pickle files with current Python version
"""
import pickle
import sys


def resave_pickle(input_path, output_path):
    """Re-save a pickle file"""
    print(f"Loading {input_path}...")

    # Try multiple methods to load
    obj = None

    # Method 1: Standard
    try:
        with open(input_path, 'rb') as f:
            obj = pickle.load(f)
        print("✅ Loaded with standard pickle")
    except Exception as e:
        print(f"Standard pickle failed: {e}")

    # Method 2: pickle5
    if obj is None:
        try:
            import pickle5
            with open(input_path, 'rb') as f:
                obj = pickle5.load(f)
            print("✅ Loaded with pickle5")
        except Exception as e:
            print(f"pickle5 failed: {e}")

    # Method 3: latin1 encoding
    if obj is None:
        try:
            with open(input_path, 'rb') as f:
                obj = pickle.load(f, encoding='latin1')
            print("✅ Loaded with latin1 encoding")
        except Exception as e:
            print(f"latin1 encoding failed: {e}")
            raise

    # Save with current protocol
    print(f"Saving to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("✅ Saved successfully")


if __name__ == "__main__":
    files = [
        'loan_kmeans_model_pipeline.pkl',
        'loan_xgb_model_pipeline.pkl',
        'shap_explainer.pkl'
    ]

    for filename in files:
        input_path = f'models/{filename}'
        output_path = f'models/{filename}.new'

        try:
            resave_pickle(input_path, output_path)
            print(f"\n✅ {filename} re-saved successfully!")
            print(f"Replace original: mv {output_path} {input_path}\n")
        except Exception as e:
            print(f"\n❌ Failed to process {filename}: {e}\n")