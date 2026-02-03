import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_fscore_support
import numpy as np
import joblib # For saving the model

try:
    import xgboost as xgb
except ImportError:
    print("XGBoost is not installed. Please install it using 'pip install xgboost'")
    xgb = None

def train_and_evaluate_final_model(file_path, threshold=0.7):
    """
    Loads data, trains the final XGBoost model, evaluates it at a specific threshold,
    and saves the trained model.
    """
    if xgb is None:
        print("XGBoost is required but not installed. Exiting.")
        return

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return

    # Assuming 'isFraud' is the target variable
    X = df.drop('isFraud', axis=1)
    y = df['isFraud']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")

    print("\n--- Training Final XGBoost Model ---")
    # Using default parameters for simplicity as per previous discussions, and no scale_pos_weight
    final_model = xgb.XGBClassifier(random_state=42)
    final_model.fit(X_train, y_train)

    # Get prediction probabilities for the positive class
    y_pred_proba = final_model.predict_proba(X_test)[:, 1]
    
    # Apply the chosen threshold
    y_pred = (y_pred_proba >= threshold).astype(int)

    print(f"\nModel Evaluation (Final XGBoost Classifier at Threshold={threshold}):")
    print(classification_report(y_test, y_pred))

    # Save the trained model
    model_filename = "final_fraud_detection_model.joblib"
    joblib.dump(final_model, model_filename)
    print(f"\nFinal model saved to {model_filename}")

if __name__ == "__main__":
    data_file = "fraud_transactions_engineered.csv"
    # Use the chosen threshold (e.g., 0.7)
    train_and_evaluate_final_model(data_file, threshold=0.7)
