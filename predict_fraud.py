import pandas as pd
import joblib
import xgboost as xgb # Required to load XGBoost model

# 1. Load the trained model
model = joblib.load('final_fraud_detection_model.joblib')

# 2. Prepare new data (replace with your actual new data)
# This example creates dummy data with the correct engineered features
new_data_dict = {
    'step': [100, 200, 300],
    'amount': [1000.0, 5000.0, 200.0],
    'isFlaggedFraud': [0, 0, 0], # Added isFlaggedFraud
    'balance_change_orig': [-1000.0, -5000.0, -200.0], # oldbalanceOrg - newbalanceOrig
    'balance_change_dest': [1000.0, 5000.0, -200.0], # newbalanceDest - oldbalanceDest
    'type_CASH_OUT': [1, 0, 0],
    'type_DEBIT': [0, 1, 0],
    'type_PAYMENT': [0, 0, 1],
    'type_TRANSFER': [0, 0, 0]
}
new_data_df = pd.DataFrame(new_data_dict)


# 3. Get prediction probabilities
fraud_probabilities = model.predict_proba(new_data_df)[:, 1]

# 4. Apply the chosen threshold (e.g., 0.7)
chosen_threshold = 0.7
predictions = (fraud_probabilities >= chosen_threshold).astype(int)

# Display results
new_data_df['Fraud_Probability'] = fraud_probabilities
new_data_df['Predicted_Fraud'] = predictions
print("\nPredictions on New Data:")
print(new_data_df)
