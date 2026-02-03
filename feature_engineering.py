
import pandas as pd
import os

input_csv_path = r"C:\Users\VASANTH RAO\Downloads\PROJECT NEW\fraud_transactions_multicollinearity_checked.csv"
output_csv_path_engineered = r"C:\Users\VASANTH RAO\Downloads\PROJECT NEW\fraud_transactions_engineered.csv"

try:
    df = pd.read_csv(input_csv_path)
    print(f"Loaded data. DataFrame shape: {df.shape}")

    # 1. Create balance change features
    df['balance_change_orig'] = df['newbalanceOrig'] - df['oldbalanceOrg']
    df['balance_change_dest'] = df['newbalanceDest'] - df['oldbalanceDest']
    print("Created 'balance_change_orig' and 'balance_change_dest' features.")

    # 2. One-hot encode the 'type' column
    df = pd.get_dummies(df, columns=['type'], drop_first=True, dtype=int)
    print("One-hot encoded 'type' column.")

    # 3. Drop original balance columns, 'type' (as it's encoded), 'nameOrig', 'nameDest'
    columns_to_drop = [
        'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest',
        'nameOrig', 'nameDest'
    ]
    df.drop(columns=columns_to_drop, inplace=True)
    print(f"Dropped columns: {columns_to_drop}")

    print(f"Engineered DataFrame shape: {df.shape}")
    print("First 5 rows of engineered data:")
    print(df.head())

    # Save the engineered DataFrame
    df.to_csv(output_csv_path_engineered, index=False)
    print(f"Engineered DataFrame saved to {output_csv_path_engineered}")

except FileNotFoundError:
    print(f"Error: The file '{input_csv_path}' was not found.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
