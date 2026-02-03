import pandas as pd
import os

input_csv_path = r"C:\Users\VASANTH RAO\Downloads\PROJECT NEW\fraud_transactions.csv"
output_csv_path_cleaned = r"C:\Users\VASANTH RAO\Downloads\PROJECT NEW\fraud_transactions_cleaned.csv"

try:
    df = pd.read_csv(input_csv_path)
    print(f"Loaded data. DataFrame shape: {df.shape}")

    print("\nChecking for missing values:")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])

    if missing_values.sum() == 0:
        print("\nNo missing values found.")
        # If no missing values, just save to a new file to proceed with the flow
        df.to_csv(output_csv_path_cleaned, index=False)
        print(f"Cleaned DataFrame (no missing values found) saved to {output_csv_path_cleaned}")
    else:
        # For this step, I will just report the missing values.
        # A more robust solution would involve imputation or dropping based on analysis.
        # For now, I'll assume we'll address them later if they are significant.
        print("\nMissing values detected. Further analysis and handling will be required.")
        # For demonstration, let's just save the current state
        df.to_csv(output_csv_path_cleaned, index=False)
        print(f"DataFrame with missing values (reported) saved to {output_csv_path_cleaned}")


except FileNotFoundError:
    print(f"Error: The file '{input_csv_path}' was not found.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
