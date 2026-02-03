import pandas as pd
import os

# Use the absolute path for the zip file
zip_file_path = r"C:\Users\VASANTH RAO\Downloads\PROJECT NEW\Fraud.csv.zip"
csv_file_name_in_zip = "Fraud.csv" # Assuming the CSV inside the zip is named Fraud.csv

try:
    # Attempt to read the CSV directly from the zip file
    # pandas.read_csv can often infer compression from the file extension
    df = pd.read_csv(zip_file_path, compression='zip')
    print(f"Successfully loaded data. DataFrame shape: {df.shape}")
    print("First 5 rows:")
    print(df.head())
    
    # Save the dataframe to a new CSV file for easier access later
    output_csv_path = r"C:\Users\VASANTH RAO\Downloads\PROJECT NEW\fraud_transactions.csv"
    df.to_csv(output_csv_path, index=False)
    print(f"DataFrame successfully saved to {output_csv_path}")

except FileNotFoundError:
    print(f"Error: The file '{zip_file_path}' was not found.")
except KeyError:
    print(f"Error: '{csv_file_name_in_zip}' not found inside '{zip_file_path}'. Please check the actual CSV name within the zip.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")