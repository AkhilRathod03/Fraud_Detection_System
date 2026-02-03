
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

input_csv_path = r"C:\Users\VASANTH RAO\Downloads\PROJECT NEW\fraud_transactions_engineered.csv"
output_eda_dir = r"C:\Users\VASANTH RAO\Downloads\PROJECT NEW\EDA_plots"

# Create directory for plots if it doesn't exist
os.makedirs(output_eda_dir, exist_ok=True)

try:
    df = pd.read_csv(input_csv_path)
    print(f"Loaded data for EDA. DataFrame shape: {df.shape}")

    # 1. Basic Information
    print("\nDataFrame Info:")
    df.info()

    print("\nDataFrame Description:")
    print(df.describe())

    # 2. Target Variable Distribution
    print("\nDistribution of 'isFraud':")
    print(df['isFraud'].value_counts())
    print(df['isFraud'].value_counts(normalize=True) * 100)

    plt.figure(figsize=(6, 4))
    sns.countplot(x='isFraud', data=df)
    plt.title('Distribution of Fraudulent vs. Non-Fraudulent Transactions')
    plt.savefig(os.path.join(output_eda_dir, 'isFraud_distribution.png'))
    plt.close()
    print("Saved 'isFraud_distribution.png'")

    # 3. Feature Distribution (Numerical)
    numerical_features = ['step', 'amount', 'balance_change_orig', 'balance_change_dest']
    for col in numerical_features:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.savefig(os.path.join(output_eda_dir, f'{col}_distribution.png'))
        plt.close()
        print(f"Saved '{col}_distribution.png'")

        # Relationship with Target Variable - Numerical
        plt.figure(figsize=(8, 5))
        sns.boxplot(x='isFraud', y=col, data=df)
        plt.title(f'{col} vs isFraud')
        plt.savefig(os.path.join(output_eda_dir, f'{col}_vs_isFraud_boxplot.png'))
        plt.close()
        print(f"Saved '{col}_vs_isFraud_boxplot.png'")

    # 4. Feature Distribution (Categorical/Binary - One-hot encoded 'type' columns)
    type_cols = [col for col in df.columns if col.startswith('type_')]
    print("\nDistribution of Transaction Types:")
    for col in type_cols:
        print(f"{col}: {df[col].value_counts(normalize=True) * 100}")
        # Relationship with Target Variable - Categorical
        fraud_percentage = df.groupby(col)['isFraud'].mean() * 100
        print(f"Fraud percentage by {col}:\n{fraud_percentage}")

    # 5. Correlation Matrix
    plt.figure(figsize=(12, 10))
    corr_matrix = df.corr(numeric_only=True)
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', linewidths=.5)
    plt.title('Correlation Matrix')
    plt.savefig(os.path.join(output_eda_dir, 'correlation_matrix.png'))
    plt.close()
    print("Saved 'correlation_matrix.png'")

    print("\nEDA completed. Plots saved to 'EDA_plots' directory.")


except FileNotFoundError:
    print(f"Error: The file '{input_csv_path}' was not found.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
