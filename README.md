# Fraud Detection System

This project is a machine learning-based fraud detection system designed to identify potentially fraudulent financial transactions. The system uses an XGBoost model trained on a large dataset of transactions to classify them as either legitimate or fraudulent.

## Project Overview

The project follows a standard data science workflow:
1.  **Data Preparation:** Raw data is loaded, cleaned, and processed.
2.  **Feature Engineering:** New, more informative features are created from the raw data to improve model performance. For example, balance changes are calculated from old and new balances.
3.  **Exploratory Data Analysis (EDA):** The data is analyzed to understand its structure, find patterns, and identify key characteristics of fraudulent transactions. The results of this analysis are in the `EDA_plots` directory.
4.  **Model Training and Evaluation:** An XGBoost model is trained on the engineered features. The model's performance is evaluated, with a focus on the trade-off between precision and recall.
5.  **Threshold Adjustment:** The model's decision threshold is adjusted to find an optimal balance between catching fraud (recall) and minimizing false alarms (precision).

## Final Model Performance

The final model is an XGBoost classifier. Based on user preference for high precision, a **decision threshold of 0.7** was chosen. At this threshold, the model's performance on the test data is:

*   **Precision (for Fraud): 0.80 (80%)**
    *   *This means that when the model flags a transaction as fraudulent, it is correct 80% of the time.*
*   **Recall (for Fraud): 0.61 (61%)**
    *   *This means that the model successfully identifies 61% of all actual fraudulent transactions.*

## Getting Started

### Prerequisites

You will need Python 3 installed. You can install the necessary Python libraries using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```
### Kaggle Dataset
Link : " kaggle kernels output mahmoudredagamail/fraud-detection-dataset -p /path/to/dest "

### File Structure

*   `data_cleaning.py`: Script for cleaning the data.
*   `feature_engineering.py`: Script for creating new features.
*   `eda.py`: Script for exploratory data analysis.
*   `load_data.py`: Script for loading the data.
*   `model_training.py`: Script to train the final model and save it.
*   `predict_fraud.py`: An example script to show how to use the saved model for predictions.
*   `final_fraud_detection_model.joblib`: The saved, trained XGBoost model.
*   `requirements.txt`: A list of the Python libraries required for this project.
*   `Task Details.pdf`: The original task details for the project.
*   `EDA_plots/`: A directory containing plots from the exploratory data analysis.

### Usage

1.  **Train the model:**
    To train the model and save it, run the `model_training.py` script:
    ```bash
    python model_training.py
    ```
    This will create the `final_fraud_detection_model.joblib` file.

2.  **Make predictions on new data:**
    The `predict_fraud.py` script contains a demonstration of how to load the trained model and make predictions on new data. You can run it with:
    ```bash
    python predict_fraud.py
    ```
    You can modify this script to use your own new data.

