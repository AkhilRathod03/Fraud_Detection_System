import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="FraudSentinel AI",
    page_icon="🛡️",
    layout="centered"
)

# --- Custom Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    .fraud-text {
        color: #ff4b4b;
        font-weight: bold;
        font-size: 24px;
    }
    .safe-text {
        color: #28a745;
        font-weight: bold;
        font-size: 24px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Load Model ---
@st.cache_resource
def load_model():
    # Ensure xgboost is installed and imported for joblib to load it
    try:
        import xgboost as xgb
    except ImportError:
        st.error("XGBoost is not installed. Please check requirements.txt")
    return joblib.load('final_fraud_detection_model.joblib')

model = load_model()

# --- App Header ---
st.title("🛡️ FraudSentinel AI")
st.subheader("Financial Transaction Fraud Detection System")
st.write("""
This application uses a trained **XGBoost Classifier** to analyze financial transactions 
and determine the probability of fraudulent activity.
""")

st.divider()

# --- Input Form ---
with st.container():
    st.write("### 📝 Enter Transaction Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        trans_type = st.selectbox("Transaction Type", ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"])
        amount = st.number_input("Transaction Amount ($)", min_value=0.0, format="%.2f")
        step = st.number_input("Step (Time Hour 1-744)", min_value=1, max_value=744, value=1)

    with col2:
        old_balance_org = st.number_input("Original Account: Old Balance ($)", min_value=0.0, format="%.2f")
        new_balance_org = st.number_input("Original Account: New Balance ($)", min_value=0.0, format="%.2f")
        
    st.write("---")
    col3, col4 = st.columns(2)
    with col3:
        old_balance_dest = st.number_input("Destination Account: Old Balance ($)", min_value=0.0, format="%.2f")
    with col4:
        new_balance_dest = st.number_input("Destination Account: New Balance ($)", min_value=0.0, format="%.2f")

    is_flagged = st.checkbox("Was this transaction flagged by the system as suspicious?")

# --- Prediction Logic ---
if st.button("Analyze Transaction"):
    # 1. Feature Engineering (Match the logic used in model training)
    balance_change_orig = new_balance_org - old_balance_org
    balance_change_dest = new_balance_dest - old_balance_dest
    
    # One-hot encoding manual mapping
    # Columns expected: 'step', 'amount', 'isFlaggedFraud', 'balance_change_orig', 'balance_change_dest', 
    # 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER'
    
    data = {
        'step': [step],
        'amount': [amount],
        'isFlaggedFraud': [1 if is_flagged else 0],
        'balance_change_orig': [balance_change_orig],
        'balance_change_dest': [balance_change_dest],
        'type_CASH_OUT': [1 if trans_type == "CASH_OUT" else 0],
        'type_DEBIT': [1 if trans_type == "DEBIT" else 0],
        'type_PAYMENT': [1 if trans_type == "PAYMENT" else 0],
        'type_TRANSFER': [1 if trans_type == "TRANSFER" else 0]
    }
    
    input_df = pd.DataFrame(data)
    
    # 2. Get Prediction
    prob = model.predict_proba(input_df)[:, 1][0]
    threshold = 0.7
    is_fraud = prob >= threshold
    
    # 3. Display Result
    st.divider()
    st.write("### 🔍 Analysis Result")
    
    # Progress bar for probability
    st.write(f"**Fraud Probability:** {prob:.2%}")
    st.progress(float(prob))
    
    if is_fraud:
        st.error("🚨 WARNING: High Risk of Fraud Detected!")
        st.markdown(f'<p class="fraud-text">PREDICTED STATUS: FRAUDULENT</p>', unsafe_allow_html=True)
        st.warning("Action Required: This transaction exceeds the 70% risk threshold and should be blocked for review.")
    else:
        st.success("✅ Transaction Appears Safe.")
        st.markdown(f'<p class="safe-text">PREDICTED STATUS: LEGITIMATE</p>', unsafe_allow_html=True)
        st.balloons()

# --- Project Info Sidebar ---
st.sidebar.title("About the Project")
st.sidebar.info("""
**Project Name:** FraudSentinel AI
**Model:** XGBoost Classifier
**Precision:** 80% 
**Recall:** 61%
**Threshold:** 0.7

**Motive:** To provide a robust, real-time solution for identifying financial anomalies and preventing fraudulent transfers.
""")

st.sidebar.markdown("---")
st.sidebar.write("Developed by **Akhil Rathod**")
st.sidebar.markdown("[View on Streamlit Cloud](https://share.streamlit.io/user/akhilrathod03)")
