import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import hashlib
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="wide"
)

# Title and description
st.title("🔍 Transaction Fraud Detection System")
st.markdown("""
This application helps detect potentially fraudulent transactions using multiple machine learning models.
Please enter your transaction details below.
""")

# Function to hash IP addresses to anonymize them
def hash_ip(ip_address):
    return int(hashlib.sha256(ip_address.encode('utf-8')).hexdigest(), 16) % 10**8

# Function to load pre-trained models and scaler
def load_models():
    """Load all trained models"""
    try:
        if not os.path.exists('models'):
            st.error("Models directory not found!")
            return None, None, None

        available_models = os.listdir('models')
        models = {}

        if 'random_forest.joblib' in available_models:
            models['Random Forest'] = joblib.load('models/random_forest.joblib')

        if 'xgboost.joblib' in available_models:
            models['XGBoost'] = joblib.load('models/xgboost.joblib')

        if 'lightgbm.joblib' in available_models:
            models['LightGBM'] = joblib.load('models/lightgbm.joblib')

        # Load the scaler
        if 'scaler.joblib' in available_models:
            scaler = joblib.load('models/scaler.joblib')
        else:
            st.error("Scaler not found!")
            return None, None, None

        # Load the label encoder
        le = LabelEncoder()
        if os.path.exists('models/label_encoder.joblib'):
            le = joblib.load('models/label_encoder.joblib')
        else:
            st.warning("LabelEncoder not found. Assuming manual input encoding.")

        if not models:
            st.error("No models could be loaded!")
            return None, None, None

        return models, scaler, le

    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None

# Function to prepare input data for prediction
def prepare_input_data(transaction_amount, transaction_type, ip_address, transaction_id, le):
    """Prepare input data for prediction"""
    ip_address_hashed = hash_ip(ip_address)
    transaction_id_encoded = le.transform([transaction_id])[0]

    # Assume 'transaction_type' is already encoded as a numeric category (0, 1, 2)
    transaction_type_encoded = {'type1': 0, 'type2': 1, 'type3': 2}[transaction_type]

    # Return a numpy array with 4 features
    input_data = np.array([[transaction_amount, transaction_type_encoded, ip_address_hashed, transaction_id_encoded]])

    return input_data

# Load models and scaler
models, scaler, le = load_models()

# Create input form
with st.form("transaction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        username = st.text_input('User Name', value='John Doe')
        transaction_date = st.date_input('Transaction Date', value=datetime.now())
        transaction_amount = st.number_input('Transaction Amount', min_value=1.0, max_value=1000.0, value=500.0)
    
    with col2:
        transaction_type = st.selectbox('Transaction Type', ['type1', 'type2', 'type3'])
        ip_address = st.text_input('IP Address', value='192.168.0.1')
        transaction_id = st.text_input('Transaction ID', value='trans_123')
    
    submit_button = st.form_submit_button("Check Transaction")

# Process form submission
if submit_button:
    if models is not None and scaler is not None and le is not None:
        try:
            # Prepare the input data
            input_data = prepare_input_data(transaction_amount, transaction_type, ip_address, transaction_id, le)

            # Apply scaling
            input_scaled = scaler.transform(input_data)

            # Predict using each model and display results
            results = {}
            overall_fraud_probability = 0
            
            for name, model in models.items():
                prediction = model.predict(input_scaled)[0]
                probability = model.predict_proba(input_scaled)[0][1]
                results[name] = {'prediction': prediction, 'probability': probability}
                overall_fraud_probability += probability

            # Calculate average fraud probability
            avg_fraud_probability = overall_fraud_probability / len(models)
            is_safe = avg_fraud_probability < 0.5

            # Display transaction details
            st.header("Transaction Details")
            details = {
                "User Name": username,
                "Transaction Date": transaction_date.strftime("%Y-%m-%d"),
                "Transaction Amount": f"${transaction_amount:,.2f}",
                "Transaction Type": transaction_type,
                "IP Address": ip_address,
                "Transaction ID": transaction_id
            }

            details_df = pd.DataFrame(list(details.items()), columns=['Field', 'Value'])
            st.table(details_df)

            # Display final verdict with large icon
            st.header("Final Verdict")
            if is_safe:
                st.markdown("""
                <div style='background-color: #d1e7dd; padding: 20px; border-radius: 10px; text-align: center;'>
                    <h1 style='color: #0f5132;'>✅ THIS TRANSACTION IS SAFE</h1>
                    <p style='font-size: 1.2em;'>Average Fraud Probability: {:.2%}</p>
                </div>
                """.format(avg_fraud_probability), unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='background-color: #f8d7da; padding: 20px; border-radius: 10px; text-align: center;'>
                    <h1 style='color: #842029;'>⚠️ THIS TRANSACTION IS NOT SAFE</h1>
                    <p style='font-size: 1.2em;'>Average Fraud Probability: {:.2%}</p>
                </div>
                """.format(avg_fraud_probability), unsafe_allow_html=True)

            # Display individual model predictions
            st.header("Detailed Model Predictions")
            
            # Create three columns for the models
            cols = st.columns(len(models))
            
            # Model descriptions
            model_descriptions = {
                'Random Forest': """
                    💡 Specializes in:
                    - Pattern recognition
                    - Handling complex relationships
                    - Robust against outliers
                """,
                'XGBoost': """
                    💡 Specializes in:
                    - High accuracy predictions
                    - Handling missing data
                    - Sequential pattern detection
                """,
                'LightGBM': """
                    💡 Specializes in:
                    - Fast processing
                    - Memory efficiency
                    - Handling large datasets
                """
            }

            for idx, (name, result) in enumerate(results.items()):
                with cols[idx]:
                    # Create a card-like container for each model
                    st.markdown("""
                    <div style='border: 1px solid #ddd; padding: 15px; border-radius: 10px;'>
                        <h3 style='text-align: center;'>{}</h3>
                    </div>
                    """.format(name), unsafe_allow_html=True)
                    
                    # Show prediction result with icon
                    if result['prediction'] == 1:
                        st.error("⚠️ Potential Fraud Detected")
                    else:
                        st.success("✅ Transaction Appears Safe")
                    
                    # Show probability
                    st.metric("Fraud Probability", f"{result['probability']:.2%}")
                    
                    # Add model description
                    st.markdown(model_descriptions[name])

            st.info("""
            💡 Note: This is a prediction based on machine learning models. 
            Please use this information as one of many factors in your decision-making process.
            """)

        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
    else:
        st.error("Unable to load models or scaler. Please check if model files exist in the 'models' directory.")

# Add footer
st.markdown("""
---
Made with ❤️ by Immortals Coders
""")