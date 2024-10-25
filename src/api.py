# src/api.py
from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Paths to the pre-trained models and data file
credit_model_path = os.path.join('model', 'Credit Card Fraud Detection with Random Forest_random_forest_model.pkl')
fraud_model_path = os.path.join('model', 'Fraud Detection with Random Forest_random_forest_model.pkl')
data_file_path = os.path.join(os.path.dirname(__file__), '../data/Fraud_Data.csv')  # Update the file name if needed

# Load the pre-trained models
credit_model = joblib.load(credit_model_path)
fraud_model = joblib.load(fraud_model_path)

# Load fraud data for insights
fraud_data_df = pd.read_csv(data_file_path)

@app.route('/')
def home():
    return "Fraud Detection API is running!"

# Endpoint for predicting credit card fraud
@app.route('/predict_credit_fraud', methods=['POST'])
def predict_credit_fraud():
    data = request.get_json(force=True)
    features = np.array(data['features']).reshape(1, -1)
    prediction = credit_model.predict(features)
    return jsonify({"prediction": int(prediction[0])})

# Endpoint for predicting general fraud
@app.route('/predict_fraud', methods=['POST'])
def predict_fraud():
    data = request.get_json(force=True)
    features = np.array(data['features']).reshape(1, -1)
    prediction = fraud_model.predict(features)
    return jsonify({"prediction": int(prediction[0])})

# Endpoint for fraud statistics
@app.route('/api/stats', methods=['GET'])
def get_fraud_stats():
    total_transactions = len(fraud_data_df)
    total_frauds = fraud_data_df['fraud_class'].sum()
    fraud_percentage = (total_frauds / total_transactions) * 100 if total_transactions > 0 else 0

    return jsonify({
        'total_transactions': total_transactions,
        'total_frauds': total_frauds,
        'fraud_percentage': fraud_percentage
    })

# Endpoint for daily fraud trends
@app.route('/api/fraud_trends', methods=['GET'])
def get_fraud_trends():
    fraud_data_df['purchase_time'] = pd.to_datetime(fraud_data_df['purchase_time'])
    fraud_trends = fraud_data_df.groupby(fraud_data_df['purchase_time'].dt.date).agg({'fraud_class': 'sum'}).reset_index()
    fraud_trends.columns = ['date', 'fraud_cases']
    return fraud_trends.to_json(orient='records')

# Endpoint for fraud cases by device
@app.route('/api/device_fraud', methods=['GET'])
def get_device_fraud():
    device_fraud = fraud_data_df.groupby('device_id')['fraud_class'].sum().reset_index()
    return device_fraud.to_json(orient='records')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
