from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load pre-trained models
credit_model_path = os.path.join('model', 'Credit Card Fraud Detection with Random Forest_random_forest_model.pkl')
fraud_model_path = os.path.join('model', 'Fraud Detection with Random Forest_random_forest_model.pkl')

credit_model = joblib.load(credit_model_path)
fraud_model = joblib.load(fraud_model_path)

@app.route('/')
def home():
    return "Fraud Detection API is running!"

# Endpoint for credit card fraud detection
@app.route('/predict_credit_fraud', methods=['POST'])
def predict_credit_fraud():
    data = request.get_json(force=True)  # Get the input data as JSON
    features = np.array(data['features']).reshape(1, -1)
    prediction = credit_model.predict(features)
    return jsonify({"prediction": int(prediction[0])})

# Endpoint for general fraud detection
@app.route('/predict_fraud', methods=['POST'])
def predict_fraud():
    data = request.get_json(force=True)  # Get the input data as JSON
    features = np.array(data['features']).reshape(1, -1)
    prediction = fraud_model.predict(features)
    return jsonify({"prediction": int(prediction[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
