import joblib  # Make sure you import joblib
import pandas as pd
import logging
from flask import Flask, request, jsonify

# Initialize the Flask application
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the first model
model_path_1 = 'model/Credit Card Fraud Detection with Random Forest_random_forest_model.pkl'
try:
    model_1 = joblib.load(model_path_1)  # Use joblib to load the model
    logger.info("Model 1 loaded successfully.")
    logger.info(f"Model 1 type: {type(model_1)}")  # Log the type of the model

    # Check if model_1 is a valid model
    if not hasattr(model_1, 'predict'):
        logger.error("Model 1 is not a valid model object.")
except Exception as e:
    logger.error(f"Error loading Model 1: {e}")
    model_1 = None  # Set to None if loading fails

# Load the second model
model_path_2 = 'model/Fraud Detection with Random Forest_random_forest_model.pkl'
try:
    model_2 = joblib.load(model_path_2)  # Use joblib to load the model
    logger.info("Model 2 loaded successfully.")
    logger.info(f"Model 2 type: {type(model_2)}")  # Log the type of the model

    # Check if model_2 is a valid model
    if not hasattr(model_2, 'predict'):
        logger.error("Model 2 is not a valid model object.")
except Exception as e:
    logger.error(f"Error loading Model 2: {e}")
    model_2 = None  # Set to None if loading fails

# Helper function to ensure input data has required columns
def validate_input(data, model):
    if model is None:
        return False, "Model is not loaded."
    
    # You should replace these with the actual feature names your model expects
    required_columns = [f'V{i}' for i in range(1, 30)] + ['Time']  # Adjust as necessary
    if set(required_columns).issubset(data.keys()):
        return True, None
    else:
        missing_cols = set(required_columns) - set(data.keys())
        return False, f"Missing columns: {missing_cols}"

# Define a route for predictions using the first model
@app.route('/predict/model1', methods=['POST'])
def predict_model1():
    # Get data from the request
    data = request.get_json()
    
    # Validate input data
    is_valid, error_msg = validate_input(data, model_1)
    if not is_valid:
        return jsonify({'error': error_msg}), 400

    # Ensure the data is in the correct format
    input_df = pd.DataFrame([data])  # Wrap data in a list to ensure it is treated as a single row

    # Make predictions using the first model
    try:
        predictions = model_1.predict(input_df)
        logger.info("Prediction made using Model 1.")
    except Exception as e:
        logger.error(f"Error making prediction with Model 1: {e}")
        return jsonify({'error': str(e)}), 500

    # Return predictions as a JSON response
    return jsonify(predictions.tolist())

# Define a route for predictions using the second model
@app.route('/predict/model2', methods=['POST'])
def predict_model2():
    # Get data from the request
    data = request.get_json()

    # Validate input data
    is_valid, error_msg = validate_input(data, model_2)
    if not is_valid:
        return jsonify({'error': error_msg}), 400

    # Ensure the data is in the correct format
    input_df = pd.DataFrame([data])  # Wrap data in a list to ensure it is treated as a single row

    # Make predictions using the second model
    try:
        predictions = model_2.predict(input_df)
        logger.info("Prediction made using Model 2.")
    except Exception as e:
        logger.error(f"Error making prediction with Model 2: {e}")
        return jsonify({'error': str(e)}), 500

    # Return predictions as a JSON response
    return jsonify(predictions.tolist())

if __name__ == '__main__':
    app.run(debug=True)
