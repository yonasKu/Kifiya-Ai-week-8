# tests/test_api.py
import pytest
import json
from src.serve_model import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_predict_fraud(client):
    data = {"feature1": [value1], "feature2": [value2]}  # Add your feature names and example values
    response = client.post('/predict/fraud', json=data)
    assert response.status_code == 200
    assert 'predictions' in response.get_json()

def test_predict_credit_card(client):
    data = {"feature1": [value1], "feature2": [value2]}  # Add your feature names and example values
    response = client.post('/predict/credit_card', json=data)
    assert response.status_code == 200
    assert 'predictions' in response.get_json()
