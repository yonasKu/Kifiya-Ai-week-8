import unittest
import json
from src.api import app

class TestFraudDetectionAPI(unittest.TestCase):
    def setUp(self):
        # Setup Flask test client
        self.client = app.test_client()
        self.client.testing = True

    def test_predict_credit_fraud(self):
        # Example input for credit fraud prediction
        example_data = {
            "features": [0.0, -1.359807, -0.072781, 2.536347, 1.378155, -0.338321, 0.462388, 0.239599, 0.098698, 0.363787, ...]  # Add remaining features
        }
        
        response = self.client.post('/predict_credit_fraud', json=example_data)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("prediction", data)

    def test_predict_fraud(self):
        # Example input for general fraud prediction
        example_data = {
            "features": [0.411032, 0.549607, -0.363124, 0.0, 7, 0.0, -0.413800, 3, 6, ...]  # Add remaining features
        }
        
        response = self.client.post('/predict_fraud', json=example_data)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("prediction", data)

    def test_fraud_stats(self):
        # Test fraud stats endpoint
        response = self.client.get('/api/stats')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("total_transactions", data)
        self.assertIn("total_frauds", data)
        self.assertIn("fraud_percentage", data)

    def test_fraud_trends(self):
        # Test fraud trends endpoint
        response = self.client.get('/api/fraud_trends')
        self.assertEqual(response.status_code, 200)

    def test_device_fraud(self):
        # Test fraud cases by device endpoint
        response = self.client.get('/api/device_fraud')
        self.assertEqual(response.status_code, 200)

if __name__ == '__main__':
    unittest.main()
