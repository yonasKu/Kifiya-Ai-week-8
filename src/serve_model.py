
# src/serve_model.py
from src.api import app

if __name__ == '__main__':
    # Configuration or environment settings can be added here
    app.run(host='0.0.0.0', port=5000)
