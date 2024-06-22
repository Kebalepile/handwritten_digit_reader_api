import os
import json
import time
import threading
import numpy as np
from flask import Flask, request, jsonify, redirect
from tensorflow.keras.models import load_model
from utils.digit_recognizer import init_model
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import logging

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(__name__)

# Enable CORS (Cross-Origin Resource Sharing) for the '/predict' endpoint
CORS(app, resources={r"/predict": {"origins": ["https://dipalo-tsa-motheo.github.io", "https://dipalo-tsa-motheo.github.io/"]}})

# Rate limiting configuration: 200 requests per day, 50 requests per hour
limiter = Limiter(get_remote_address, app=app, default_limits=["200 per day", "50 per hour"])

# Placeholder for the model
model = None
model_ready = threading.Event()

def load_model_thread():
    global model
    logger.info("Loading model...")
    model = init_model('handwritten_digits_reader.h5')
    logger.info("Model loaded.")
    model_ready.set()

# Start loading the model in a separate thread
threading.Thread(target=load_model_thread).start()

# Redirect all paths to '/predict' endpoint
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    return redirect('/predict')

# Function to clean and preprocess input data for prediction
def clean_input(input_data):
    try:
        # Convert input JSON data to a numpy array of float32 and normalize
        input_array = np.array(json.loads(input_data), dtype=np.float32).reshape(1, 28, 28, 1) / 255.0
        return input_array
    except (ValueError, TypeError) as e:
        raise ValueError("Invalid input data")

# Endpoint for predicting handwritten digits
@app.route('/predict', methods=['POST'])
@limiter.limit("50 per hour")  # Apply rate limiting to this endpoint
def predict():
    try:
        if not model_ready.is_set():
            return jsonify({'error': 'Model is still loading, please try again later'}), 503

        input_data = request.form.get('input')
        if not input_data:
            return jsonify({'error': 'No input data provided'}), 400
        
        logger.info(f"Received input data: {input_data}")
        input_array = clean_input(input_data)
        
        logger.info("Data preprocessed successfully, starting prediction...")
        prediction = model.predict(input_array)
        logger.info("Prediction completed.")
        
        digit = np.argmax(prediction)
        response = jsonify({'digit': int(digit)})

        return response

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

# Health check endpoint to verify the status of the API
@app.route('/health', methods=['GET'])
def health_check():
    status = 'ok' if model_ready.is_set() else 'loading'
    response = jsonify({'status': status})
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response
