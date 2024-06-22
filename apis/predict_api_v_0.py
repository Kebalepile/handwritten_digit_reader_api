import os
import json
import time
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, redirect
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import logging

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_model(model_name):
    try:
        path = "./models/" + model_name  # Adjust path as necessary
        model = load_model(path)
        return model
    except Exception as e:
        print(f"Error loading the model: {e}")
        exit(1)

def prepare_image(image, target_size):
   
    if image.mode != 'L':
        image = image.convert('L')
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    
    return image

# Placeholder for the model
model = None

def load_model_before_fork():
    global model
    try:
        logger.info("Loading model before forking...")
        model = init_model('handwritten_digits_reader.h5')
        logger.info("Model loaded successfully before forking.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")


# Initialize Flask application
app = Flask(__name__)

# Enable CORS (Cross-Origin Resource Sharing) for the '/predict' endpoint
CORS(app, resources={r"/predict": {"origins": ["https://dipalo-tsa-motheo.github.io", "https://dipalo-tsa-motheo.github.io/"]}})

# Rate limiting configuration: 200 requests per day, 50 requests per hour
limiter = Limiter(get_remote_address, app=app, default_limits=["200 per day", "50 per hour"])

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
        if model is None:
            logger.error("Model is still loading, cannot process request.")
            return jsonify({'error': 'Model is still loading, please try again later'}), 503

        input_data = request.form.get('input')
        if not input_data:
            logger.error("No input data provided.")
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
    status = 'ok' if model is not None else 'loading'
    response = jsonify({'status': status})
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

if __name__ == '__main__':
    load_model_before_fork()
    app.run(host='0.0.0.0', port=10000)