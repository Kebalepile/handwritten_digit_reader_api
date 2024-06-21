import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array


def init_model(model_name):
    try:
        path = os.path.join(os.path.dirname(__file__), '..','models', model_name)  # Adjust path as necessary
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
