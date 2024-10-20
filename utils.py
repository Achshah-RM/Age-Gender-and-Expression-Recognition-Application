import numpy as np
from PIL import Image as PILImage
import tensorflow as tf

def preprocess_image_rgb(image):
    """
    Preprocesses an RGB image for a model that expects an input of shape (128, 128, 3).
    - Resizes image to 128x128.
    - Normalizes pixel values to [0, 1].
    - Ensures the image is of type FLOAT32.
    - Adds a batch dimension, resulting in shape (1, 128, 128, 3).
    """
    image = image.resize((128, 128))  # Resize to 128x128
    image = np.array(image) / 255.0  # Normalize pixel values
    image = image.astype(np.float32)  # Ensure FLOAT32
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def preprocess_image_grayscale(image):
    """
    Preprocesses a grayscale image for a model that expects an input of shape (128, 128, 1).
    - Resizes image to 128x128.
    - Converts image to grayscale (if not already).
    - Normalizes pixel values to [0, 1].
    - Ensures the image is of type FLOAT32.
    - Adds a batch dimension and a channel dimension, resulting in shape (1, 128, 128, 1).
    """
    image = image.resize((128, 128))  # Resize to 128x128
    image = image.convert('L')  # Convert to grayscale
    image = np.array(image) / 255.0  # Normalize pixel values
    image = image.astype(np.float32)  # Ensure FLOAT32
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = np.expand_dims(image, axis=-1)  # Add channel dimension for grayscale (128, 128, 1)
    return image

def load_tflite_model(model_path):
    """
    Loads the TensorFlow Lite model from the specified path and prepares it for inference.
    - Returns a model_inference function that takes preprocessed input data and returns the model's output.
    """
    try:
        # Load the TensorFlow Lite model
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

        # Get input and output details of the model
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Define the inference function
        def model_inference(image):
            """
            Performs inference on the provided preprocessed image using the loaded model.
            - image: A preprocessed image (e.g., output from preprocess_image_rgb or preprocess_image_grayscale).
            - Returns the output(s) of the model.
            """
            # Set the input tensor to the image
            interpreter.set_tensor(input_details[0]['index'], image)
            interpreter.invoke()  # Run inference
            
            # Extract and return the model output(s)
            return [interpreter.get_tensor(output_detail['index']) for output_detail in output_details]
        
        return model_inference
    
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None

