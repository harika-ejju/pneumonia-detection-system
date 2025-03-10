#!/usr/bin/env python3
"""
Pneumonia Detection Prediction Script

This script loads a trained pneumonia detection model and makes predictions on chest X-ray images.
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input

# Define constants
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, 'pneumonia_model.h5')
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']
TARGET_SIZE = (224, 224)  # Model input size


def load_trained_model(model_path=DEFAULT_MODEL_PATH):
    """
    Load the trained pneumonia detection model.
    
    Args:
        model_path (str): Path to the trained model file
        
    Returns:
        The loaded model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")
    
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    print("Model loaded successfully.")
    
    return model


def preprocess_image(image_path, target_size=TARGET_SIZE):
    """
    Preprocess an image for prediction.
    
    Args:
        image_path (str): Path to the image file
        target_size (tuple): Target size for the image (height, width)
        
    Returns:
        Preprocessed image as a numpy array
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found at {image_path}")
    
    # Load and resize the image
    img = load_img(image_path, target_size=target_size)
    
    # Convert to array and expand dimensions for batch
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Preprocess the image (normalize)
    img_array = preprocess_input(img_array)
    
    return img_array


def predict(model, image_path):
    """
    Make a prediction on an image.
    
    Args:
        model: The trained model
        image_path (str): Path to the image file
        
    Returns:
        Tuple of (class_name, confidence)
    """
    # Preprocess the image
    processed_image = preprocess_image(image_path)
    
    # Make prediction
    predictions = model.predict(processed_image)
    
    # Get the class and confidence
    if predictions.shape[1] == 1:  # Binary classification with sigmoid
        confidence = float(predictions[0][0])
        predicted_class_idx = 1 if confidence > 0.5 else 0
        confidence = confidence if predicted_class_idx == 1 else 1 - confidence
    else:  # Multi-class with softmax
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
    
    return CLASS_NAMES[predicted_class_idx], confidence


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Predict pneumonia from chest X-ray images')
    parser.add_argument('--image', required=True, help='Path to the chest X-ray image')
    parser.add_argument('--model', default=DEFAULT_MODEL_PATH, help='Path to the trained model')
    
    args = parser.parse_args()
    
    try:
        # Load model
        model = load_trained_model(args.model)
        
        # Make prediction
        class_name, confidence = predict(model, args.image)
        
        # Print results
        print(f"Prediction: {class_name}")
        print(f"Confidence: {confidence:.2%}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

