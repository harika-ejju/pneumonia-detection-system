import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os
import time
import base64
from PIL import Image
import io
import tempfile
import glob
import random

# Set page configuration
st.set_page_config(
    page_title="Pneumonia Detection System",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Pneumonia Detection System"
    }
)

# Set theme to dark mode
st.markdown("""
    <style>
        [data-testid="stAppViewContainer"], [data-testid="stSidebar"] {
            background-color: #0E1117;
        }
    </style>
""", unsafe_allow_html=True)

# Color scheme - using vibrant colors
PRIMARY = "#001f3f"    # Navy blue
SECONDARY = "#3D9970"  # Olive green
ACCENT = "#01FF70"     # Lime green
SUCCESS = "#01FF70"    # Lime green
WARNING = "#FFDC00"    # Yellow
DANGER = "#FF4136"     # Red
LIGHT = "#f8f9fa"      # Light gray
DARK = "#121212"       # Dark gray
BACKGROUND = "#0E1117" # Dark background
CARD_BG = "#1E2130"    # Dark card background

# Custom CSS
def local_css():
    st.markdown("""
    <style>
        /* Font imports */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&family=Open+Sans:wght@300;400;600&display=swap');
        
        /* Base styles */
        html, body, [class*="css"] {
            font-family: 'Open Sans', sans-serif;
            color: #f8f9fa;
        }
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Poppins', sans-serif;
            font-weight: 600;
            color: #3D9970;
        }
        
        /* Layout */
        .main .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
            margin-bottom: 0.8rem;
        }
        
        /* Reduce spacing between elements */
        p, div, h1, h2, h3, h4, h5, h6 {
            margin-bottom: 0.5rem;
        }
        
        /* Custom card */
        .card {
            border-radius: 10px;
            background-color: #1E2130;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            padding: 1.2rem;
            margin-bottom: 1rem;
            transition: all 0.3s ease;
            color: #f8f9fa; /* Ensure text is light on dark background */
            border-left: 3px solid #3D9970;
        }
        .card:hover {
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
            transform: translateY(-2px);
        }
        
        /* Header */
        .app-header {
            background: linear-gradient(90deg, #001f3f, #3D9970);
            color: white;
            padding: 1.2rem;
            border-radius: 10px;
            margin-bottom: 1rem;
        }
        .app-header h1 {
            color: white;
            margin: 0;
        }
        
        /* File uploader */
        .uploadFile > label {
            color: #01FF70 !important;
            font-weight: 600;
        }
        
        /* Badges */
        .badge {
            padding: 0.5rem 1rem;
            border-radius: 50px;
            font-weight: 600;
            display: inline-block;
            font-size: 0.875rem;
        }
        .badge-normal {
            background-color: #01FF70;
            color: #001f3f;
            font-weight: bold;
        }
        .badge-pneumonia {
            background-color: #FF4136;
            color: white;
            font-weight: bold;
        }
        
        /* Button */
        .custom-button {
            background: linear-gradient(90deg, #3D9970, #01FF70);
            color: #001f3f;
            padding: 0.5rem 1rem;
            border-radius: 5px;
            border: none;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.3s ease;
            display: inline-block;
            text-align: center;
        }
        .custom-button:hover {
            opacity: 0.9;
            transform: translateY(-1px);
        }
        
        /* Progress bar */
        .stProgress > div > div {
            background-color: #3D9970;
        }
        
        /* Sidebar */
        .css-1cypcdb, [data-testid="stSidebar"] {
            background-color: #161a24;
        }
        
        /* Footer */
        .footer {
            text-align: center;
            margin-top: 1.5rem;
            padding-top: 0.8rem;
            border-top: 1px solid #3D9970;
            color: #f8f9fa;
            font-size: 0.875rem;
        }
        .footer a {
            color: #01FF70;
            text-decoration: none;
        }
        .footer a:hover {
            text-decoration: underline;
        }
        
        /* Confidence meter */
        .confidence-meter {
            margin-top: 1rem;
            margin-bottom: 1rem;
        }
        .confidence-label {
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.25rem;
        }
        .confidence-bar {
            height: 10px;
            background-color: #161a24;
            border-radius: 5px;
            overflow: hidden;
            position: relative;
        }
        .confidence-fill {
            height: 100%;
            border-radius: 5px;
            transition: width 0.5s ease;
            background: linear-gradient(90deg, #01FF70, #3D9970);
        }
        
        /* Image container */
        .img-container {
            text-align: center;
            margin-bottom: 1.5rem;
        }
        .img-container img {
            max-width: 100%;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        
        /* Animation */
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        .animated {
            animation: fadeIn 0.5s ease;
        }
    </style>
    """, unsafe_allow_html=True)

# Function to load and preprocess the image
def preprocess_image(image_path, target_size=(150, 150)):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Function to make prediction (mock function for now)
def predict(img_array):
    # This would normally use the actual model
    # model = load_model('models/pneumonia_detection_model.h5')
    # prediction = model.predict(img_array)
    
    # Mock prediction for demonstration
    import random
    time.sleep(1)  # Simulate processing time
    prediction = random.random()
    
    if prediction > 0.5:
        return "PNEUMONIA", prediction
    else:
        return "NORMAL", 1 - prediction

# App header component
def render_header():
    st.markdown("""
    <div class="app-header animated">
        <h1>Pneumonia Detection System</h1>
        <p style="color: white;">Upload a chest X-ray image to detect pneumonia using our AI-powered analysis tool.</p>
    </div>
    """, unsafe_allow_html=True)

# Card component
def render_card(title, content, key=None):
    st.markdown(f"""
    <div class="card animated" id="{key if key else ''}">
        <h3>{title}</h3>
        {content}
    </div>
    """, unsafe_allow_html=True)

# Result display component
def render_result(prediction, confidence):
    if prediction == "PNEUMONIA":
        badge_class = "badge-pneumonia"
        fill_color = "#FF4136"
    else:
        badge_class = "badge-normal"
        fill_color = "#01FF70"
    
    confidence_percentage = int(confidence * 100)
    
    result_html = f"""
    <h2>Analysis Result</h2>
    <div class="animated">
        <div>
            <span class="badge {badge_class}">{prediction}</span>
            <span style="margin-left: 1rem; font-size: 0.875rem;">Confidence: {confidence_percentage}%</span>
        </div>
        
        <div class="confidence-meter">
            <div class="confidence-label">
                <span>0%</span>
                <span>50%</span>
                <span>100%</span>
            </div>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: {confidence_percentage}%; background-color: {fill_color};"></div>
            </div>
        </div>
        
        <div>
            <h4>Assessment:</h4>
            <p style="color: #f8f9fa;">
                {("Our analysis indicates a high likelihood of pneumonia. Please consult with a healthcare professional for proper diagnosis and treatment." 
                if prediction == "PNEUMONIA" else 
                "The X-ray appears normal. However, if you're experiencing symptoms, please consult with a healthcare professional.")}
            </p>
        </div>
    """
    
    render_card("Result", result_html, "result-card")

# Function to get random sample image from dataset
def get_sample_image(category):
    # Define possible paths where dataset images might be located
    dataset_paths = [
        f"data/chest_xray/train/{category}",
        f"data/chest_xray/val/{category}",
        f"data/chest_xray/test/{category}"
    ]
    
    # Default fallback images (base64 encoded) in case no local images are found
    fallback_images = {
        "NORMAL": "https://i.imgur.com/IUEwNPw.jpeg",
        "PNEUMONIA": "https://i.imgur.com/VGcb8iO.jpeg"
    }
    
    # Try to find images in the dataset
    for path in dataset_paths:
        if os.path.exists(path):
            images = glob.glob(f"{path}/*.jpeg") + glob.glob(f"{path}/*.jpg") + glob.glob(f"{path}/*.png")
            if images:
                return images[random.randint(0, len(images)-1)]
    
    # If no images found, return the fallback URL
    return fallback_images[category]

# Sample images component
def render_sample_images():
    st.markdown("""
    <h3>X-ray Reference Images</h3>
    <p style="color: #f8f9fa;">Compare your X-ray with these reference images:</p>
    """, unsafe_allow_html=True)
    
    # Get sample images
    normal_image = get_sample_image("NORMAL")
    pneumonia_image = get_sample_image("PNEUMONIA")
    
    # Add spacing before images
    st.markdown("""<div style="height: 20px;"></div>""", unsafe_allow_html=True)
    
    # Create two columns for side by side layout
    col1, col2 = st.columns(2)
    
    # Normal X-ray in first column
    with col1:
        st.markdown("""
        <div class="card">
            <h4 style="font-size: 1.4rem; white-space: nowrap;">Normal X-ray</h4>
            <div class="img-container">
        """, unsafe_allow_html=True)
        
        # Check if the image is a local file or a URL
        if os.path.exists(normal_image):
            st.image(normal_image, caption="Normal X-ray Sample", use_column_width=True)
        else:
            st.markdown(f"""
                <img src="{normal_image}" alt="Normal X-ray" style="max-width:100%; border-radius:5px;">
            """, unsafe_allow_html=True)
            
        st.markdown("""
            </div>
            <div style="margin-top: 1rem;">
                <p style="color: #f8f9fa;"><strong>Characteristics:</strong> Clear lung fields without white spots or opacities.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Pneumonia X-ray in second column
    with col2:
        st.markdown("""
        <div class="card">
            <h4 style="font-size: 1.4rem; white-space: nowrap;">Pneumonia X-ray</h4>
            <div class="img-container">
        """, unsafe_allow_html=True)
        
        # Check if the image is a local file or a URL
        if os.path.exists(pneumonia_image):
            st.image(pneumonia_image, caption="Pneumonia X-ray Sample", use_column_width=True)
        else:
            st.markdown(f"""
                <img src="{pneumonia_image}" alt="Pneumonia X-ray" style="max-width:100%; border-radius:5px;">
            """, unsafe_allow_html=True)
        
        st.markdown("""
            </div>
            <div style="margin-top: 1rem;">
                <p style="color: #f8f9fa;"><strong>Characteristics:</strong> White spots or areas of opacity in the lung fields.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
# Sidebar content component
def render_sidebar():
    st.sidebar.markdown("""
    <h2>About Pneumonia</h2>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("""
    <div class="card">
        <h4>What is Pneumonia?</h4>
        <p style="color: #f8f9fa;">Pneumonia is an infection that inflames the air sacs in one or both lungs. The air sacs may fill with fluid or pus, causing symptoms such as cough with phlegm, fever, chills, and difficulty breathing.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("""
    <div class="card">
        <h4>Risk Factors</h4>
        <ul style="color: #f8f9fa;">
            <li>Age (very young or over 65)</li>
            <li>Weakened immune system</li>
            <li>Chronic diseases</li>
            <li>Smoking</li>
            <li>Hospitalization</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("""
    <div class="card">
        <h4>X-ray Analysis</h4>
        <p style="color: #f8f9fa;">Radiologists look for white spots (called "infiltrates") in the lungs, which may indicate an infection. Our AI system has been trained to identify these patterns.</p>
    </div>
    """, unsafe_allow_html=True)

# Footer component
def render_footer():
    st.markdown("""
    <div class="footer">
        <p>Pneumonia Detection System © 2023 | Developed with love by harika_ejju</p>
        <p><small>This application is for educational purposes only and not intended to replace professional medical advice.</small></p>
    </div>
    """, unsafe_allow_html=True)

# Main application
def main():
    # Apply custom CSS
    local_css()
    
    # Render sidebar
    render_sidebar()
    
    # App header
    render_header()
    
    # Main content with increased spacing
    st.markdown("""<div style="height: 20px;"></div>""", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("""
        <div class="card">
            <h3>Upload X-ray Image</h3>
            <p style="color: #f8f9fa; margin-bottom: 1.5rem;">Please upload a chest X-ray image in JPEG, JPG, or PNG format:</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Choose an X-ray image", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Save the uploaded file to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_path = temp_file.name
            
            # Display the uploaded image
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded X-ray Image", use_column_width=True)
            
            # Process the image
            with st.spinner("Analyzing image..."):
                img_array = preprocess_image(temp_path)
                prediction, confidence = predict(img_array)
            
            # Clean up the temporary file
            os.unlink(temp_path)
            
            # Show the result in the second column
            with col2:
                render_result(prediction, confidence)
    
    # Reference images - display side by side in a row when no file is uploaded
    if uploaded_file is None:
        st.markdown("""<div style="height: 30px;"></div>""", unsafe_allow_html=True)
        render_sample_images()
    
    # Footer
    render_footer()

if __name__ == "__main__":
    main()

