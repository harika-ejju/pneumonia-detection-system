import streamlit as st
import numpy as np
import os
import tempfile
from PIL import Image
import time
import random
from pathlib import Path

# Set page configuration
st.set_page_config(
    page_title="Pneumonia Detection Assistant",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
def load_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
        
        :root {
            --primary: #1565C0;
            --primary-light: #1E88E5;
            --secondary: #26A69A;
            --accent: #FF5722;
            --background: #F5F7F9;
            --text: #263238;
            --light-text: #607D8B;
            --card-bg: #FFFFFF;
            --success: #4CAF50;
            --warning: #FFC107;
            --error: #E53935;
        }
        
        html, body, [class*="css"] {
            font-family: 'Poppins', sans-serif;
            color: var(--text);
        }
        
        .main {
            background-color: var(--background);
            padding: 1rem;
        }
        
        .stApp {
            background-color: var(--background);
        }
        
        h1, h2, h3 {
            color: var(--primary);
            font-weight: 600;
        }
        
        .card {
            background-color: var(--card-bg);
            border-radius: 10px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 1rem;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
        }
        
        .header-container {
            display: flex;
            align-items: center;
            margin-bottom: 1rem;
        }
        
        .header-icon {
            font-size: 2.5rem;
            margin-right: 0.5rem;
            color: var(--primary);
        }
        
        .header-text {
            display: inline-block;
        }
        
        .highlight {
            color: var(--accent);
            font-weight: 600;
        }
        
        .btn-primary {
            background-color: var(--primary);
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 5px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 1rem;
            margin: 0.5rem 0;
            transition: background-color 0.3s ease;
            border: none;
            cursor: pointer;
        }
        
        .btn-primary:hover {
            background-color: var(--primary-light);
        }
        
        .confidence-meter {
            margin: 1rem 0;
            height: 20px;
            background-color: #eceff1;
            border-radius: 10px;
            overflow: hidden;
        }
        
        .confidence-meter-fill {
            height: 100%;
            border-radius: 10px;
            transition: width 0.5s ease-out;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 0.8rem;
            font-weight: 500;
        }
        
        .footer {
            color: var(--light-text);
            font-size: 0.8rem;
            text-align: center;
            padding: 1rem 0;
            border-top: 1px solid #eceff1;
            margin-top: 2rem;
        }
        
        .result-normal {
            color: var(--success);
            font-weight: 600;
        }
        
        .result-pneumonia {
            color: var(--error);
            font-weight: 600;
        }
        
        .badge {
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.05rem;
        }
        
        .badge-normal {
            background-color: rgba(76, 175, 80, 0.2);
            color: var(--success);
        }
        
        .badge-pneumonia {
            background-color: rgba(229, 57, 53, 0.2);
            color: var(--error);
        }
        
        .sidebar-content {
            padding: 1rem;
        }
        
        .info-box {
            background-color: rgba(21, 101, 192, 0.1);
            border-left: 4px solid var(--primary);
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 0 5px 5px 0;
        }
        
        .image-container {
            display: flex;
            justify-content: center;
            align-items: center;
            margin: 1rem 0;
        }
        
        .image-container img {
            max-width: 100%;
            max-height: 300px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        /* Animation for loading */
        @keyframes pulse {
            0% { opacity: 0.6; }
            50% { opacity: 1; }
            100% { opacity: 0.6; }
        }
        
        .loading {
            animation: pulse 1.5s infinite;
        }
    </style>
    """, unsafe_allow_html=True)

# Mock functions for prediction (replace with actual model code)
def load_trained_model():
    # Mock function to simulate model loading
    time.sleep(1)
    return "Model Loaded"

def preprocess_image(image_path):
    # Mock function to preprocess an image
    try:
        img = Image.open(image_path)
        img = img.resize((224, 224))
        img = img.convert('RGB')
        return np.array(img) / 255.0
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

def predict(image):
    # Mock function to make a prediction
    # Replace with actual model prediction code
    time.sleep(2)
    result = random.choice(["NORMAL", "PNEUMONIA"])
    confidence = random.uniform(0.7, 0.98)
    return result, confidence

# Helper functions
def get_confidence_color(confidence):
    """Return a color based on confidence level"""
    if confidence < 0.75:
        return "#FFA726"  # Orange
    elif confidence < 0.9:
        return "#26A69A"  # Teal
    else:
        return "#66BB6A"  # Green

def create_example_dir():
    """Create a directory for storing example images"""
    example_dir = Path("examples")
    example_dir.mkdir(exist_ok=True)
    return example_dir

# Main app functions
def header():
    """Display the application header"""
    st.markdown("""
    <div class="header-container">
        <div class="header-icon">🫁</div>
        <div class="header-text">
            <h1>Pneumonia Detection Assistant</h1>
        </div>
    </div>
    <p>Upload a chest X-ray image to detect potential pneumonia indicators.</p>
    """, unsafe_allow_html=True)

def sidebar_content():
    """Display sidebar content"""
    st.sidebar.markdown("""
    <div class="sidebar-content">
        <h2>About Pneumonia</h2>
        <p>Pneumonia is an infection that inflames the air sacs in one or both lungs. The air sacs may fill with fluid or pus, causing symptoms such as cough, fever, chills, and difficulty breathing.</p>
        
        <div class="info-box">
            <h3>Key Indicators in X-rays</h3>
            <ul>
                <li>White opacities or consolidations</li>
                <li>Fluid in the pleural space</li>
                <li>Air bronchograms</li>
                <li>Silhouette sign</li>
            </ul>
        </div>
        
        <h3>Important Notes</h3>
        <p>This tool is for educational purposes only and should not replace professional medical advice. Always consult a healthcare provider for proper diagnosis and treatment.</p>
    </div>
    """, unsafe_allow_html=True)

def upload_section():
    """Display the file upload section"""
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<h2>Upload X-ray Image</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "jpeg", "png"])
    st.markdown("</div>", unsafe_allow_html=True)
    return uploaded_file

def sample_xrays_section():
    """Display sample X-ray images for comparison"""
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<h2>Sample X-rays</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h3>Normal Lung X-ray</h3>", unsafe_allow_html=True)
        st.markdown("""
        <div class="image-container">
            <img src="https://i.imgur.com/gGt7WoA.jpg" alt="Normal X-ray">
        </div>
        <p><span class="badge badge-normal">Normal</span> Clear lung fields without significant opacities.</p>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("<h3>Pneumonia X-ray</h3>", unsafe_allow_html=True)
        st.markdown("""
        <div class="image-container">
            <img src="https://i.imgur.com/ghLfsSU.jpg" alt="Pneumonia X-ray">
        </div>
        <p><span class="badge badge-pneumonia">Pneumonia</span> Note the white opacities indicating fluid in the lungs.</p>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

def process_prediction(uploaded_file):
    """Process the uploaded file and display prediction results"""
    if uploaded_file is not None:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<h2>Analysis Results</h2>", unsafe_allow_html=True)
        
        # Create columns for image and results
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("<h3>Uploaded X-ray</h3>", unsafe_allow_html=True)
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
        
        with col2:
            st.markdown("<h3>Diagnostic Assessment</h3>", unsafe_allow_html=True)
            
            # Temporary file for prediction
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                img_path = tmp_file.name
                image.save(img_path)
                
            # Display processing status
            with st.spinner('Analyzing X-ray...'):
                # Load model (mock)
                model = load_trained_model()
                
                # Preprocess image
                processed_img = preprocess_image(img_path)
                
                if processed_img is not None:
                    # Make prediction
                    result, confidence = predict(processed_img)
                
                # Clean up temp file
                try:
                    os.unlink(img_path)
                except:
                    pass
            
            # Display result
            result_class = "result-normal" if result == "NORMAL" else "result-pneumonia"
            badge_class = "badge-normal" if result == "NORMAL" else "badge-pneumonia"
            
            st.markdown(f"""
            <h4>Prediction: <span class="{result_class}">{result}</span></h4>
            <p>Confidence: <span class="highlight">{confidence:.2%}</span></p>
            
            <div class="confidence-meter">
                <div class="confidence-meter-fill" style="width: {confidence*100}%; background-color: {get_confidence_color(confidence)}">
                    {confidence:.0%}
                </div>
            </div>
            
            <p><span class="badge {badge_class}">{result}</span></p>
            """, unsafe_allow_html=True)
            
            # Recommendation based on result
            if result == "NORMAL":
                st.markdown("""
                <div class="info-box">
                    <h4>Assessment</h4>
                    <p>No significant indicators of pneumonia detected in this X-ray. The lung fields appear clear.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="info-box" style="border-left-color: #E53935;">
                    <h4>Assessment</h4>
                    <p>Potential indicators of pneumonia detected. Observe the areas of opacity which may represent fluid in the lungs.</p>
                    <p><strong>Recommendation:</strong> Please consult a healthcare professional for proper diagnosis and treatment.</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

def footer():
    """Display footer information"""
    st.markdown("""
    <div class="footer">
        <p>Pneumonia Detection Assistant | Developed for educational purposes only</p>
        <p>© 2023 | Not for clinical use</p>
    </div>
    """, unsafe_allow_html=True)

# Main app
def main():
    # Load custom CSS
    load_css()
    
    # App layout
    header()
    sidebar_content()
    
    # Main content in columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = upload_section()
        process_prediction(uploaded_file)
    
    with col2:
        sample_xrays_section()
    
    footer()

if __name__ == "__main__":
    main()

import streamlit as st
import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
import tempfile
from PIL import Image
import io
import base64
from pathlib import Path
import requests

# Import our prediction module (assuming it exists in scripts/predict.py)
# If it doesn't exist, we'll provide mock functionality
try:
    from scripts.predict import load_trained_model, preprocess_image, predict
    model_available = True
except ImportError:
    model_available = False
    print("Prediction module not found, using mock functionality")

# Set page configuration
st.set_page_config(
    page_title="Pneumonia Detection AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define color scheme
COLORS = {
    "primary": "#1E88E5",
    "secondary": "#26A69A",
    "background": "#F5F7FA",
    "text": "#333333",
    "accent": "#FF5252",
    "success": "#4CAF50",
    "warning": "#FFC107",
    "light": "#ECEFF1",
    "dark": "#263238",
    "white": "#FFFFFF",
    "black": "#000000",
    "header": "#0D47A1",
    "card": "#FFFFFF",
    "gradient_start": "#1E88E5",
    "gradient_end": "#0D47A1"
}

# Define custom CSS
def load_css():
    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
        
        html, body, [class*="css"] {{
            font-family: 'Poppins', sans-serif;
            color: {COLORS["text"]};
        }}
        
        h1, h2, h3, h4, h5, h6 {{
            font-family: 'Poppins', sans-serif;
            font-weight: 600;
            color: {COLORS["header"]};
        }}
        
        .stApp {{
            background-color: {COLORS["background"]};
        }}
        
        .main-header {{
            background: linear-gradient(135deg, {COLORS["gradient_start"]}, {COLORS["gradient_end"]});
            padding: 1.5rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        
        .card {{
            background-color: {COLORS["card"]};
            border-radius: 10px;
            padding: 1.5rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 1rem;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}
        
        .card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
        }}
        
        .metric-card {{
            background-color: {COLORS["card"]};
            border-radius: 10px;
            padding: 1rem;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            border-left: 4px solid {COLORS["primary"]};
        }}
        
        .prediction-normal {{
            background-color: {COLORS["success"]};
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 5px;
            font-weight: 600;
            display: inline-block;
        }}
        
        .prediction-pneumonia {{
            background-color: {COLORS["accent"]};
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 5px;
            font-weight: 600;
            display: inline-block;
        }}
        
        .stButton>button {{
            background-color: {COLORS["primary"]};
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 5px;
            font-weight: 500;
            transition: all 0.3s ease;
        }}
        
        .stButton>button:hover {{
            background-color: {COLORS["gradient_end"]};
            box-shadow: 0 5px 10px rgba(0, 0, 0, 0.1);
        }}
        
        .xray-image {{
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            max-width: 100%;
            transition: transform 0.3s ease;
        }}
        
        .xray-image:hover {{
            transform: scale(1.05);
        }}
        
        .info-text {{
            background-color: {COLORS["light"]};
            border-left: 4px solid {COLORS["primary"]};
            padding: 1rem;
            border-radius: 5px;
            margin: 1rem 0;
        }}
        
        .footer {{
            text-align: center;
            margin-top: 2rem;
            padding: 1rem;
            background-color: {COLORS["dark"]};
            color: {COLORS["white"]};
            border-radius: 10px;
        }}
        
        .animate-pulse {{
            animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
        }}
        
        @keyframes pulse {{
            0%, 100% {{
                opacity: 1;
            }}
            50% {{
                opacity: 0.7;
            }}
        }}
        
        .confidence-meter {{
            height: 20px;
            background-color: #f0f0f0;
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }}
        
        .confidence-fill {{
            height: 100%;
            border-radius: 10px;
            background: linear-gradient(90deg, #FFC107 0%, #4CAF50 100%);
            transition: width 0.5s ease;
        }}
        
        .divider {{
            height: 1px;
            background-color: #e0e0e0;
            margin: 1.5rem 0;
        }}
        
        .badge {{
            display: inline-block;
            padding: 0.25rem 0.5rem;
            background-color: {COLORS["light"]};
            color: {COLORS["text"]};
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-right: 0.5rem;
        }}
        
        .badge-primary {{
            background-color: {COLORS["primary"]};
            color: white;
        }}
        
        .badge-secondary {{
            background-color: {COLORS["secondary"]};
            color: white;
        }}
        
        .example-section {{
            margin-top: 2rem;
        }}
        
        .example-image {{
            border: 2px solid {COLORS["light"]};
            border-radius: 10px;
            padding: 0.25rem;
            transition: all 0.3s ease;
        }}
        
        .example-image:hover {{
            border-color: {COLORS["primary"]};
            transform: scale(1.03);
        }}
    </style>
    """, unsafe_allow_html=True)

# Load the custom CSS
load_css()

# Create directory for example images if it doesn't exist
example_dir = Path("static/examples")
example_dir.mkdir(parents=True, exist_ok=True)

# Define function to get/download example images
def get_example_images():
    normal_image_path = example_dir / "normal.jpg"
    pneumonia_image_path = example_dir / "pneumonia.jpg"
    
    # URLs for sample images (these are from reliable medical sources)
    normal_url = "https://prod-images-static.radiopaedia.org/images/53748702/e4fe194def4ec2ec09f5b5a6b71abb_gallery.jpeg"
    pneumonia_url = "https://prod-images-static.radiopaedia.org/images/53870893/52ba6834fcb073c8c3534f0801ad05_gallery.jpeg"
    
    # Download if not exists
    if not normal_image_path.exists():
        try:
            response = requests.get(normal_url)
            with open(normal_image_path, "wb") as f:
                f.write(response.content)
        except Exception as e:
            st.warning(f"Failed to download normal example image: {e}")
    
    if not pneumonia_image_path.exists():
        try:
            response = requests.get(pneumonia_url)
            with open(pneumonia_image_path, "wb") as f:
                f.write(response.content)
        except Exception as e:
            st.warning(f"Failed to download pneumonia example image: {e}")
    
    return normal_image_path, pneumonia_image_path

# Mock prediction function if the real one is not available
def mock_predict(image_path):
    # Simulate processing time
    time.sleep(1.5)
    # Random prediction for demo purposes
    prediction = np.random.choice(["NORMAL", "PNEUMONIA"], p=[0.4, 0.6])
    confidence = np.random.uniform(0.7, 0.95)
    return prediction, confidence

# Function to create a sidebar with educational information
def create_sidebar():
    st.sidebar.markdown("<h2 style='text-align: center;'>About Pneumonia</h2>", unsafe_allow_html=True)
    
    st.sidebar.markdown("""
    <div class="info-text">
        <p>Pneumonia is an infection that inflames the air sacs in one or both lungs. 
        The air sacs may fill with fluid or pus, causing symptoms such as cough with phlegm, 
        fever, chills, and difficulty breathing.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("### Key Facts")
    st.sidebar.markdown("""
    - Pneumonia can be life-threatening, especially for infants and elderly people
    - Pneumonia affects millions of people worldwide each year
    - It can be caused by bacteria, viruses, or fungi
    - Chest X-rays are a key diagnostic tool
    """)
    
    st.sidebar.markdown("### X-ray Indicators")
    st.sidebar.markdown("""
    <div class="card">
        <p><span class="badge badge-primary">Normal X-rays</span> show clear lung fields with well-defined heart and diaphragm borders.</p>
        <p><span class="badge badge-secondary">Pneumonia X-rays</span> often display white opacities (consolidations) where air should be, indicating infection.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("### AI Detection Benefits")
    st.sidebar.markdown("""
    - Assists radiologists in diagnosis
    - Can help prioritize urgent cases
    - Provides objective assessment
    - Potentially improves healthcare in underserved areas
    """)

# Function to create the header section
def create_header():
    st.markdown("""
    <div class="main-header">
        <h1>🫁 Pneumonia Detection AI</h1>
        <p>Upload a chest X-ray image to detect potential pneumonia cases using artificial intelligence</p>
    </div>
    """, unsafe_allow_html=True)

# Function to display prediction results
def display_prediction(image, prediction, confidence):
    if prediction == "NORMAL":
        prediction_class = "prediction-normal"
        recommendation = "No signs of pneumonia detected. Always consult with a healthcare professional for definitive diagnosis."
    else:
        prediction_class = "prediction-pneumonia"
        recommendation = "Potential signs of pneumonia detected. Please consult with a healthcare professional for proper diagnosis and treatment."
    
    st.markdown(f"""
    <div class="card">
        <h3>AI Prediction Result</h3>
        <div style="display: flex; align-items: center; gap: 10px;">
            <span class="{prediction_class}">{prediction}</span>
            <span>with {confidence:.1%} confidence</span>
        </div>
        
        <div class="confidence-meter">
            <div class="confidence-fill" style="width: {confidence * 100}%;"></div>
        </div>
        
        <div class="divider"></div>
        
        <h4>Recommendation</h4>
        <p>{recommendation}</p>
    </div>
    """, unsafe_allow_html=True)

# Function to display example images
def display_examples():
    st.markdown("<h3>Understanding X-ray Results</h3>", unsafe_allow_html=True)
    
    normal_path, pneumonia_path = get_example_images()
    
    # Check if example images were successfully downloaded
    normal_exists = normal_path.exists()
    pneumonia_exists = pneumonia_path.exists()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h4>Normal X-ray</h4>", unsafe_allow_html=True)
        if normal_exists:
            st.image(str(normal_path), use_column_width=True, caption="Normal chest X-ray with clear lung fields")
        else:
            st.warning("Example image not available")
        
        st.markdown("""
        <div class="info-text">
            <p>Normal chest X-rays typically show:</p>
            <ul>
                <li>Clear lung fields</li>
                <li>Well-defined heart border</li>
                <li>Visible lung markings without opacities</li>
                <li>Sharp costophrenic angles</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("<h4>Pneumonia X-ray</h4>", unsafe_allow_html=True)
        if pneumonia_exists:
            st.image(str(pneumonia_path), use_column_width=True, caption="Pneumonia chest X-ray with opacities")
        else:
            st.warning("Example image not available")
        
        st.markdown("""
        <div class="info-text">
            <p>Pneumonia chest X-rays often show:</p>
            <ul>
                <li>White opacities/consolidations in lung fields</li>
                <li>Sometimes blurred heart borders</li>
                <li>Areas of increased density</li>
                <li>Possible pleural effusions (fluid)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Function to display metrics and additional information
def display_metrics():
    st.markdown("<h3>Model Performance Metrics</h3>", unsafe_allow_html=True)
    
    # Create a 3x2 grid for metrics
    col1

import streamlit as st
import numpy as np
import pandas as pd
import os
import time
import base64
from PIL import Image
import io
import matplotlib.pyplot as plt
import tempfile
import requests
from datetime import datetime

# For the mock prediction function - in real app, replace with model prediction
import random

# Set page configuration
st.set_page_config(
    page_title="PneumoScan AI | Pneumonia Detection",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define color palette
colors = {
    'primary': '#3498db',
    'primary_dark': '#2980b9',
    'secondary': '#e74c3c',
    'accent': '#2ecc71',
    'background': '#f8f9fa',
    'card_bg': '#ffffff',
    'text_primary': '#2c3e50',
    'text_secondary': '#7f8c8d',
    'normal': '#2ecc71',
    'pneumonia': '#e74c3c',
    'gradient_start': '#3498db',
    'gradient_end': '#2980b9',
}

# App header and custom CSS
def load_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@300;400;600&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
    }
    
    p, li, span {
        font-family: 'Open Sans', sans-serif;
    }
    
    .main > div {
        padding: 1rem 3rem;
        max-width: 100%;
    }
    
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Custom Header */
    .header {
        background: linear-gradient(120deg, #3498db, #2980b9);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 8px 20px rgba(52, 152, 219, 0.3);
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .header p {
        font-size: 1.2rem;
        opacity: 0.9;
        margin-top: 10px;
    }
    
    .header::before {
        content: "";
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0) 60%);
        transform: rotate(15deg);
    }
    
    /* Card Styling */
    .card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.12);
    }
    
    .card-header {
        border-bottom: 1px solid #f0f0f0;
        padding-bottom: 1rem;
        margin-bottom: 1rem;
    }
    
    .card-header h3 {
        color: #2c3e50;
        margin: 0;
        display: flex;
        align-items: center;
    }
    
    .card-icon {
        margin-right: 10px;
        background: #f0f7ff;
        width: 30px;
        height: 30px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: 50%;
        color: #3498db;
    }
    
    /* Button Styling */
    .custom-btn {
        background: linear-gradient(45deg, #3498db, #2980b9);
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 50px;
        font-weight: 500;
        cursor: pointer;
        display: inline-block;
        text-align: center;
        box-shadow: 0 4px 10px rgba(52, 152, 219, 0.3);
        transition: all 0.3s ease;
    }
    
    .custom-btn:hover {
        box-shadow: 0 6px 15px rgba(52, 152, 219, 0.4);
        transform: translateY(-2px);
    }
    
    /* Badge Styling */
    .badge {
        font-size: 0.75rem;
        padding: 5px 10px;
        border-radius: 50px;
        font-weight: 600;
        display: inline-block;
    }
    
    .badge-normal {
        background-color: rgba(46, 204, 113, 0.15);
        color: #2ecc71;
    }
    
    .badge-pneumonia {
        background-color: rgba(231, 76, 60, 0.15);
        color: #e74c3c;
    }
    
    /* Upload Zone */
    .upload-zone {
        border: 2px dashed #d1d8e0;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
        background-color: rgba(52, 152, 219, 0.05);
    }
    
    .upload-zone:hover {
        border-color: #3498db;
        background-color: rgba(52, 152, 219, 0.1);
    }
    
    /* Result Display */
    .result-normal {
        font-size: 1.5rem;
        color: #2ecc71;
        font-weight: 600;
        padding: 10px 20px;
        border-radius: 10px;
        background-color: rgba(46, 204, 113, 0.1);
        display: inline-block;
        margin-top: 10px;
    }
    
    .result-pneumonia {
        font-size: 1.5rem;
        color: #e74c3c;
        font-weight: 600;
        padding: 10px 20px;
        border-radius: 10px;
        background-color: rgba(231, 76, 60, 0.1);
        display: inline-block;
        margin-top: 10px;
    }
    
    /* Progress Meter */
    .progress-container {
        width: 100%;
        background-color: #f0f0f0;
        border-radius: 50px;
        height: 10px;
        margin: 15px 0;
        overflow: hidden;
    }
    
    .progress-bar {
        height: 100%;
        border-radius: 50px;
        background: linear-gradient(90deg, #3498db, #2ecc71);
        transition: width 0.5s ease;
    }
    
    /* Info Cards */
    .info-card {
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
        border-left: 4px solid;
    }
    
    .info-primary {
        background-color: rgba(52, 152, 219, 0.1);
        border-left-color: #3498db;
    }
    
    .info-warning {
        background-color: rgba(241, 196, 15, 0.1);
        border-left-color: #f1c40f;
    }
    
    .info-danger {
        background-color: rgba(231, 76, 60, 0.1);
        border-left-color: #e74c3c;
    }
    
    /* Example Images */
    .example-container {
        display: flex;
        justify-content: space-between;
        margin-top: 15px;
    }
    
    .example-item {
        width: 48%;
        text-align: center;
        padding: 10px;
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    
    .example-item img {
        max-width: 100%;
        border-radius: 5px;
        transition: all 0.3s ease;
    }
    
    .example-item:hover img {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    /* Animated Pulse */
    @keyframes pulse {
        0% {
            box-shadow: 0 0 0 0 rgba(52, 152, 219, 0.4);
        }
        70% {
            box-shadow: 0 0 0 10px rgba(52, 152, 219, 0);
        }
        100% {
            box-shadow: 0 0 0 0 rgba(52, 152, 219, 0);
        }
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    /* Tooltip */
    .tooltip {
        position: relative;
        display: inline-block;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 120px;
        background-color: #333;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        transform: translateX(-50%);
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 20px;
        margin-top: 30px;
        color: #7f8c8d;
        border-top: 1px solid #eee;
    }
    
    /* Sidebar */
    .sidebar-content h3 {
        color: #3498db;
        border-bottom: 2px solid #eee;
        padding-bottom: 10px;
        margin-bottom: 15px;
    }
    
    /* Loading Animation */
    .loader {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #3498db;
        border-radius: 50%;
        width: 30px;
        height: 30px;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Image Caption */
    .image-caption {
        font-size: 0.8rem;
        color: #7f8c8d;
        text-align: center;
        margin-top: 5px;
    }
    
    /* Stat Card */
    .stat-container {
        display: flex;
        justify-content: space-between;
        margin-top: 15px;
    }
    
    .stat-card {
        flex: 1;
        padding: 15px;
        background: white;
        border-radius: 10px;
        text-align: center;
        margin: 0 5px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    
    .stat-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #3498db;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: #7f8c8d;
    }
    
    /* Alert Box */
    .alert {
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    
    .alert-info {
        background-color: rgba(52, 152, 219, 0.15);
        border-left: 4px solid #3498db;
    }
    
    .alert-warning {
        background-color: rgba(241, 196, 15, 0.15);
        border-left: 4px solid #f1c40f;
    }
    
    .alert-danger {
        background-color: rgba(231, 76, 60, 0.15);
        border-left: 4px solid #e74c3c;
    }
    </style>
    """, unsafe_allow_html=True)

# Mock functions to simulate model prediction
def preprocess_image(img):
    # In a real app, you would preprocess the image for your model
    img = img.resize((224, 224))
    return img

def predict(img_path):
    # In a real app, this would use your trained model
    # For demo, we'll just return random results
    img = Image.open(img_path)
    img = preprocess_image(img)
    
    # Simulate processing delay
    time.sleep(1)
    
    # Random prediction (replace with actual model prediction)
    prediction = random.choice(["NORMAL", "PNEUMONIA"])
    confidence = random.uniform(0.6, 0.99)
    
    return prediction, confidence, img

# Download example images if they don't exist
def get_example_images():
    # Create directory if it doesn't exist
    os.makedirs("examples", exist_ok=True)
    
    # Example image URLs
    normal_url = "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/00000001_000.png"
    pneumonia_url = "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/00000099_002.png"
    
    # Download the images if they don't exist
    normal_path = "examples/normal_example.png"
    pneumonia_path = "examples/pneumonia_example.png"
    
    if not os.path.exists(normal_path):
        with open(normal_path, "wb") as f:
            f.write(requests.get(normal_url).content)
    
    if not os.

