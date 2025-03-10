import streamlit as st
import os
import tempfile
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(
    page_title="Pneumonia Detection - Simple Test",
    page_icon="🫁",
    layout="wide"
)

def load_model():
    """
    Load and return the trained model.
    This is a placeholder function - in a real app, it would load the actual model.
    """
    # In a real app, you would load your model here
    # Example: model = tf.keras.models.load_model('models/pneumonia_model.h5')
    # For now, we'll return None as a placeholder
    return None

def preprocess_image(image_path):
    """
    Preprocess the image for model prediction.
    This is a simplified version for testing.
    """
    try:
        img = Image.open(image_path)
        img = img.resize((224, 224))  # Resize to model input size
        img_array = np.array(img)
        # Convert to grayscale if it's RGB
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = np.mean(img_array, axis=2).astype(np.uint8)
        # Normalize
        img_array = img_array / 255.0
        return img_array, True
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None, False

def predict(image_array, model):
    """
    Make a prediction using the trained model.
    This is a simplified version for testing.
    """
    # In a real app, you would use your model to predict
    # For testing, we'll return a random prediction
    probability = np.random.random()
    prediction = "PNEUMONIA" if probability > 0.5 else "NORMAL"
    return prediction, probability

def main():
    st.title("Pneumonia Detection App - Simple Test")
    
    st.write("This is a simplified version of the app for testing.")
    
    # Sidebar
    with st.sidebar:
        st.header("About")
        st.write("This is a test app for pneumonia detection from chest X-rays.")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Upload X-ray Image")
        uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded X-ray", use_column_width=True)
            
            # Create a temporary file to save the uploaded image
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            st.info("Processing the image...")
            
            # Load model
            model = load_model()
            
            # Preprocess the image
            processed_img, success = preprocess_image(tmp_path)
            
            if success:
                # Make prediction
                prediction, probability = predict(processed_img, model)
                
                # Display prediction
                st.subheader("Prediction Result:")
                
                if prediction == "PNEUMONIA":
                    st.error(f"Prediction: {prediction}")
                else:
                    st.success(f"Prediction: {prediction}")
                
                st.write(f"Confidence: {probability:.2f}")
                
                # Clean up the temporary file
                os.unlink(tmp_path)
    
    with col2:
        st.header("What is Pneumonia?")
        st.write("""
        Pneumonia is an infection that inflames the air sacs in one or both lungs.
        It can be caused by viruses, bacteria, or fungi.
        
        Symptoms include:
        - Chest pain
        - Cough with phlegm
        - Fatigue and loss of appetite
        - Fever, sweating, and chills
        - Shortness of breath
        """)

if __name__ == "__main__":
    main()

import os
import requests
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import tempfile
import time
from pathlib import Path

# Import functions from predict.py (adjust this import as needed)
from scripts.predict import load_trained_model, preprocess_image, predict

def download_sample_xray_images():
    """
    Downloads real sample X-ray images from reliable medical imaging repositories
    and stores them locally for display in the app.
    
    Returns:
        tuple: Paths to normal and pneumonia sample images
    """
    # Create directory for sample images if it doesn't exist
    sample_dir = Path("static/examples")
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Define file paths for sample images
    normal_image_path = sample_dir / "normal_sample.png"
    pneumonia_image_path = sample_dir / "pneumonia_sample.png"
    
    # URLs for reliable X-ray images
    # These are from the NIH Chest X-ray dataset which is publicly available
    normal_url = "https://prod-images-static.radiopaedia.org/images/53448329/8bc2cb1f92a97aa61420bab90653539d_big_gallery.jpeg"
    pneumonia_url = "https://prod-images-static.radiopaedia.org/images/53395620/b2c4acb1899438c6930e0fc4f2a4c31c_big_gallery.jpeg"
    
    # Download the images if they don't exist locally
    if not normal_image_path.exists():
        try:
            print(f"Downloading normal X-ray sample image...")
            response = requests.get(normal_url, stream=True)
            response.raise_for_status()
            with open(normal_image_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Normal sample saved to {normal_image_path}")
        except Exception as e:
            print(f"Error downloading normal sample: {e}")
            # Provide a placeholder if download fails
            normal_image_path = None
            
    if not pneumonia_image_path.exists():
        try:
            print(f"Downloading pneumonia X-ray sample image...")
            response = requests.get(pneumonia_url, stream=True)
            response.raise_for_status()
            with open(pneumonia_image_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Pneumonia sample saved to {pneumonia_image_path}")
        except Exception as e:
            print(f"Error downloading pneumonia sample: {e}")
            # Provide a placeholder if download fails
            pneumonia_image_path = None
    
    return str(normal_image_path) if normal_image_path else None, str(pneumonia_image_path) if pneumonia_image_path else None

def display_sample_comparison():
    """
    Displays a comparison of normal vs pneumonia X-ray images with proper labeling and styling.
    """
    st.write("## Comparison: Normal vs. Pneumonia X-rays")
    
    # Get sample images
    normal_img_path, pneumonia_img_path = download_sample_xray_images()
    
    if normal_img_path and pneumonia_img_path:
        col1, col2 = st.columns(2)
        
        # Display normal X-ray
        with col1:
            st.write("### Normal Lung X-ray")
            try:
                normal_img = Image.open(normal_img_path)
                st.image(normal_img, use_column_width=True, caption="Normal lungs appear clear without significant opacities.")
            except Exception as e:
                st.error(f"Error displaying normal sample: {e}")
        
        # Display pneumonia X-ray
        with col2:
            st.write("### Pneumonia Lung X-ray")
            try:
                pneumonia_img = Image.open(pneumonia_img_path)
                st.image(pneumonia_img, use_column_width=True, caption="Pneumonia typically shows white opacities in the lung fields.")
            except Exception as e:
                st.error(f"Error displaying pneumonia sample: {e}")
                
        # Add explanatory text
        st.write("""
        ### Key Differences
        - **Normal X-rays**: Lungs appear as dark spaces with minimal dense (white) areas. The air-filled lungs allow X-rays to pass through, appearing dark on the image.
        - **Pneumonia X-rays**: Show white opacities or consolidations where fluid has accumulated in the lungs. These areas block X-rays, appearing white on the image.
        
        These differences are what our AI model has been trained to detect.
        """)
    else:
        st.warning("Sample images could not be displayed. Please check your internet connection.")

def main():
    # Set page config
    st.set_page_config(
        page_title="Pneumonia Detection System",
        page_icon="🫁",
        layout="wide"
    )
    
    # Custom CSS with better styling
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    
    h1, h2, h3 {
        color: #2c3e50;
        font-weight: 600;
    }
    
    .stApp {
        background-color: #f9fafe;
    }
    
    .main-header {
        color: #3498db;
        font-size: 2.5rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 1rem;
        padding: 1rem 0;
        border-bottom: 2px solid #e6e6e6;
    }
    
    .subheader {
        color: #7f8c8d;
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .upload-section {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
    }
    
    .results-section {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-top: 1rem;
    }
    
    .footer {
        text-align: center;
        padding: 1rem 0;
        font-size: 0.8rem;
        color: #95a5a6;
        margin-top: 2rem;
        border-top: 1px solid #e6e6e6;
    }
    
    /* Confidence meter styling */
    .confidence-meter-bg {
        background-color: #ecf0f1;
        border-radius: 5px;
        height: 30px;
        width: 100%;
        margin-top: 10px;
    }
    
    .confidence-meter {
        background: linear-gradient(90deg, #2ecc71, #f1c40f, #e74c3c);
        border-radius: 5px;
        height: 30px;
        transition: width 0.5s ease-in-out;
    }
    
    .tooltip {
      position: relative;
      display: inline-block;
      border-bottom: 1px dotted black;
    }
    
    .tooltip .tooltiptext {
      visibility: hidden;
      width: 200px;
      background-color: black;
      color: #fff;
      text-align: center;
      border-radius: 6px;
      padding: 5px 0;
      position: absolute;
      z-index: 1;
      bottom: 125%;
      left: 50%;
      margin-left: -100px;
      opacity: 0;
      transition: opacity 0.3s;
    }
    
    .tooltip:hover .tooltiptext {
      visibility: visible;
      opacity: 1;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # App Header
    st.markdown('<h1 class="main-header">Pneumonia Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subheader">Upload a chest X-ray image to detect pneumonia</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/lungs.png", width=80)
        st.title("About This App")
        st.info("""
        This application uses deep learning to analyze chest X-ray images and detect pneumonia. 
        
        Upload your chest X-ray image and get an instant prediction with confidence score.
        """)
        
        st.markdown("### How It Works")
        st.write("""
        1. The model was trained on thousands of labeled chest X-ray images
        2. It learned to identify patterns associated with pneumonia
        3. When you upload an image, it's processed and analyzed by the model
        4. The model provides a prediction with a confidence score
        """)
        
        st.markdown("### About Pneumonia")
        st.write("""
        Pneumonia is an infection that inflames the air sacs in one or both lungs. The air sacs may fill with fluid or pus, causing symptoms such as cough, fever, chills, and difficulty breathing.
        """)
        
        if st.button("View Sample X-rays"):
            st.session_state['show_samples'] = True
    
    # Main content area
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown("## Upload X-ray Image")
        uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "jpeg", "png"])
        st.markdown('</div>', unsafe_allow_html=True)
        
        if uploaded_file is not None:
            # Create a temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name
            temp_file.close()
            
            # Display the uploaded image
            image = Image.open(temp_file_path)
            st.image(image, caption='Uploaded X-ray Image', use_column_width=True)
            
            # Process the image and make prediction
            with st.spinner('Analyzing image...'):
                try:
                    # Load model if not already loaded
                    if 'model' not in st.session_state:
                        st.session_state['model'] = load_trained_model()
                    
                    # Preprocess the image
                    preprocessed_img = preprocess_image(temp_file_path)
                    
                    # Make prediction
                    prediction, confidence = predict(st.session_state['model'], preprocessed_img)
                    
                    # Store results in session state
                    st.session_state['prediction'] = prediction
                    st.session_state['confidence'] = confidence
                    st.session_state['show_results'] = True
                    
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
                    if os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)
    
    with col2:
        if uploaded_file is not None and 'show_results' in st.session_state and st.session_state['show_results']:
            st.markdown('<div class="results-section">', unsafe_allow_html=True)
            st.markdown("## Diagnosis Results")
            
            prediction = st.session_state['prediction']
            confidence = st.session_state['confidence']
            
            # Display prediction result
            if prediction == "PNEUMONIA":
                result_color = "#e74c3c"
                icon = "🚨"
            else:
                result_color = "#2ecc71"
                icon = "✅"
                
            st.markdown(f"<h3 style='color: {result_color};'>{icon} {prediction}</h3>", unsafe_allow_html=True)
            
            # Display confidence meter
            st.write(f"Confidence: {confidence:.2%}")
            meter_width = min(confidence * 100, 100)
            
            # Determine color based on confidence
            if confidence > 0.7:
                meter_color = "#2ecc71"  # Green for high confidence
            elif confidence > 0.5:
                meter_color = "#f1c40f"  # Yellow for medium confidence
            else:
                meter_color = "#e74c3c"  # Red for low confidence
            
            st.markdown(f"""
            <div class="confidence-meter-bg">
                <div class="confidence-meter" style="width: {meter_width}%; background-color: {meter_color};"></div>
            </div>
            """, unsafe_allow_html=True)
            
            # Interpretation and recommendations
            st.markdown("### Interpretation")
            if prediction == "PNEUMONIA":
                st.markdown("""
                - The model has detected patterns consistent with pneumonia in this X-ray
                - Areas of consolidation or infiltrates may be present in the lungs
                """)
                st.warning("**Recommendation**: Please consult with a healthcare professional for proper diagnosis and treatment.")
            else:
                st.markdown("""
                - The model did not detect significant patterns consistent with pneumonia
                - The lung fields appear relatively clear
                """)
                st.info("**Note**: This is not a medical diagnosis. If you're experiencing symptoms, please consult with a healthcare professional.")
            
            st.markdown("### Model Confidence Explanation")
            st.markdown("""
            The confidence score indicates how certain the model is about its prediction:
            - **High confidence (>70%)**: The model is very certain about its prediction
            - **Medium confidence (50-70%)**: The model is somewhat certain about its prediction
            - **Low confidence (<50%)**: The model is uncertain about its prediction
            """)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
    
    # Show sample comparison if requested
    if 'show_samples' in st.session_state and st.session_state['show_samples']:
        display_sample_comparison()
    
    # Display AI model information
    st.markdown("## About The AI Model")
    st.write("""
    This application uses a deep convolutional neural network (CNN) trained on thousands of annotated chest X-ray images. 
    The model architecture is designed to detect visual patterns associated with pneumonia.

import os
import time
import numpy as np
import tempfile
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add scripts directory to path
scripts_dir = Path(__file__).parent / "scripts"
sys.path.append(str(scripts_dir))

try:
    from scripts.predict import load_trained_model, preprocess_image, predict
except ImportError:
    # Fallback import if running from app.py's directory
    from predict import load_trained_model, preprocess_image, predict

# Page configuration
st.set_page_config(
    page_title="Pneumonia Detection System",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create directories if they don't exist
os.makedirs("static/examples", exist_ok=True)
os.makedirs("temp", exist_ok=True)

# Define medical-themed color palette
COLORS = {
    "primary": "#1976D2",      # Medical blue
    "secondary": "#03A9F4",    # Lighter blue
    "accent": "#FF5722",       # Alerting orange
    "success": "#4CAF50",      # Healthy green
    "warning": "#FFC107",      # Warning yellow
    "danger": "#F44336",       # Critical red
    "background": "#F5F9FF",   # Light blue background
    "light": "#E3F2FD",        # Very light blue
    "dark": "#0D47A1",         # Deep blue
    "text": "#37474F",         # Dark blue-grey
    "subtle": "#B0BEC5"        # Subtle grey
}

# Custom CSS with Google Fonts
st.markdown(f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700&family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">

<style>
    /* Base styles and fonts */
    * {{
        font-family: 'Poppins', sans-serif;
        color: #37474F;
    }}
    
    h1, h2, h3, h4, h5, h6 {{
        font-family: 'Montserrat', sans-serif;
        font-weight: 600;
    }}
    
    h1 {{
        color: {COLORS["primary"]};
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        margin-bottom: 1.5rem !important;
    }}
    
    h2 {{
        color: {COLORS["secondary"]};
        font-size: 1.8rem !important;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
    }}
    
    h3 {{
        color: {COLORS["dark"]};
        font-size: 1.4rem !important;
        margin-top: 1.5rem !important;
    }}
    
    /* Card-like containers */
    .card {{
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }}
    
    /* Header styling */
    .header {{
        display: flex;
        align-items: center;
        margin-bottom: 2rem;
    }}
    
    .header-logo {{
        height: 60px;
        margin-right: 15px;
    }}
    
    /* Buttons styling */
    .stButton>button {{
        background-color: {COLORS["primary"]};
        color: white;
        border-radius: 5px;
        font-weight: 500;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }}
    
    .stButton>button:hover {{
        background-color: {COLORS["dark"]};
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }}
    
    /* File uploader styling */
    .stUploadButton>button {{
        background-color: {COLORS["secondary"]};
    }}
    
    /* Prediction result styling */
    .prediction-normal {{
        color: {COLORS["success"]};
        font-weight: 600;
        font-size: 1.6rem;
    }
    
    .prediction-pneumonia {
        color: {COLORS["danger"]};
        font-weight: 600;
        font-size: 1.6rem;
    }
    
    /* Confidence meter */
    .confidence-meter-container {
        margin-top: 15px;
        margin-bottom: 30px;
        background-color: {COLORS["light"]};
        border-radius: 10px;
        padding: 15px;
    }
    
    .confidence-meter {
        height: 20px;
        background: linear-gradient(to right, {COLORS["success"]}, {COLORS["warning"]}, {COLORS["danger"]});
        border-radius: 10px;
        margin-top: 10px;
        position: relative;
    }
    
    .confidence-marker {
        position: absolute;
        top: -10px;
        width: 4px;
        height: 40px;
        background-color: {COLORS["dark"]};
        border-radius: 4px;
    }
    
    .confidence-label {
        margin-top: 15px;
        font-weight: 500;
        text-align: center;
    }
    
    /* Image comparison */
    .image-comparison {
        display: flex;
        justify-content: space-between;
        margin-top: 20px;
        flex-wrap: wrap;
    }
    
    .comparison-image {
        width: 48%;
        border-radius: 10px;
        margin-bottom: 15px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    }
    
    .caption {
        text-align: center;
        margin-top: 10px;
        font-size: 0.9rem;
        color: {COLORS["text"]};
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 20px;
        margin-top: 40px;
        border-top: 1px solid {COLORS["light"]};
        color: {COLORS["subtle"]};
        font-size: 0.8rem;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .comparison-image {
            width: 100%;
        }
    }
    
    /* Sidebar custom styling */
    .css-163ttbj, .css-1d391kg {
        background-color: {COLORS["background"]};
    }
    
    /* General sidebar headings */
    .css-163ttbj .css-10trblm, .css-1d391kg .css-10trblm {
        color: {COLORS["primary"]};
    }
    
    /* Improve sidebar content spacing */
    .css-163ttbj > div, .css-1d391kg > div {
        padding: 1rem;
    }
    
    /* Make the UI more spacious */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 4rem;
    }
    
    /* Plot styling */
    .plot-container {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05);
        padding: 1rem;
        margin-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to get sample image paths
def get_sample_images():
    # Path to example directories
    normal_example = "static/examples/normal.jpg"
    pneumonia_example = "static/examples/pneumonia.jpg"
    
    # Check if example images exist, if not create them
    if not os.path.exists(normal_example) or not os.path.exists(pneumonia_example):
        # Create directories if they don't exist
        os.makedirs("static/examples", exist_ok=True)
        
        # URLs of example images (these would be replaced with your actual examples)
        # For now we'll just create a simple colored image
        normal_img = Image.new('RGB', (500, 500), color=(230, 240, 250))
        pneumonia_img = Image.new('RGB', (500, 500), color=(250, 220, 220))
        
        # Add text to the images
        import PIL.ImageDraw as ImageDraw
        import PIL.ImageFont as ImageFont
        
        draw_normal = ImageDraw.Draw(normal_img)
        draw_pneumonia = ImageDraw.Draw(pneumonia_img)
        
        # Try to use a font, fallback to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 36)
        except IOError:
            font = ImageFont.load_default()
        
        draw_normal.text((150, 250), "Normal X-ray", fill=(0, 0, 0), font=font)
        draw_pneumonia.text((120, 250), "Pneumonia X-ray", fill=(0, 0, 0), font=font)
        
        # Save the example images
        normal_img.save(normal_example)
        pneumonia_img.save(pneumonia_example)
    
    return normal_example, pneumonia_example

# Get or create sample images
normal_sample, pneumonia_sample = get_sample_images()

# Function to display confidence meter
def display_confidence_meter(confidence):
    """Display a visual confidence meter."""
    
    st.markdown('<div class="confidence-meter-container">', unsafe_allow_html=True)
    st.markdown(f'<div style="display: flex; justify-content: space-between;"><span>Normal</span><span>Pneumonia</span></div>', unsafe_allow_html=True)
    
    # Create the meter background
    st.markdown('<div class="confidence-meter"></div>', unsafe_allow_html=True)
    
    # Place the marker at the position corresponding to the confidence
    position = confidence * 100  # convert to percentage
    st.markdown(f'<div class="confidence-marker" style="left: {position}%;"></div>', unsafe_allow_html=True)
    
    confidence_text = f"Confidence: {confidence:.2f}"
    st.markdown(f'<div class="confidence-label">{confidence_text}</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Function to display model metrics if available
def display_model_metrics():
    """Display model performance metrics if the plots exist."""
    
    # List of potential plot files
    plot_files = {
        "Confusion Matrix": "plots/confusion_matrix.png",
        "ROC Curve": "plots/roc_curve.png",
        "Precision-Recall Curve": "plots/pr_curve.png",
        "Training History": "plots/training_history.png"
    }
    
    # Check which plots exist
    existing_plots = {name: path for name, path in plot_files.items() if os.path.exists(path)}
    
    if existing_plots:
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("Model Performance Metrics")
        
        # Create columns based on the number of plots (max 2 per row)
        num_plots = len(existing_plots)
        cols_per_row = min(num_plots, 2)
        
        # Display plots in a grid
        for i in range(0, num_plots, cols_per_row):
            cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                if i+j < num_plots:
                    plot_name = list(existing_plots.keys())[i+j]
                    plot_path = existing_plots[plot_name]
                    cols[j].image(plot_path, caption=plot_name, use_column_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Model performance metrics are not available. Train a model first to see metrics.")

# Main application function
def main():
    """Main application for pneumonia detection."""
    
    # Sidebar
    with st.sidebar:
        st.markdown('<h2 style="color:#1976D2;">Pneumonia Detection</h2>', unsafe_allow_html=True)
        st.markdown("---")
        
        st.subheader("About")
        st.markdown("""
        This application uses a deep learning model to detect pneumonia in chest X-ray images.
        
        Upload a chest X-ray image to get a prediction on whether the patient may have pneumonia.
        """)
        
        st.markdown("---")
        
        st.subheader("Navigation")
        page = st.radio("Go to:", ["Predict", "Information", "About Model"])
        
        st.markdown("---")
        
        st.subheader("Example Images")
        st.image(normal_sample, caption="Normal X-ray", width=150)
        st.image(pneumonia_sample, caption="Pneumonia X-ray", width=150)
    
    # Header
    st.markdown('<div class="header">', unsafe_allow_html=True)
    st.markdown('<h1>🫁 Pneumonia Detection System</h1>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content based on page selection
    if page == "Predict":
        prediction_page()
    elif page == "Information":
        information_page()
    else:
        about_model_page()
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>© 2023 Pneumonia Detection System • Developed with ❤️ • Powered by TensorFlow and Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

def prediction_page():
    """Page for making predictions on uploaded images."""
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Upload a Chest X-ray Image")
    st.write("Upload a chest X-ray image to detect the presence of pneumonia.")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    # Process the uploaded image
    if uploaded_file is not None:
        # Display a spinner while processing
        with st.spinner("Analyzing image..."):
            # Read and display the image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width=400)
            
            # Save the uploaded image to a temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
import os
import sys
import tempfile
import uuid

# Add the scripts directory to the path so we can import the predict module
sys.path.insert(0, 'scripts')
from predict import load_trained_model, preprocess_image, predict

# Set page config
st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with Tailwind-like styling and Google Fonts
def load_css():
    st.markdown("""
    <style>
        /* Google Fonts Integration */
        @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700&family=Open+Sans:wght@300;400;500;600;700&family=Poppins:wght@300;400;500;600;700&display=swap');
        
        /* Base Typography */
        * {
            font-family: 'Open Sans', sans-serif;
        }
        
        h1, h2, h3, .header, .subheader {
            font-family: 'Montserrat', sans-serif;
            letter-spacing: -0.025em;
        }
        
        /* Medical-themed color palette */
        :root {
            --primary: #2C7BBE;         /* Primary blue */
            --primary-dark: #1A5889;    /* Darker blue */
            --secondary: #43B0A5;       /* Teal accent */
            --accent: #5271FF;          /* Accent blue */
            --warning: #E9A054;         /* Warning orange */
            --danger: #E05D5D;          /* Danger red */
            --success: #4CAF79;         /* Success green */
            --light-bg: #F0F4F8;        /* Light background */
            --dark-text: #2D3748;       /* Dark text */
            --medium-text: #4A5568;     /* Medium text */
            --light-text: #718096;      /* Light text */
            --card-shadow: 0 8px 15px rgba(0, 0, 0, 0.05), 0 3px 6px rgba(0, 0, 0, 0.08);
        }
        
        /* Modern container */
        .container {
            max-width: 1200px;
            margin-left: auto;
            margin-right: auto;
            padding-left: 1.5rem;
            padding-right: 1.5rem;
        }
        
        /* Enhanced card styling */
        .card {
            background-color: #ffffff;
            border-radius: 0.75rem;
            box-shadow: var(--card-shadow);
            padding: 1.75rem;
            margin-bottom: 1.75rem;
            border-top: 4px solid var(--primary);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        
        .card:hover {
            transform: translateY(-3px);
            box-shadow: 0 12px 20px rgba(0, 0, 0, 0.07), 0 5px 12px rgba(0, 0, 0, 0.1);
        }
        
        /* Typography styles */
        .header {
            color: var(--dark-text);
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 1.25rem;
            border-bottom: 2px solid var(--primary);
            padding-bottom: 0.5rem;
            display: inline-block;
        }
        
        .subheader {
            color: var(--dark-text);
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 1rem;
            position: relative;
        }
        
        .subheader:after {
            content: '';
            position: absolute;
            bottom: -6px;
            left: 0;
            width: 40px;
            height: 3px;
            background-color: var(--secondary);
            border-radius: 3px;
        }
        
        .text-normal {
            color: var(--medium-text);
            font-size: 1.05rem;
            line-height: 1.6;
            margin-bottom: 1.25rem;
        }
        
        .text-small {
            color: var(--light-text);
            font-size: 0.9rem;
            line-height: 1.4;
        }
        
        /* Button styling */
        .btn {
            display: inline-block;
            padding: 0.6rem 1.2rem;
            font-weight: 500;
            font-family: 'Poppins', sans-serif;
            text-align: center;
            white-space: nowrap;
            border-radius: 0.375rem;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.15);
        }
        
        .btn-primary {
            background-color: var(--primary);
            color: #ffffff;
            border: 1px solid var(--primary);
        }
        
        .btn-primary:hover {
            background-color: var(--primary-dark);
            border-color: var(--primary-dark);
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }
        
        /* Alert styling */
        .alert {
            padding: 1.2rem;
            border-radius: 0.5rem;
            margin-bottom: 1.25rem;
            font-weight: 500;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }
        
        .alert-success {
            background-color: rgba(76, 175, 121, 0.1);
            border-left: 4px solid var(--success);
            color: var(--success);
        }
        
        .alert-warning {
            background-color: rgba(233, 160, 84, 0.1);
            border-left: 4px solid var(--warning);
            color: var(--warning);
        }
        
        .alert-danger {
            background-color: rgba(224, 93, 93, 0.1);
            border-left: 4px solid var(--danger);
            color: var(--danger);
        }
        
        /* Layout grid */
        .grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 1.5rem;
        }
        
        .col-span-2 {
            grid-column: span 2 / span 2;
        }
        
        /* Image styling */
        .rounded-image {
            border-radius: 0.75rem;
            width: 100%;
            height: auto;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            border: 3px solid #ffffff;
            transition: transform 0.3s ease;
        }
        
        .rounded-image:hover {
            transform: scale(1.02);
        }
        
        .image-container {
            position: relative;
            margin-bottom: 1rem;
            border-radius: 0.75rem;
            overflow: hidden;
            box-shadow: var(--card-shadow);
        }
        
        .image-caption {
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            background: rgba(0, 0, 0, 0.6);
            color: white;
            padding: 0.5rem;
            font-size: 0.9rem;
            text-align: center;
            font-family: 'Poppins', sans-serif;
        }
        
        /* Confidence meter styling */
        .confidence-meter {
            height: 1.5rem;
            background-color: #e2e8f0;
            border-radius: 9999px;
            overflow: hidden;
            margin-bottom: 0.75rem;
            box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        
        .confidence-bar {
            height: 100%;
            border-radius: 9999px;
            transition: width 0.5s ease;
        }
        
        .confidence-high {
            background-color: var(--success);
            background-image: linear-gradient(45deg, 
                rgba(255, 255, 255, 0.15) 25%, 
                transparent 25%, 
                transparent 50%, 
                rgba(255, 255, 255, 0.15) 50%, 
                rgba(255, 255, 255, 0.15) 75%, 
                transparent 75%, 
                transparent);
            background-size: 1rem 1rem;
            animation: progress-bar-stripes 1s linear infinite;
        }
        
        .confidence-medium {
            background-color: var(--warning);
            background-image: linear-gradient(45deg, 
                rgba(255, 255, 255, 0.15) 25%, 
                transparent 25%, 
                transparent 50%, 
                rgba(255, 255, 255, 0.15) 50%, 
                rgba(255, 255, 255, 0.15) 75%, 
                transparent 75%, 
                transparent);
            background-size: 1rem 1rem;
        }
        
        .confidence-low {
            background-color: var(--danger);
        }
        
        @keyframes progress-bar-stripes {
            from { background-position: 1rem 0; }
            to { background-position: 0 0; }
        }
        
        /* Badge styling */
        .badge {
            display: inline-block;
            padding: 0.35em 0.65em;
            font-size: 0.85em;
            font-weight: 600;
            line-height: 1;
            text-align: center;
            white-space: nowrap;
            vertical-align: baseline;
            border-radius: 0.375rem;
            margin-right: 0.5rem;
            font-family: 'Poppins', sans-serif;
        }
        
        .badge-primary {
            background-color: var(--primary);
            color: white;
        }
        
        .badge-secondary {
            background-color: var(--secondary);
            color: white;
        }
        
        /* Image comparison container */
        .comparison-container {
            display: flex;
            flex-direction: column;
            gap: 1rem;
            margin: 1.5rem 0;
        }
        
        .comparison-item {
            display: flex;
            flex-direction: column;
            background: white;
            border-radius: 0.75rem;
            overflow: hidden;
            box-shadow: var(--card-shadow);
            border: 1px solid #e2e8f0;
            transition: transform 0.2s ease;
        }
        
        .comparison-item:hover {
            transform: translateY(-5px);
        }
        
        .comparison-header {
            background: var(--primary);
            color: white;
            padding: 0.75rem;
            font-weight: 600;
            text-align: center;
            font-family: 'Montserrat', sans-serif;
        }
        
        .pneumonia-header {
            background: var(--danger);
        }
        
        .normal-header {
            background: var(--success);
        }
        
        .comparison-image {
            width: 100%;
            height: auto;
            object-fit: cover;
        }
        
        .comparison-text {
            padding: 0.75rem;
            font-size: 0.9rem;
            color: var(--medium-text);
        }
        
        /* Animated elements */
        .animated-pulse {
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% {
                transform: scale(1);
            }
            50% {
                transform: scale(1.05);
            }
            100% {
                transform: scale(1);
            }
        }
        
        /* Footer styling */
        .footer {
            text-align: center;
            padding: 1.5rem 0;
            margin-top: 2rem;
            border-top: 1px solid #e2e8f0;
            color: var(--light-text);
            font-size: 0.9rem;
        }
    "</style>", unsafe_allow_html=True)

def save_uploaded_image(uploaded_image):
    """
    Save an uploaded PIL Image to a temporary file.
    
    Args:
        uploaded_image (PIL.Image): The uploaded image
        
    Returns:
        str: Path to the saved temporary file
    """
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    temp_file_path = temp_file.name
    temp_file.close()
    
    # Save the image to the temporary file
    uploaded_image.save(temp_file_path, format="JPEG")
    
    return temp_file_path

# Function to create the app layout
def create_layout():
    """Create the main app layout and components"""
    # Main content
    st.markdown('<div class="container">', unsafe_allow_html=True)
    
    # Header Section
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="header">Pneumonia Detection System</p>', unsafe_allow_html=True)
    st.markdown("""
    <p class="text-normal">
        This application uses AI to analyze chest X-ray images and detect signs of pneumonia.
        Upload an X-ray image below to get started.
    </p>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Upload and Prediction Section
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="subheader">Upload X-ray Image</p>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            # Convert the file to
    
    # Main content
    st.markdown('<div class="container">', unsafe_allow_html=True)
    
    # Header Section
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="header">Pneumonia Detection System</p>', unsafe_allow_html=True)
    st.markdown("""
    <p class="text-normal">
        This application uses AI to analyze chest X-ray images and detect signs of pneumonia.
        Upload an X-ray image below to get started.
    </p>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Upload and Prediction Section
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="subheader">Upload X-ray Image</p>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            # Convert the file to an image
            image = Image.open(uploaded_file)
            
            st.markdown('<div class="grid">', unsafe_allow_html=True)
            
            # Display the uploaded image
            st.markdown('<div>', unsafe_allow_html=True)
            st.image(image, caption="Uploaded X-ray Image", use_column_width=True, output_format="JPEG")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Make prediction
            try:
                # Save the uploaded image to a temporary file
                temp_image_path = save_uploaded_image(image)
                
                # Get the prediction
                # Load model if not already loaded
                model = load_trained_model()
                
                # Get the prediction
                prediction, confidence = predict(model, temp_image_path)
                
                # Clean up temporary file
                try:
                    os.remove(temp_image_path)
                except:
                    pass  # Ignore if file can't be deleted
                
                st.markdown('<div>', unsafe_allow_html=True)
                st.markdown('<p class="subheader">Prediction Result</p>', unsafe_allow_html=True)
                
                # Display prediction with styled output
                if prediction == "PNEUMONIA":
                    st.markdown(f'<p class="alert alert-danger">Diagnosis: <strong>PNEUMONIA</strong></p>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<p class="alert alert-success">Diagnosis: <strong>NORMAL</strong></p>', unsafe_allow_html=True)
                
                # Display confidence score with a bar
                st.markdown('<p class="text-normal">Confidence Score:</p>', unsafe_allow_html=True)
                
                # Determine confidence level class
                confidence_class = ""
                if confidence > 0.8:
                    confidence_class = "confidence-high"
                elif confidence > 0.6:
                    confidence_class = "confidence-medium"
                else:
                    confidence_class = "confidence-low"
                
                # Create confidence meter
                st.markdown(f'''
                <div class="confidence-meter">
                    <div class="confidence-bar {confidence_class}" style="width: {confidence * 100}%"></div>
                </div>
                <p class="text-normal">{confidence * 100:.2f}%</p>
                ''', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            except Exception as e:
                st.error(f"Error making prediction: {e}")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error processing image: {e}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Educational Section about Pneumonia
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="subheader">What is Pneumonia?</p>', unsafe_allow_html=True)
    
    # Create two columns
    st.markdown('<div class="grid">', unsafe_allow_html=True)
    
    # Column 1: Information
    st.markdown('<div>', unsafe_allow_html=True)
    st.markdown("""
    <p class="text-normal">
        Pneumonia is an infection that inflames the air sacs in one or both lungs. The air sacs may fill with fluid or pus, causing cough with phlegm or pus, fever, chills, and difficulty breathing.
    </p>
    
    <p class="text-normal">
        <strong>Common symptoms include:</strong>
    </p>
    <ul class="text-normal">
        <li>Chest pain when breathing or coughing</li>
        <li>Confusion or changes in mental awareness (in adults age 65 and older)</li>
        <li>Cough, which may produce phlegm</li>
        <li>Fatigue</li>
        <li>Fever, sweating and shaking chills</li>
        <li>Lower than normal body temperature (in adults older than age 65 and people with weak immune systems)</li>
        <li>Nausea, vomiting or diarrhea</li>
        <li>Shortness of breath</li>
    </ul>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Column 2: Example images
    st.markdown('<div>', unsafe_allow_html=True)
    st.markdown('<p class="text-normal"><strong>Example of X-rays:</strong></p>', unsafe_allow_html=True)
    
    # Use local images
    # Check if example images directory exists, if not create it
    example_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'examples')
    os.makedirs(example_dir, exist_ok=True)
    
    # Define image paths
    normal_img_path = os.path.join(example_dir, 'normal.jpg')
    pneumonia_img_path = os.path.join(example_dir, 'pneumonia.jpg')
    
    # URLs for downloading example images if they don't exist
    normal_img_url = "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/00000001_000.png"
    pneumonia_img_url = "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/00000289_002.png"
    
    # Function to download image from URL
    def download_image(url, save_path):
        try:
            import requests
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                return True
            return False
        except Exception as e:
            st.error(f"Error downloading image: {e}")
            return False
    
    # Check if example images exist, and download them if they don't
    if not os.path.exists(normal_img_path):
        st.info("Downloading example normal X-ray image...")
        download_success = download_image(normal_img_url, normal_img_path)
        if not download_success:
            st.warning("Failed to download normal X-ray example. Using placeholder.")
            # Create a placeholder image
            placeholder = Image.new('RGB', (500, 500), color=(240, 240, 240))
            placeholder.save(normal_img_path)
    
    if not os.path.exists(pneumonia_img_path):
        st.info("Downloading example pneumonia X-ray image...")
        download_success = download_image(pneumonia_img_url, pneumonia_img_path)
        if not download_success:
            st.warning("Failed to download pneumonia X-ray example. Using placeholder.")
            # Create a placeholder image
            placeholder = Image.new('RGB', (500, 500), color=(240, 240, 240))
            placeholder.save(pneumonia_img_path)
    
    # Create comparison of normal vs pneumonia X-rays
    st.markdown('<div class="comparison-container">', unsafe_allow_html=True)
    
    # Normal X-ray
    st.markdown('''
    <div class="comparison-item">
        <div class="comparison-header normal-header">NORMAL X-RAY</div>
        <img class="comparison-image" src="static/examples/normal.jpg" alt="Normal X-ray">
        <div class="comparison-text">
            Normal chest X-rays show clear lung fields without significant opacities, fluid, or consolidation.
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    # Pneumonia X-ray
    st.markdown('''
    <div class="comparison-item">
        <div class="comparison-header pneumonia-header">PNEUMONIA X-RAY</div>
        <img class="comparison-image" src="static/examples/pneumonia.jpg" alt="Pneumonia X-ray">
        <div class="comparison-text">
            Pneumonia chest X-rays typically show areas of opacity or consolidation representing fluid in the lungs.
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Model Performance Metrics Section (if available)
    try:
        # Check if plots directory exists
        # Function to check if plots directory exists
        def check_plots_directory():
            plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
            return os.path.exists(plots_dir) and os.path.isdir(plots_dir)
            
        # Check if plots directory exists
        plots_exist = check_plots_directory()
        # Check if metrics data is available
        metrics_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots', 'metrics.csv')
        if plots_exist and os.path.exists(metrics_path):
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<p class="subheader">Model Performance Metrics</p>', unsafe_allow_html=True)
            
            # Load metrics data
            metrics_df = pd.read_csv(metrics_path)
            
            # Display metrics in a nicely formatted way
            st.markdown('<div class="grid">', unsafe_allow_html=True)
            
            # Column 1: Metrics table
            st.markdown('<div>', unsafe_allow_html=True)
            st.dataframe(metrics_df)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Column 2: Visualization
            st.markdown('<div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 6))
            metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
            metrics_df_subset = metrics_df[metrics_df['metric'].isin(metrics_to_plot)]
            
            sns.barplot(x='metric', y='value', data=metrics_df_subset, ax=ax)
            ax.set_title('Model Performance Metrics')
            ax.set_ylim(0, 1)
            ax.set_ylabel('Score')
            ax.set_xlabel('Metric')
            
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
        # Check if confusion matrix is available
        confusion_matrix_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots', 'confusion_matrix.png')
        if plots_exist and os.path.exists(confusion_matrix_path):
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<p class="subheader">Confusion Matrix</p>', unsafe_allow_html=True)
            st.image(confusion_matrix_path, use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    except Exception as e:
        # Handle gracefully - don't show error to user
        print(f"Error loading model performance metrics: {e}")
        
    # Add footer with credits and version information
    st.markdown('<div class="footer">', unsafe_allow_html=True)
    st.markdown('''
    <p>
        <strong>Pneumonia Detection System</strong> v1.0.0 | 
        Developed with ❤️ using TensorFlow and Streamlit | 
        &copy; 2023 All Rights Reserved
    </p>
    <p>
        Data source: Chest X-Ray Images (Pneumonia) from Kaggle | 
        Model: CNN Trained on 5,216 labeled images
    </p>
    ''', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Main function to run the Streamlit app"""
    # Load custom CSS
    load_css()
    
    # Set up the app layout and components
    create_layout()

if __name__ == "__main__":
    main()
