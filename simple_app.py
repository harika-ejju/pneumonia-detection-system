import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import io

# Set page configuration
st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🫁",
    layout="wide"
)

# Custom CSS for basic styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
    }
    .result-normal {
        color: #4CAF50;
        font-weight: bold;
        font-size: 1.8rem;
    }
    .result-pneumonia {
        color: #F44336;
        font-weight: bold;
        font-size: 1.8rem;
    }
</style>
""", unsafe_allow_html=True)

# App header
st.markdown('<p class="main-header">Pneumonia Detection from Chest X-Rays</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload a chest X-ray image to detect pneumonia</p>', unsafe_allow_html=True)

# Create two columns for the layout
col1, col2 = st.columns([1, 1])

# File uploader for X-ray image
with col1:
    st.subheader("Upload X-ray Image")
    uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded X-ray Image", use_column_width=True)

# Mock prediction function (in a real app, this would use a trained model)
def predict_pneumonia(image):
    """Mock function to predict pneumonia from X-ray images"""
    # This is just a mock function that returns random results
    # In a real app, you would use your trained model here
    prediction = np.random.choice(["NORMAL", "PNEUMONIA"], p=[0.4, 0.6])
    confidence = np.random.uniform(0.7, 0.99)
    return prediction, confidence

# Display prediction results
with col2:
    st.subheader("Prediction Results")
    
    if uploaded_file is not None:
        # Get prediction
        with st.spinner('Analyzing X-ray image...'):
            prediction, confidence = predict_pneumonia(image)
        
        # Display results
        st.write("### Diagnosis:")
        if prediction == "NORMAL":
            st.markdown(f'<p class="result-normal">NORMAL</p>', unsafe_allow_html=True)
        else:
            st.markdown(f'<p class="result-pneumonia">PNEUMONIA DETECTED</p>', unsafe_allow_html=True)
        
        # Display confidence
        st.write(f"Confidence: {confidence:.2%}")
        
        # Confidence meter
        st.progress(confidence)
        
        # Additional information
        if prediction == "PNEUMONIA":
            st.warning("**Recommendation:** Please consult with a healthcare professional for further evaluation.")
        else:
            st.success("**Recommendation:** No signs of pneumonia detected. Regular check-ups are still recommended.")
    else:
        st.info("Please upload an X-ray image to get a prediction.")

# Sidebar information
with st.sidebar:
    st.header("About Pneumonia")
    st.write("""
    Pneumonia is an infection that inflames the air sacs in one or both lungs. 
    The air sacs may fill with fluid or pus, causing symptoms such as a cough, 
    fever, chills, and difficulty breathing.
    """)
    
    st.write("### Key Indicators in X-rays:")
    st.write("""
    - **White opacities**: Areas of inflammation
    - **Consolidation**: Lungs filled with liquid instead of air
    - **Blurry lung margins**: Compared to clear, sharp margins in healthy lungs
    """)
    
    st.write("### Disclaimer:")
    st.warning("""
    This is a demonstration application. The predictions are simulated and should not be used for medical diagnosis. 
    Always consult with healthcare professionals for medical advice.
    """)

# Footer
st.markdown("---")
st.markdown(
    "Made with ❤️ by Your Name | Demo Version 1.0"
)

