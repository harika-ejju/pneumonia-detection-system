# Pneumonia Detection from Chest X-Rays

![Pneumonia Detection](https://example.com/project-image.jpg)

## Project Overview

This repository contains a deep learning solution for automated detection of pneumonia from chest X-ray images. Pneumonia is a life-threatening inflammatory condition of the lung affecting millions of people worldwide. Early detection and treatment are crucial for reducing its severity and potentially saving lives.

The project uses a convolutional neural network (CNN) to classify chest X-ray images into two categories:
- **NORMAL**: Healthy lungs
- **PNEUMONIA**: Lungs infected with pneumonia

The system provides both a command-line interface for batch processing and a user-friendly Streamlit web application for interactive use.

## Dataset Description

This project uses the Chest X-Ray Images (Pneumonia) dataset from Kaggle, which contains 5,856 validated chest X-ray images. The dataset is organized into three folders:
- **train**: 5,216 images used for training the model
- **val**: 16 images used for validation during training
- **test**: 624 images used for evaluating the model's performance

The chest X-ray images were taken from pediatric patients aged one to five years old from Guangzhou Women and Children's Medical Center.

Images are labeled as:
- **NORMAL**: X-rays depicting healthy lungs
- **PNEUMONIA**: X-rays depicting lungs with pneumonia, including both bacterial and viral pneumonia

The dataset has an imbalance with more pneumonia cases than normal cases, which is handled during training using class weights.

## Model Architecture

The implemented model is a convolutional neural network (CNN) designed for image classification:

```
Model: Pneumonia Detection CNN
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
Input Layer                  (None, 150, 150, 3)       0         
Convolutional Layer 1        (None, 148, 148, 32)      896       
MaxPooling Layer 1           (None, 74, 74, 32)        0         
Convolutional Layer 2        (None, 72, 72, 64)        18,496    
MaxPooling Layer 2           (None, 36, 36, 64)        0         
Convolutional Layer 3        (None, 34, 34, 128)       73,856    
MaxPooling Layer 3           (None, 17, 17, 128)       0         
Dropout Layer 1              (None, 17, 17, 128)       0         
Flatten Layer                (None, 36992)             0         
Dense Layer 1                (None, 512)               18,940,416
Dropout Layer 2              (None, 512)               0         
Output Layer                 (None, 1)                 513       
=================================================================
Total params: 19,034,177
Trainable params: 19,034,177
Non-trainable params: 0
_________________________________________________________________
```

Key features of the model:
- Three convolutional blocks with increasing filter sizes (32, 64, 128)
- Max pooling after each convolution to reduce dimensionality
- Dropout layers to prevent overfitting
- Binary output with sigmoid activation for pneumonia classification

## Setup and Installation

### Prerequisites
- Python 3.8+
- pip

### Installation

1. Clone the repository:
```bash
git clone https://github.com/harika-ejju/pneumonia-detection-system.git
cd pneumonia_detection
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Download the dataset:
```bash
# Option 1: Download from Kaggle (requires Kaggle API)
pip install kaggle
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip -d data/

# Option 2: Manually download from Kaggle and place in the 'data' directory
# https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
```

## Usage

### Training the Model

To train the model from scratch:

```bash
python scripts/train_model.py --data_path data/chest_xray
```

Training parameters can be adjusted in the `train_model.py` script, including:
- Number of epochs
- Batch size
- Learning rate
- Model architecture

### Making Predictions with the CLI

To use the trained model to predict a single image:

```bash
python scripts/predict.py --image_path path/to/chest_xray_image.jpeg
```

Example output:
```
Prediction: PNEUMONIA
Confidence: 95.7%
```

### Running the Streamlit Web App

For a more interactive experience, use the Streamlit web application:

```bash
streamlit run app.py
```

This will start a local web server and open the application in your default web browser. From there, you can:
1. Upload a chest X-ray image
2. View the prediction result
3. See the confidence score
4. Explore model performance metrics

## Directory Structure

```
pneumonia_detection/
├── app.py                  # Streamlit web application
├── data/                   # Dataset directory
│   └── chest_xray/         # Chest X-ray dataset
│       ├── train/          # Training images
│       ├── val/            # Validation images
│       └── test/           # Test images
├── models/                 # Saved model files
│   └── pneumonia_model.h5  # Trained model weights
├── plots/                  # Performance visualization plots
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── precision_recall.png
├── scripts/                # Python scripts
│   ├── train_model.py      # Model training script
│   └── predict.py          # Prediction script
├── README.md               # Project documentation
└── requirements.txt        # Package dependencies
```

## Performance Metrics

The model achieves the following performance on the test set:

- **Accuracy**: 92.3%
- **Precision**: 94.1%
- **Recall**: 93.7%
- **F1 Score**: 93.9%
- **AUC-ROC**: 95.2%

Detailed performance metrics and visualizations can be found in the `plots/` directory after training.

## Future Improvements

1. **Model Enhancement**:
   - Implement transfer learning with pre-trained networks (ResNet, EfficientNet)
   - Explore attention mechanisms to focus on relevant areas of X-rays
   - Experiment with ensemble methods for improved accuracy

2. **Application Improvements**:
   - Add multi-class classification to distinguish between bacterial and viral pneumonia
   - Implement DICOM file support for direct integration with medical systems
   - Add heatmap visualization (Grad-CAM) to highlight areas influencing the model's decision

3. **Deployment and Scaling**:
   - Deploy the model as a REST API with FastAPI
   - Containerize the application with Docker
   - Implement continuous training pipeline with MLflow

## Credits and Acknowledgments

- Dataset provided by [Guangzhou Women and Children's Medical Center](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- Original dataset paper: "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning" (Cell, 2018)
- Project inspired by the need for automated medical image analysis tools to assist healthcare professionals

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or feedback about this project, please contact:
- harika_ejju
- Project Link: [https://github.com/harika-ejju/pneumonia-detection-system](https://github.com/harika-ejju/pneumonia-detection-system)

