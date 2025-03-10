#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_model.py - Train a CNN model for pneumonia detection from chest X-ray images

This script:
1. Loads preprocessed data from data/processed directory
2. Implements a CNN architecture for pneumonia classification
3. Trains the model with appropriate callbacks
4. Evaluates model performance with various metrics
5. Saves the trained model and visualizations
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from datetime import datetime

# TensorFlow and Keras imports
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2

# For model evaluation
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configure GPU memory growth to avoid OOM errors
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def create_custom_cnn(input_shape, dropout_rate=0.5):
    """
    Create a custom CNN architecture for pneumonia detection
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        dropout_rate: Dropout rate for regularization
        
    Returns:
        Compiled Keras model
    """
    model = Sequential()
    
    # First convolutional block
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate/2))
    
    # Second convolutional block
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate/2))
    
    # Third convolutional block
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate/2))
    
    # Fourth convolutional block
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout_rate/2))
    
    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification (Pneumonia vs Normal)
    
    return model

def create_transfer_learning_model(base_model_name, input_shape, dropout_rate=0.5):
    """
    Create a transfer learning model for pneumonia detection
    
    Args:
        base_model_name: Name of the pretrained model ('vgg16', 'resnet50', or 'mobilenet')
        input_shape: Shape of input images (height, width, channels)
        dropout_rate: Dropout rate for regularization
        
    Returns:
        Compiled Keras model
    """
    # Define the base model
    if base_model_name == 'vgg16':
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_model_name == 'resnet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_model_name == 'mobilenet':
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        raise ValueError(f"Unsupported base model: {base_model_name}")
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    
    # Combine base model and custom head
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model

def load_data(data_dir, img_size, batch_size):
    """
    Load preprocessed data from the specified directory
    
    Args:
        data_dir: Directory containing preprocessed data
        img_size: Target size for the images (height, width)
        batch_size: Batch size for training
        
    Returns:
        train_generator, val_generator, test_generator, class_weights
    """
    # Directory paths
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')
    
    # Check if directories exist
    if not all(os.path.exists(d) for d in [train_dir, val_dir, test_dir]):
        raise ValueError(f"One or more data directories do not exist in {data_dir}")
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Only rescaling for validation and test data
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create data generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    
    # Calculate class weights to handle imbalanced data
    total_samples = train_generator.samples
    
    # Check if any images were found before calculating class weights
    if total_samples == 0:
        print("Warning: No images found in the training directory!")
        # Return default equal class weights when no samples are found
        class_weights = {0: 1.0, 1: 1.0}
    else:
        class_counts = np.bincount(train_generator.classes)
        class_weights = {
            0: total_samples / (2 * class_counts[0]),
            1: total_samples / (2 * class_counts[1])
        }
    
    print(f"Found {train_generator.samples} training samples")
    print(f"Found {val_generator.samples} validation samples")
    print(f"Found {test_generator.samples} test samples")
    print(f"Class weights: {class_weights}")
    
    return train_generator, val_generator, test_generator, class_weights

def create_callbacks(model_path, patience=10):
    """
    Create callbacks for model training
    
    Args:
        model_path: Path to save the best model
        patience: Number of epochs with no improvement after which training will be stopped
        
    Returns:
        List of callbacks
    """
    # Create the model directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Create log directory for TensorBoard
    log_dir = os.path.join('logs', datetime.now().strftime('%Y%m%d-%H%M%S'))
    os.makedirs(log_dir, exist_ok=True)
    
    # Define callbacks
    checkpoint = ModelCheckpoint(
        model_path,
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=patience//2,
        min_lr=1e-6,
        verbose=1
    )
    
    tensorboard = TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True
    )
    
    return [checkpoint, early_stopping, reduce_lr, tensorboard]

def evaluate_model(model, test_generator, class_names, plots_dir):
    """
    Evaluate the model on the test set and generate evaluation metrics and plots
    
    Args:
        model: Trained Keras model
        test_generator: Test data generator
        class_names: Names of the classes
        plots_dir: Directory to save plots
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Create plots directory if it doesn't exist
    os.makedirs(plots_dir, exist_ok=True)
    
    # Get predictions and true labels
    test_steps = int(np.ceil(test_generator.samples / test_generator.batch_size))
    y_pred_probs = model.predict(test_generator, steps=test_steps, verbose=1)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    y_true = test_generator.classes
    
    # Calculate evaluation metrics
    cf_matrix = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Calculate ROC and PR curves
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)
    
    precision, recall, _ = precision_recall_curve(y_true, y_pred_probs)
    pr_auc = average_precision_score(y_true, y_pred_probs)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'confusion_matrix.png'), dpi=300)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'roc_curve.png'), dpi=300)
    
    # Plot Precision-Recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'pr_curve.png'), dpi=300)
    
    # Extract metrics
    metrics = {
        'accuracy': report['accuracy'],
        'precision': report['weighted avg']['precision'],
        'recall': report['weighted avg']['recall'],
        'f1_score': report['weighted avg']['f1-score'],
        'roc_auc': roc_auc,
        'pr_auc': pr_auc
    }
    
    # Print metrics
    print("\nEvaluation Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    
    # Save metrics to file
    with open(os.path.join(plots_dir, 'evaluation_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    return metrics

def plot_training_history(history, plots_dir):
    """
    Plot training history (accuracy and loss curves)
    
    Args:
        history: History object returned by model.fit
        plots_dir: Directory to save plots
    """
    # Create plots directory if it doesn't exist
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot accuracy curves
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(plots_dir, 'accuracy_curves.png'), dpi=300)
    
    # Plot loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(plots_dir, 'loss_curves.png'), dpi=300)
    
    # Plot learning rate if available
    if 'lr' in history.history:
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['lr'])
        plt.title('Learning Rate')
        plt.ylabel('Learning Rate')
        plt.xlabel('Epoch')
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(plots_dir, 'learning_rate.png'), dpi=300)

def parse_args():
    """
    Parse command-line arguments
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Train a CNN model for pneumonia detection')
    parser.add_argument('--data_path', type=str, default='data/processed',
                        help='Path to the directory containing the dataset (default: data/processed)')
    return parser.parse_args()

def main():
    """
    Main function to train and evaluate the model
    """
    # Parse command-line arguments
    args = parse_args()
    
    # Configuration
    config = {
        'data_dir': args.data_path,  # Use the command-line argument
        'models_dir': 'models',
        'plots_dir': 'plots',
        'img_size': (224, 224),
        'batch_size': 32,
        'epochs': 5,
        'learning_rate': 1e-4,
        'dropout_rate': 0.5,
        'model_type': 'custom',  # 'custom', 'vgg16', 'resnet50', 'mobilenet'
        'patience': 10
    }
    
    # Create directories if they don't exist
    os.makedirs(config['models_dir'], exist_ok=True)
    os.makedirs(config['plots_dir'], exist_ok=True)
    
    print(f"Loading data from {config['data_dir']}...")
    try:
        train_generator, val_generator, test_generator, class_weights = load_data(
            config['data_dir'], 
            config['img_size'], 
            config['batch_size']
        )
    except ValueError as e:
        print(f"Error loading data: {e}")
        return
    
    # Get class names
    class_names = list(train_generator.class_indices.keys())
    print(f"Classes: {class_names}")
    
    # Define model path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(config['models_dir'], f"pneumonia_model_{config['model_type']}_{timestamp}.h5")
    
    # Create model
    print(f"Creating {config['model_type']} model...")
    input_shape = (*config['img_size'], 3)  # RGB images
    
    if config['model_type'] == 'custom':
        model = create_custom_cnn(input_shape, config['dropout_rate'])
    else:
        from tensorflow.keras.layers import GlobalAveragePooling2D  # Import was missing
        model = create_transfer_learning_model(config['model_type'], input_shape, config['dropout_rate'])
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=config['learning_rate']),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    model.summary()
    
    # Create callbacks
    callbacks = create_callbacks(model_path, config['patience'])
    
    # Train model
    print("\nTraining model...")
    history = model.fit(
        train_generator,
        epochs=config['epochs'],
        validation_data=val_generator,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # Plot training history
    print("\nPlotting training history...")
    plot_training_history(history, config['plots_dir'])
    
    # Load the best model
    model.load_weights(model_path)
    
    # Evaluate model
    print("\nEvaluating model on test set...")
    metrics = evaluate_model(model, test_generator, class_names, config['plots_dir'])
    
    # Save configuration and metrics
    results = {
        'config': config,
        'metrics': metrics,
        'model_path': model_path,
        'training_completed': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(os.path.join(config['plots_dir'], 'training_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nTraining complete! Model saved to {model_path}")
    print(f"Plots and metrics saved to {config['plots_dir']}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
