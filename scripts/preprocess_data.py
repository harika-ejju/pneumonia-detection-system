#!/usr/bin/env python3
"""
Chest X-ray Image Preprocessing Script for Pneumonia Detection

This script preprocesses chest x-ray images from the chest-xray-pneumonia dataset
by resizing, normalizing, and splitting the data into training, validation, and test sets.
It also applies appropriate data augmentation techniques to improve model training.

The script assumes the following data directory structure:
- data/
  - chest_xray/
    - train/
      - NORMAL/
      - PNEUMONIA/
    - val/
      - NORMAL/
      - PNEUMONIA/
    - test/
      - NORMAL/
      - PNEUMONIA/

Author: AI Assistant
Date: 2023
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import cv2
import shutil
import random
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration parameters
IMAGE_SIZE = (224, 224)  # Standard size for many CNN architectures
CHANNELS = 3  # RGB images
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2  # If no validation set is provided
TEST_SPLIT = 0.1  # If no test set is provided
RANDOM_SEED = 42
BASE_DATA_DIR = 'data/chest_xray'
OUTPUT_DIR = 'data/processed'

def create_directories():
    """Create necessary directories for processed data."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create subdirectories for train, validation, and test splits
    for split in ['train', 'val', 'test']:
        for category in ['NORMAL', 'PNEUMONIA']:
            os.makedirs(os.path.join(OUTPUT_DIR, split, category), exist_ok=True)
            
    logger.info(f"Created directories in {OUTPUT_DIR}")

def load_and_preprocess_images(data_dir, target_size=IMAGE_SIZE):
    """
    Load images from the given directory and preprocess them.
    
    Args:
        data_dir: Path to the directory containing category subdirectories
        target_size: Tuple (height, width) for resizing images
        
    Returns:
        Tuple of (images, labels)
    """
    images = []
    labels = []
    categories = ['NORMAL', 'PNEUMONIA']
    
    for idx, category in enumerate(categories):
        category_path = os.path.join(data_dir, category)
        if not os.path.exists(category_path):
            logger.warning(f"Category path {category_path} does not exist. Skipping.")
            continue
            
        logger.info(f"Processing {category} images from {category_path}")
        
        for img_path in os.listdir(category_path):
            if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            # Read and preprocess image
            img = cv2.imread(os.path.join(category_path, img_path))
            if img is None:
                logger.warning(f"Failed to load image: {img_path}")
                continue
                
            # Convert BGR to RGB (OpenCV loads as BGR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize to target size
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
            
            # Normalize pixel values to [0, 1]
            img = img.astype(np.float32) / 255.0
            
            images.append(img)
            labels.append(idx)  # 0 for NORMAL, 1 for PNEUMONIA
    
    return np.array(images), np.array(labels)

def create_data_generators(train_data_dir, val_data_dir=None, test_data_dir=None):
    """
    Create data generators for training, validation, and test sets.
    
    Args:
        train_data_dir: Directory containing training data
        val_data_dir: Directory containing validation data (optional)
        test_data_dir: Directory containing test data (optional)
        
    Returns:
        Tuple of (train_generator, validation_generator, test_generator)
    """
    # Data augmentation for training set
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=VALIDATION_SPLIT if val_data_dir is None else 0
    )
    
    # No augmentation for validation and test sets, only normalization
    val_test_datagen = ImageDataGenerator()
    
    # Create train generator
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training' if val_data_dir is None else None,
        seed=RANDOM_SEED
    )
    
    # Create validation generator
    if val_data_dir:
        validation_generator = val_test_datagen.flow_from_directory(
            val_data_dir,
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='binary',
            seed=RANDOM_SEED
        )
    else:
        validation_generator = train_datagen.flow_from_directory(
            train_data_dir,
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='binary',
            subset='validation',
            seed=RANDOM_SEED
        )
    
    # Create test generator
    if test_data_dir:
        test_generator = val_test_datagen.flow_from_directory(
            test_data_dir,
            target_size=IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='binary',
            shuffle=False,
            seed=RANDOM_SEED
        )
    else:
        test_generator = None
    
    return train_generator, validation_generator, test_generator

def split_dataset(data_dir, output_dir, val_split=VALIDATION_SPLIT, test_split=TEST_SPLIT):
    """
    Split dataset into train, validation, and test sets if not already split.
    
    Args:
        data_dir: Path to the directory containing images organized by category
        output_dir: Path to save the split datasets
        val_split: Fraction of data to use for validation
        test_split: Fraction of data to use for testing
    """
    categories = ['NORMAL', 'PNEUMONIA']
    
    # Check if the data directory has predefined splits
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')
    
    if os.path.exists(train_dir) and os.path.exists(test_dir):
        logger.info("Dataset already has train and test splits.")
        
        # If there's no validation set, create one from the training set
        if not os.path.exists(val_dir) or len(os.listdir(os.path.join(val_dir, 'NORMAL'))) == 0:
            logger.info("Creating validation set from training set...")
            os.makedirs(val_dir, exist_ok=True)
            
            for category in categories:
                os.makedirs(os.path.join(val_dir, category), exist_ok=True)
                train_category_dir = os.path.join(train_dir, category)
                val_category_dir = os.path.join(val_dir, category)
                
                # Get list of files in the category
                files = [f for f in os.listdir(train_category_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                # Determine number of files to move to validation set
                num_val_files = int(len(files) * val_split)
                val_files = random.sample(files, num_val_files)
                
                # Move files to validation directory
                for file in val_files:
                    src = os.path.join(train_category_dir, file)
                    dst = os.path.join(val_category_dir, file)
                    shutil.copy(src, dst)
                
                logger.info(f"Moved {len(val_files)} {category} images to validation set")
        
        return train_dir, val_dir, test_dir
    
    # If no predefined splits, create them
    logger.info("Creating train, validation, and test splits...")
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)
    
    for category in categories:
        # Create category directories
        os.makedirs(os.path.join(output_dir, 'train', category), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'val', category), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'test', category), exist_ok=True)
        
        # Get all files in category
        category_dir = os.path.join(data_dir, category)
        if not os.path.exists(category_dir):
            logger.warning(f"Category directory {category_dir} does not exist. Skipping.")
            continue
            
        files = [f for f in os.listdir(category_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Split into train, validation, and test sets
        train_files, test_files = train_test_split(
            files, test_size=test_split, random_state=RANDOM_SEED
        )
        
        train_files, val_files = train_test_split(
            train_files, test_size=val_split/(1-test_split), random_state=RANDOM_SEED
        )
        
        # Copy files to respective directories
        for file_list, split_name in [
            (train_files, 'train'), 
            (val_files, 'val'), 
            (test_files, 'test')
        ]:
            for file in file_list:
                src = os.path.join(data_dir, category, file)
                dst = os.path.join(output_dir, split_name, category, file)
                shutil.copy(src, dst)
            
            logger.info(f"Copied {len(file_list)} {category} images to {split_name} set")
    
    return (
        os.path.join(output_dir, 'train'),
        os.path.join(output_dir, 'val'),
        os.path.join(output_dir, 'test')
    )

def visualize_augmentation(image, save_path=None):
    """
    Visualize data augmentation on a sample image.
    
    Args:
        image: A sample image to apply augmentation
        save_path: Path to save the visualization (optional)
    """
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Reshape image for data generator
    image = np.expand_dims(image, 0)
    
    # Create an iterator
    aug_iter = datagen.flow(image, batch_size=1)
    
    # Generate and plot augmented images
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Original image
    axes[0].imshow(image[0])
    axes[0].set_title('Original Image')
    
    # Augmented images
    for i in range(1, 6):
        augmented_image = next(aug_iter)[0]
        axes[i].imshow(augmented_image)
        axes[i].set_title(f'Augmented {i}')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Saved augmentation visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()

def save_processed_data_info(train_generator, val_generator, test_generator=None):
    """
    Save information about the processed datasets.
    
    Args:
        train_generator: Training data generator
        val_generator: Validation data generator
        test_generator: Test data generator (optional)
    """
    info = {
        'image_size': IMAGE_SIZE,
        'train_samples': train_generator.samples,
        'train_classes': train_generator.class_indices,
        'validation_samples': val_generator.samples,
        'validation_classes': val_generator.class_indices
    }
    
    if test_generator:
        info['test_samples'] = test_generator.samples
        info['test_classes'] = test_generator.class_indices
    
    # Save as text file
    with open(os.path.join(OUTPUT_DIR, 'dataset_info.txt'), 'w') as f:
        for key, value in info.items():
            f.write(f"{key}: {value}\n")
    
    logger.info(f"Saved dataset information to {os.path.join(OUTPUT_DIR, 'dataset_info.txt')}")

def main():
    """Main function to execute the preprocessing pipeline."""
    logger.info("Starting chest X-ray image preprocessing for pneumonia detection")
    
    # Check if the base data directory exists
    if not os.path.exists(BASE_DATA_DIR):
        logger.error(f"Data directory {BASE_DATA_DIR} does not exist.")
        return
    
    # Create necessary directories
    create_directories()
    
    # Determine if dataset is already split into train, val, test
    chest_xray_dir = BASE_DATA_DIR
    
    # Check if we have the standard chest_xray directory structure
    if os.path.exists(os.path.join(BASE_DATA_DIR, 'train')):
        # Split data if needed
        train_dir, val_dir, test_dir = split_dataset(
            chest_xray_dir, OUTPUT_DIR
        )
    else:
        # If data is not structured properly, try to infer structure
        logger.warning("Standard directory structure not found. Attempting to infer structure.")
        
        # Check if categories are at the top level
        if (os.path.exists(os.path.join(BASE_DATA_DIR, 'NORMAL')) and
            os.path.exists(os.path.join(BASE_DATA_DIR, 'PNEUMONIA'))):
            logger.info("Found categories at top level. Creating splits.")
            train_dir, val_dir, test_dir = split_dataset(
                BASE_DATA_DIR, OUTPUT_DIR
            )
        else:
            # Check for other common dataset structures
            possible_data_dir = os.path.join(BASE_DATA_DIR, 'chest_xray')
            if os.path.exists(possible_data_dir):
                logger.info(f"Found data in {possible_data_dir}")
                train_dir, val_dir, test_dir = split_dataset(
                    possible_data_dir, OUTPUT_DIR
                )
            else:
                # Last attempt: search for any directory with image files
                logger.warning("No standard structure found. Searching for images...")
                img_found = False
                
                for root, dirs, files in os.walk(BASE_DATA_DIR):
                    if any(f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in files):
                        logger.info(f"Found images in {root}")
                        img_found = True
                        
                        # Determine category from parent directory name
                        for file in files:
                            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                                if 'normal' in root.lower():
                                    category = 'NORMAL'
                                elif 'pneumonia' in root.lower():
                                    category = 'PNEUMONIA'
                                else:
                                    continue
                                
                                # Create category directory if needed
                                os.makedirs(os.path.join(OUTPUT_DIR, 'train', category), exist_ok=True)
                                
                                # Copy file
                                src = os.path.join(root, file)
                                dst = os.path.join(OUTPUT_DIR, 'train', category, file)
                                shutil.copy(src, dst)
                
                if img_found:
                    # If we found and copied images, now split the collated data
                    train_dir, val_dir, test_dir = split_dataset(
                        os.path.join(OUTPUT_DIR, 'train'), OUTPUT_DIR
                    )
                else:
                    logger.error("No usable image data found in any directory structure")
                    return
    
    # Create data generators
    logger.info("Creating data generators...")
    train_generator, val_generator, test_generator = create_data_generators(
        train_dir, val_dir, test_dir
    )
    
    # Save information about the processed data
    save_processed_data_info(train_generator, val_generator, test_generator)
    
    # Visualize data augmentation on a sample image
    logger.info("Visualizing data augmentation on a sample image...")
    try:
        # Get a sample image from the training set
        for images, _ in train_generator:
            sample_image = images[0]
            break
        
        # Visualize augmentations
        os.makedirs(os.path.join(OUTPUT_DIR, 'visualizations'), exist_ok=True)
        visualize_augmentation(
            sample_image,
            save_path=os.path.join(OUTPUT_DIR, 'visualizations', 'augmentation_examples.png')
        )
    except Exception as e:
        logger.error(f"Error visualizing augmentation: {e}")
    
    logger.info("Preprocessing completed successfully!")
    logger.info(f"Processed data available in {OUTPUT_DIR}")
    logger.info(f"Train samples: {train_generator.samples}")
    logger.info(f"Validation samples: {val_generator.samples}")
    if test_generator:
        logger.info(f"Test samples: {test_generator.samples}")

if __name__ == "__main__":
    main()
