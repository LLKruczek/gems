#!/usr/bin/env python3
"""
Hierarchical Gemstone Classification using Random Forest

This script implements a two-stage classification approach with Random Forest:
1. Main model: Classifies all 68 gemstones
2. Specialized models: Fine-grained classification for confusing color groups
   - Yellow stones classifier (Sapphire Yellow, Citrine, Beryl Golden, etc.)
   - Purple stones classifier (Sapphire Purple, Amethyst, Ametrine)

Features:
- Fast training (seconds/minutes vs hours for deep learning)
- Can use k-means clustering as preprocessing (optional)
- Uses traditional ML features: color histograms, LBP, texture, etc.
- Based on the best-performing approach from the original paper (69.4% accuracy)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os
import time
import warnings
warnings.filterwarnings('ignore')

# Image processing libraries
from PIL import Image
import cv2
from skimage import feature
import mahotas
from scipy import ndimage

# Clustering for preprocessing (optional)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Configuration
DATA_DIR = '.'
SEED = 1
USE_KMEANS_PREPROCESSING = True  # Use k-means clustering as feature preprocessing
KMEANS_N_CLUSTERS = 32  # Number of clusters for k-means
USE_PCA = True  # Apply PCA after k-means
PCA_N_COMPONENTS = 16  # PCA components (if using k-means)

# Random Forest hyperparameters - UPDATED TO MATCH IMPROVED MODEL
RF_N_ESTIMATORS = 300  # Increased from 200
RF_MAX_DEPTH = 20  # Limit depth to reduce overfitting (was None)
RF_MIN_SAMPLES_SPLIT = 5  # Increased from 2
RF_MIN_SAMPLES_LEAF = 2  # Increased from 1
RF_MAX_FEATURES = 'sqrt'  # Limit features per split
RF_CLASS_WEIGHT = 'balanced'  # Handle class imbalance
RF_N_JOBS = -1  # Use all CPU cores

# Color group definitions
YELLOW_STONES = [
    'Aventurine Yellow',
    'Beryl Golden', 
    'Citrine',
    'Quartz Lemon',
    'Sapphire Yellow',
    'Sunstone'
]

PURPLE_STONES = [
    'Amethyst',
    'Sapphire Purple',
    'Ametrine'
]


def extract_color_histogram(image, bins=8, color_space='RGB'):
    """Extract color histogram features."""
    if color_space == 'RGB':
        hist_r = np.histogram(image[:, :, 0].flatten(), bins=bins, range=(0, 256))[0]
        hist_g = np.histogram(image[:, :, 1].flatten(), bins=bins, range=(0, 256))[0]
        hist_b = np.histogram(image[:, :, 2].flatten(), bins=bins, range=(0, 256))[0]
        return np.concatenate([hist_r, hist_g, hist_b])
    elif color_space == 'HSV':
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        hist_h = np.histogram(hsv[:, :, 0].flatten(), bins=bins, range=(0, 180))[0]
        hist_s = np.histogram(hsv[:, :, 1].flatten(), bins=bins, range=(0, 256))[0]
        hist_v = np.histogram(hsv[:, :, 2].flatten(), bins=bins, range=(0, 256))[0]
        return np.concatenate([hist_h, hist_s, hist_v])
    elif color_space == 'LAB':
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        hist_l = np.histogram(lab[:, :, 0].flatten(), bins=bins, range=(0, 256))[0]
        hist_a = np.histogram(lab[:, :, 1].flatten(), bins=bins, range=(0, 256))[0]
        hist_b = np.histogram(lab[:, :, 2].flatten(), bins=bins, range=(0, 256))[0]
        return np.concatenate([hist_l, hist_a, hist_b])


def extract_lbp(image, radius=3, n_points=24):
    """Extract Local Binary Pattern (LBP) features."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    lbp = feature.local_binary_pattern(gray, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2))
    hist = hist.astype(float)
    hist /= (hist.sum() + 1e-8)  # Normalize
    return hist


def extract_haralick_texture(image):
    """Extract Haralick texture features."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    try:
        features = mahotas.features.haralick(gray).mean(axis=0)
        return features
    except:
        # Fallback if mahotas fails
        return np.zeros(13)


def extract_glcm_features(image):
    """Extract Grey-Level Co-occurrence Matrix (GLCM) features."""
    from skimage.feature import graycomatrix, graycoprops
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Quantize to reduce computation
    gray = (gray / 16).astype(np.uint8)
    
    # Calculate GLCM
    glcm = graycomatrix(gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                        levels=16, symmetric=True, normed=True)
    
    # Extract properties
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    features = []
    for prop in properties:
        features.append(graycoprops(glcm, prop).mean())
    
    return np.array(features)


def extract_kmeans_features(image, kmeans_model, scaler, pca_model=None):
    """
    Extract k-means clustering features from image.
    
    This creates a histogram of cluster assignments, which can be a powerful
    feature for distinguishing similar-colored stones.
    """
    # Reshape image to pixels
    h, w, c = image.shape
    pixels = image.reshape(-1, c).astype(np.float32)
    
    # Remove black pixels (background)
    non_black_mask = np.sum(pixels, axis=1) > 30  # Threshold for black
    if np.sum(non_black_mask) < 100:
        return np.zeros(KMEANS_N_CLUSTERS)
    
    non_black_pixels = pixels[non_black_mask]
    
    # Normalize (if scaler was used during training)
    if scaler is not None:
        non_black_pixels = scaler.transform(non_black_pixels)
    
    # Apply PCA if used
    if pca_model is not None:
        non_black_pixels = pca_model.transform(non_black_pixels)
    
    # Predict clusters
    cluster_labels = kmeans_model.predict(non_black_pixels)
    
    # Create histogram of cluster assignments
    hist, _ = np.histogram(cluster_labels, bins=KMEANS_N_CLUSTERS, range=(0, KMEANS_N_CLUSTERS))
    hist = hist.astype(float)
    hist /= (hist.sum() + 1e-8)  # Normalize
    
    return hist


def extract_all_features(image_path, kmeans_model=None, scaler=None, pca_model=None, use_kmeans_override=None):
    """Extract all features from an image.
    
    Args:
        image_path: Path to image
        kmeans_model: K-means model (None if not using k-means)
        scaler: Scaler for k-means (None if not using k-means)
        pca_model: PCA model for k-means (None if not using k-means)
        use_kmeans_override: If provided, override the global USE_KMEANS_PREPROCESSING flag
    """
    # Load image
    image = np.array(Image.open(image_path).convert('RGB'))
    
    features = []
    
    # Color histograms (best feature from paper)
    features.append(extract_color_histogram(image, bins=8, color_space='RGB'))
    features.append(extract_color_histogram(image, bins=8, color_space='HSV'))
    features.append(extract_color_histogram(image, bins=8, color_space='LAB'))
    
    # Local Binary Pattern (LBP) - second best feature from paper
    features.append(extract_lbp(image, radius=3, n_points=24))
    
    # Haralick texture
    features.append(extract_haralick_texture(image))
    
    # GLCM features
    features.append(extract_glcm_features(image))
    
    # K-means clustering features (optional preprocessing)
    # Use override if provided, otherwise check global flag AND that kmeans_model is not None
    should_use_kmeans = use_kmeans_override if use_kmeans_override is not None else (USE_KMEANS_PREPROCESSING and kmeans_model is not None)
    
    if should_use_kmeans and kmeans_model is not None:
        kmeans_features = extract_kmeans_features(image, kmeans_model, scaler, pca_model)
        features.append(kmeans_features)
    
    return np.concatenate(features)


def load_dataset(data_dir, subset_classes=None, kmeans_model=None, scaler=None, pca_model=None, use_kmeans_override=None):
    """Load dataset and extract features.
    
    Args:
        data_dir: Directory containing train_masked and test_masked
        subset_classes: List of classes to load (None for all)
        kmeans_model: K-means model for feature extraction (optional)
        scaler: Scaler for k-means (optional)
        pca_model: PCA model for k-means (optional)
        use_kmeans_override: Override for USE_KMEANS_PREPROCESSING flag
    """
    print(f"\nLoading dataset from {data_dir}...")
    
    train_dir = os.path.join(data_dir, 'train_masked')
    test_dir = os.path.join(data_dir, 'test_masked')
    
    # Get all class directories
    if subset_classes is None:
        classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    else:
        classes = subset_classes
    
    print(f"Classes: {len(classes)}")
    print(f"  {', '.join(classes)}")
    
    # Extract features from training set
    print("\nExtracting features from training images...")
    X_train = []
    y_train = []
    
    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(train_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"  Warning: {class_dir} not found, skipping...")
            continue
        
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"  Processing {class_name}: {len(images)} images")
        
        for img_file in images:
            img_path = os.path.join(class_dir, img_file)
            try:
                features = extract_all_features(img_path, kmeans_model, scaler, pca_model, use_kmeans_override)
                X_train.append(features)
                y_train.append(class_idx)
            except Exception as e:
                print(f"    Error processing {img_path}: {e}")
                continue
    
    # Extract features from test set
    print("\nExtracting features from test images...")
    X_test = []
    y_test = []
    
    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.exists(class_dir):
            continue
        
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_file in images:
            img_path = os.path.join(class_dir, img_file)
            try:
                features = extract_all_features(img_path, kmeans_model, scaler, pca_model, use_kmeans_override)
                X_test.append(features)
                y_test.append(class_idx)
            except Exception as e:
                continue
    
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    print(f"\nTraining set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test, classes


def train_kmeans_preprocessing(X_train, X_test):
    """Train k-means model for feature preprocessing."""
    print("\n" + "="*70)
    print("Training K-means Preprocessing Model")
    print("="*70)
    
    # Extract pixel features from a sample of images
    print("Extracting pixel features for k-means...")
    pixel_features = []
    
    train_dir = os.path.join(DATA_DIR, 'train_masked')
    classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    
    # Sample images for k-means training
    max_images = 100  # Limit for speed
    image_count = 0
    
    for class_name in classes:
        if image_count >= max_images:
            break
        class_dir = os.path.join(train_dir, class_name)
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_file in images[:5]:  # 5 images per class
            if image_count >= max_images:
                break
            img_path = os.path.join(class_dir, img_file)
            try:
                image = np.array(Image.open(img_path).convert('RGB'))
                h, w, c = image.shape
                pixels = image.reshape(-1, c).astype(np.float32)
                
                # Remove black pixels
                non_black_mask = np.sum(pixels, axis=1) > 30
                if np.sum(non_black_mask) > 100:
                    non_black_pixels = pixels[non_black_mask]
                    # Sample pixels
                    if len(non_black_pixels) > 1000:
                        indices = np.random.choice(len(non_black_pixels), 1000, replace=False)
                        non_black_pixels = non_black_pixels[indices]
                    pixel_features.append(non_black_pixels)
                    image_count += 1
            except:
                continue
    
    if len(pixel_features) == 0:
        print("Warning: Could not extract pixel features for k-means")
        return None, None, None
    
    all_pixels = np.vstack(pixel_features)
    print(f"Extracted {len(all_pixels):,} pixels for k-means training")
    
    # Normalize
    scaler = StandardScaler()
    all_pixels_scaled = scaler.fit_transform(all_pixels)
    
    # Apply PCA if enabled
    pca_model = None
    if USE_PCA and all_pixels_scaled.shape[1] > PCA_N_COMPONENTS:
        print(f"Applying PCA: {all_pixels_scaled.shape[1]} -> {PCA_N_COMPONENTS}")
        pca_model = PCA(n_components=PCA_N_COMPONENTS)
        all_pixels_scaled = pca_model.fit_transform(all_pixels_scaled)
    
    # Train k-means
    print(f"Training k-means with {KMEANS_N_CLUSTERS} clusters...")
    kmeans = KMeans(n_clusters=KMEANS_N_CLUSTERS, random_state=SEED, n_init=3)
    kmeans.fit(all_pixels_scaled)
    
    print("K-means preprocessing model trained!")
    return kmeans, scaler, pca_model


def train_specialized_model(color_name, color_classes):
    """Train a specialized Random Forest model for a color group."""
    print("\n" + "="*70)
    print(f"Training {color_name} Stones Random Forest Classifier")
    print("="*70)
    print(f"Classes: {', '.join(color_classes)}")
    
    # Train k-means preprocessing if enabled
    kmeans_model = None
    scaler = None
    pca_model = None
    
    if USE_KMEANS_PREPROCESSING:
        kmeans_model, scaler, pca_model = train_kmeans_preprocessing(None, None)
    
    # Load dataset for this color group - PASS K-MEANS MODELS
    X_train, X_test, y_train, y_test, classes = load_dataset(
        DATA_DIR, 
        subset_classes=color_classes,
        kmeans_model=kmeans_model,
        scaler=scaler,
        pca_model=pca_model,
        use_kmeans_override=USE_KMEANS_PREPROCESSING
    )
    
    if len(X_train) == 0:
        print(f"Error: No training data found for {color_name} stones")
        return None, None
    
    print(f"\nTraining Random Forest...")
    print(f"  n_estimators: {RF_N_ESTIMATORS}")
    print(f"  max_depth: {RF_MAX_DEPTH}")
    print(f"  min_samples_split: {RF_MIN_SAMPLES_SPLIT}")
    print(f"  min_samples_leaf: {RF_MIN_SAMPLES_LEAF}")
    print(f"  max_features: {RF_MAX_FEATURES}")
    print(f"  class_weight: {RF_CLASS_WEIGHT}")
    print(f"  Using k-means: {USE_KMEANS_PREPROCESSING}")
    print(f"  Feature count: {X_train.shape[1]}")
    
    start_time = time.time()
    
    # Train Random Forest with improved hyperparameters
    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_split=RF_MIN_SAMPLES_SPLIT,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        max_features=RF_MAX_FEATURES,
        class_weight=RF_CLASS_WEIGHT,
        n_jobs=RF_N_JOBS,
        random_state=SEED,
        verbose=1
    )
    
    rf.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate
    train_pred = rf.predict(X_train)
    test_pred = rf.predict(X_test)
    
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    print(f"\n{color_name} Classifier Results:")
    print(f"Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Gap (overfitting measure): {train_acc - test_acc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, test_pred, target_names=classes))
    
    # Save model
    model_path = f'rf_{color_name.lower()}_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': rf,
            'classes': classes,
            'kmeans': kmeans_model,
            'scaler': scaler,
            'pca': pca_model,
            'use_kmeans': USE_KMEANS_PREPROCESSING
        }, f)
    print(f"\nModel saved to {model_path}")
    
    # Save confusion matrix
    cm = confusion_matrix(y_test, test_pred)
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=classes, yticklabels=classes)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'{color_name} Stones Random Forest Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f'rf_{color_name.lower()}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return rf, classes


def main():
    """Main training function."""
    print("="*70)
    print("Hierarchical Random Forest Gemstone Classification")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  K-means preprocessing: {USE_KMEANS_PREPROCESSING}")
    print(f"  Random Forest trees: {RF_N_ESTIMATORS}")
    print(f"  Max depth: {RF_MAX_DEPTH}")
    
    # Train specialized models
    yellow_rf, yellow_classes = train_specialized_model('Yellow', YELLOW_STONES)
    purple_rf, purple_classes = train_specialized_model('Purple', PURPLE_STONES)
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print("\nSaved models:")
    print("  - rf_yellow_model.pkl")
    print("  - rf_purple_model.pkl")
    print("\nThese models can be used with hierarchical_inference_rf.py")


if __name__ == '__main__':
    main()

