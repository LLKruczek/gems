#!/usr/bin/env python3
"""
Train Main Random Forest Classifier for All 68 Gemstone Classes

This script trains the main Random Forest model that classifies all 68 gemstone classes.
This model is used as the first stage in the hierarchical classification system.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
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

# Import feature extraction functions from hierarchical script
import sys
sys.path.append('.')
from train_hierarchical_rf import (
    extract_color_histogram, extract_lbp, extract_haralick_texture,
    extract_glcm_features, extract_kmeans_features, extract_all_features,
    train_kmeans_preprocessing, load_dataset
)

# Configuration
DATA_DIR = '.'
SEED = 1
USE_KMEANS_PREPROCESSING = True  # Use k-means clustering as feature preprocessing
KMEANS_N_CLUSTERS = 32
USE_PCA = True
PCA_N_COMPONENTS = 16

# Random Forest hyperparameters
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = None
RF_MIN_SAMPLES_SPLIT = 2
RF_MIN_SAMPLES_LEAF = 1
RF_N_JOBS = -1

# All 68 classes
ALL_CLASSES = [
    'Alexandrite', 'Almandine', 'Amazonite', 'Amber', 'Amethyst', 'Ametrine',
    'Andradite', 'Aquamarine', 'Aventurine Green', 'Aventurine Yellow', 'Benitoite',
    'Beryl Golden', 'Bixbite', 'Bloodstone', 'Blue Lace Agate', 'Carnelian',
    'Chalcedony', 'Chalcedony Blue', 'Chrome Diopside', 'Chrysoberyl', 'Chrysocolla',
    'Chrysoprase', 'Citrine', 'Coral', 'Diamond', 'Diaspore', 'Dumortierite',
    'Emerald', 'Fluorite', 'Hessonite', 'Iolite', 'Jasper', 'Kunzite', 'Kyanite',
    'Lapis Lazuli', 'Malachite', 'Onyx Black', 'Onyx Green', 'Onyx Red', 'Peridot',
    'Prehnite', 'Pyrite', 'Pyrope', 'Quartz Beer', 'Quartz Lemon', 'Quartz Rutilated',
    'Quartz Smoky', 'Rhodochrosite', 'Rhodolite', 'Rhodonite', 'Ruby', 'Sapphire Blue',
    'Sapphire Pink', 'Sapphire Purple', 'Sapphire Yellow', 'Serpentine', 'Sodalite',
    'Spessartite', 'Sphene', 'Sunstone', 'Tanzanite', 'Tigers Eye', 'Topaz',
    'Tourmaline', 'Tsavorite', 'Turquoise', 'Zircon', 'Zoisite'
]


def train_main_model():
    """Train the main Random Forest classifier for all 68 classes."""
    # Use local variable to track k-means usage
    use_kmeans = USE_KMEANS_PREPROCESSING
    
    print("="*70)
    print("Training Main Random Forest Classifier (68 Classes)")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  K-means preprocessing: {use_kmeans}")
    print(f"  Random Forest trees: {RF_N_ESTIMATORS}")
    print(f"  Max depth: {RF_MAX_DEPTH}")
    
    # Train k-means preprocessing if enabled
    kmeans_model = None
    scaler = None
    pca_model = None
    
    if use_kmeans:
        print("\n" + "-"*70)
        print("Step 1: Training K-means Preprocessing")
        print("-"*70)
        kmeans_model, scaler, pca_model = train_kmeans_preprocessing(None, None)
        if kmeans_model is None:
            print("Warning: K-means training failed, continuing without it...")
            use_kmeans = False
    
    # Load full dataset
    print("\n" + "-"*70)
    print("Step 2: Loading Dataset and Extracting Features")
    print("-"*70)
    
    # We need to modify load_dataset to use k-means if available
    # For now, let's extract features manually with k-means support
    print(f"\nLoading dataset from {DATA_DIR}...")
    
    train_dir = os.path.join(DATA_DIR, 'train_masked')
    test_dir = os.path.join(DATA_DIR, 'test_masked')
    
    classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    print(f"Total classes: {len(classes)}")
    
    # Extract features from training set
    print("\nExtracting features from training images...")
    X_train = []
    y_train = []
    
    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(train_dir, class_name)
        if not os.path.exists(class_dir):
            continue
        
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"  Processing {class_name}: {len(images)} images", end='\r')
        
        for img_file in images:
            img_path = os.path.join(class_dir, img_file)
            try:
                # Use extract_all_features with k-means support
                if use_kmeans and kmeans_model is not None:
                    features = extract_all_features(img_path, kmeans_model, scaler, pca_model)
                else:
                    features = extract_all_features(img_path, None, None, None)
                X_train.append(features)
                y_train.append(class_idx)
            except Exception as e:
                continue
        print()  # New line after each class
    
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
                if use_kmeans and kmeans_model is not None:
                    features = extract_all_features(img_path, kmeans_model, scaler, pca_model)
                else:
                    features = extract_all_features(img_path, None, None, None)
                X_test.append(features)
                y_test.append(class_idx)
            except Exception as e:
                continue
    
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    print(f"\nDataset Summary:")
    print(f"  Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"  Test set: {X_test.shape[0]} samples")
    
    # Train Random Forest
    print("\n" + "-"*70)
    print("Step 3: Training Random Forest")
    print("-"*70)
    print(f"\nTraining Random Forest with {RF_N_ESTIMATORS} trees...")
    print(f"  This may take a few minutes...")
    
    start_time = time.time()
    
    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_split=RF_MIN_SAMPLES_SPLIT,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        n_jobs=RF_N_JOBS,
        random_state=SEED,
        verbose=1
    )
    
    rf.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    # Evaluate
    print("\n" + "-"*70)
    print("Step 4: Evaluation")
    print("-"*70)
    
    train_pred = rf.predict(X_train)
    test_pred = rf.predict(X_test)
    
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    print(f"\nMain Model Results:")
    print(f"  Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"  Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test, test_pred, target_names=classes))
    
    # Save model
    model_path = 'rf_main_model.pkl'
    print(f"\nSaving model to {model_path}...")
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': rf,
            'classes': classes,
            'kmeans': kmeans_model,
            'scaler': scaler,
            'pca': pca_model,
            'use_kmeans': use_kmeans,
            'feature_names': [
                'RGB_histogram', 'HSV_histogram', 'LAB_histogram',
                'LBP', 'Haralick_texture', 'GLCM',
                'KMeans_histogram' if use_kmeans else None
            ]
        }, f)
    print(f"Model saved!")
    
    # Save confusion matrix
    print("\nGenerating confusion matrix...")
    cm = confusion_matrix(y_test, test_pred)
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, ax = plt.subplots(figsize=(20, 20), dpi=300)
    
    # Create mask for zeros (to make visualization cleaner)
    cm_mask = np.zeros_like(cm, dtype=bool)
    cm_mask[cm == 0] = True
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=classes, yticklabels=classes, mask=cm_mask,
                cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Main Random Forest Confusion Matrix (68 Classes)', fontsize=14)
    plt.xticks(rotation=90, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig('rf_main_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Confusion matrix saved to rf_main_confusion_matrix.png")
    
    # Feature importance
    print("\nTop 20 Most Important Features:")
    feature_importances = rf.feature_importances_
    top_indices = np.argsort(feature_importances)[-20:][::-1]
    
    feature_names = []
    idx = 0
    # RGB histogram (24 features)
    for i in range(24):
        feature_names.append(f'RGB_hist_{i}')
    # HSV histogram (24 features)
    for i in range(24):
        feature_names.append(f'HSV_hist_{i}')
    # LAB histogram (24 features)
    for i in range(24):
        feature_names.append(f'LAB_hist_{i}')
    # LBP (26 features)
    for i in range(26):
        feature_names.append(f'LBP_{i}')
    # Haralick (13 features)
    for i in range(13):
        feature_names.append(f'Haralick_{i}')
    # GLCM (5 features)
    for i in range(5):
        feature_names.append(f'GLCM_{i}')
    # K-means (if used)
    if use_kmeans:
        for i in range(KMEANS_N_CLUSTERS):
            feature_names.append(f'KMeans_{i}')
    
    for idx in top_indices:
        if idx < len(feature_names):
            print(f"  {feature_names[idx]}: {feature_importances[idx]:.6f}")
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"\nModel saved to: {model_path}")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print("\nNext step: Use evaluate_hierarchical_rf.py to test the full hierarchical system")
    
    return rf, classes, test_acc


if __name__ == '__main__':
    train_main_model()

