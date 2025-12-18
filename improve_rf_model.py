#!/usr/bin/env python3
"""
Improved Random Forest Training with Hyperparameter Tuning and Class Balancing

This script addresses the issues found in the current model:
1. Overfitting (100% train vs 67.61% test)
2. Poor performance on some classes (Beryl Golden, Citrine, etc.)
3. Hyperparameter optimization
4. Class imbalance handling
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, cross_val_score
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier
import pickle
import os
import time
import warnings
warnings.filterwarnings('ignore')

# Import feature extraction
import sys
sys.path.append('.')
from train_hierarchical_rf import extract_all_features, train_kmeans_preprocessing
from train_main_rf import load_dataset

# Configuration
DATA_DIR = '.'
SEED = 1
USE_KMEANS_PREPROCESSING = True
USE_CLASS_BALANCING = True  # Use SMOTE or BalancedRandomForest
USE_HYPERPARAMETER_TUNING = True  # Grid search for best params

# Improved Random Forest hyperparameters
RF_N_ESTIMATORS = 300  # Increased from 200
RF_MAX_DEPTH = 20  # Limit depth to reduce overfitting (was None)
RF_MIN_SAMPLES_SPLIT = 5  # Increased from 2
RF_MIN_SAMPLES_LEAF = 2  # Increased from 1
RF_MAX_FEATURES = 'sqrt'  # Limit features per split
RF_CLASS_WEIGHT = 'balanced'  # Handle class imbalance
RF_N_JOBS = -1


def train_improved_model():
    """Train improved Random Forest with better hyperparameters."""
    print("="*70)
    print("Training Improved Random Forest Model")
    print("="*70)
    print("\nImprovements:")
    print("  1. Reduced max_depth to prevent overfitting")
    print("  2. Increased min_samples_split/leaf for regularization")
    print("  3. Class balancing (balanced class weights)")
    print("  4. More trees (300 vs 200)")
    print("  5. Limited features per split (sqrt)")
    
    # Train k-means if enabled
    kmeans_model = None
    scaler = None
    pca_model = None
    use_kmeans = USE_KMEANS_PREPROCESSING
    
    if use_kmeans:
        print("\nTraining k-means preprocessing...")
        kmeans_model, scaler, pca_model = train_kmeans_preprocessing(None, None)
        if kmeans_model is None:
            use_kmeans = False
    
    # Load dataset (simplified - you'll need to adapt load_dataset)
    print("\nLoading dataset...")
    train_dir = os.path.join(DATA_DIR, 'train_masked')
    test_dir = os.path.join(DATA_DIR, 'test_masked')
    
    classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    
    # Extract features
    X_train, y_train = [], []
    X_test, y_test = [], []
    
    print("Extracting features from training images...")
    for class_idx, class_name in enumerate(classes):
        train_class_dir = os.path.join(train_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)
        
        # Training
        if os.path.exists(train_class_dir):
            images = [f for f in os.listdir(train_class_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            for img_file in images:
                img_path = os.path.join(train_class_dir, img_file)
                try:
                    if use_kmeans and kmeans_model is not None:
                        features = extract_all_features(img_path, kmeans_model, scaler, pca_model)
                    else:
                        features = extract_all_features(img_path, None, None, None)
                    X_train.append(features)
                    y_train.append(class_idx)
                except:
                    continue
        
        # Test
        if os.path.exists(test_class_dir):
            images = [f for f in os.listdir(test_class_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            for img_file in images:
                img_path = os.path.join(test_class_dir, img_file)
                try:
                    if use_kmeans and kmeans_model is not None:
                        features = extract_all_features(img_path, kmeans_model, scaler, pca_model)
                    else:
                        features = extract_all_features(img_path, None, None, None)
                    X_test.append(features)
                    y_test.append(class_idx)
                except:
                    continue
    
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    print(f"\nDataset: {X_train.shape[0]} train, {X_test.shape[0]} test")
    print(f"Features: {X_train.shape[1]}")
    
    # Apply SMOTE for class balancing if enabled
    if USE_CLASS_BALANCING:
        print("\nApplying SMOTE for class balancing...")
        smote = SMOTE(random_state=SEED, k_neighbors=3)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        print(f"After SMOTE: {X_train_balanced.shape[0]} samples (was {X_train.shape[0]})")
    else:
        X_train_balanced, y_train_balanced = X_train, y_train
    
    # Hyperparameter tuning
    if USE_HYPERPARAMETER_TUNING:
        print("\n" + "-"*70)
        print("Hyperparameter Tuning (this may take a while...)")
        print("-"*70)
        
        param_grid = {
            'n_estimators': [200, 300, 400],
            'max_depth': [15, 20, 25, None],
            'min_samples_split': [3, 5, 7],
            'min_samples_leaf': [1, 2, 3],
            'max_features': ['sqrt', 'log2']
        }
        
        # Use smaller subset for faster tuning
        sample_size = min(1000, len(X_train_balanced))
        indices = np.random.choice(len(X_train_balanced), sample_size, replace=False)
        X_tune = X_train_balanced[indices]
        y_tune = y_train_balanced[indices]
        
        rf_base = RandomForestClassifier(
            class_weight='balanced',
            n_jobs=RF_N_JOBS,
            random_state=SEED
        )
        
        grid_search = GridSearchCV(
            rf_base,
            param_grid,
            cv=3,  # 3-fold CV for speed
            scoring='accuracy',
            n_jobs=RF_N_JOBS,
            verbose=1
        )
        
        print("Running grid search...")
        grid_search.fit(X_tune, y_tune)
        
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        best_params = grid_search.best_params_
    else:
        best_params = {
            'n_estimators': RF_N_ESTIMATORS,
            'max_depth': RF_MAX_DEPTH,
            'min_samples_split': RF_MIN_SAMPLES_SPLIT,
            'min_samples_leaf': RF_MIN_SAMPLES_LEAF,
            'max_features': RF_MAX_FEATURES
        }
    
    # Train final model with best parameters
    print("\n" + "-"*70)
    print("Training Final Model")
    print("-"*70)
    
    rf = RandomForestClassifier(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],
        max_features=best_params['max_features'],
        class_weight='balanced',  # Handle class imbalance
        n_jobs=RF_N_JOBS,
        random_state=SEED,
        verbose=1
    )
    
    print(f"Training with {best_params['n_estimators']} trees...")
    start_time = time.time()
    rf.fit(X_train_balanced, y_train_balanced)
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate
    train_pred = rf.predict(X_train)
    test_pred = rf.predict(X_test)
    
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    print("\n" + "="*70)
    print("Results Comparison")
    print("="*70)
    print(f"Training Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"Gap (overfitting measure): {train_acc - test_acc:.4f}")
    print(f"\nOriginal paper: 69.4%")
    print(f"Your previous model: 67.61%")
    print(f"Improved model: {test_acc*100:.2f}%")
    print(f"Improvement: {test_acc*100 - 67.61:.2f}%")
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test, test_pred, target_names=classes))
    
    # Save model
    model_path = 'rf_main_improved.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': rf,
            'classes': classes,
            'kmeans': kmeans_model,
            'scaler': scaler,
            'pca': pca_model,
            'use_kmeans': use_kmeans,
            'best_params': best_params
        }, f)
    
    print(f"\nModel saved to {model_path}")
    
    return rf, classes, test_acc


if __name__ == '__main__':
    train_improved_model()

