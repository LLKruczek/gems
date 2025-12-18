#!/usr/bin/env python3
"""
Evaluate Improved Random Forest Model with Top-K Accuracy

This script evaluates the improved RF model and computes:
- Top-1 accuracy (standard accuracy)
- Top-3 accuracy
- Top-5 accuracy (optional)
"""

import numpy as np
import pickle
import os
from sklearn.metrics import accuracy_score, top_k_accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import feature extraction
import sys
sys.path.append('.')
from train_hierarchical_rf import extract_all_features

# Configuration
DATA_DIR = '.'
MODEL_PATH = 'rf_main_improved.pkl'


def evaluate_improved_model(model_path=MODEL_PATH, data_dir=DATA_DIR, split='test'):
    """Evaluate the improved RF model with top-k accuracy metrics."""
    
    # Load model
    print("="*70)
    print("Evaluating Improved Random Forest Model")
    print("="*70)
    print(f"\nLoading model from {model_path}...")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
        model = model_data['model']
        classes = model_data['classes']
        kmeans_model = model_data.get('kmeans', None)
        scaler = model_data.get('scaler', None)
        pca_model = model_data.get('pca', None)
        use_kmeans = model_data.get('use_kmeans', False)
    
    print(f"Model loaded: {len(classes)} classes")
    print(f"Using k-means: {use_kmeans}")
    
    # Load test dataset
    split_dir = os.path.join(data_dir, f'{split}_masked')
    if not os.path.exists(split_dir):
        raise FileNotFoundError(f"Data directory not found: {split_dir}")
    
    print(f"\nLoading {split} dataset from {split_dir}...")
    
    X_test = []
    y_test = []
    
    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"  Warning: {class_dir} not found, skipping...")
            continue
        
        images = [f for f in os.listdir(class_dir) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"  Processing {class_name}: {len(images)} images", end='\r')
        
        for img_file in images:
            img_path = os.path.join(class_dir, img_file)
            try:
                if use_kmeans and kmeans_model is not None:
                    features = extract_all_features(
                        img_path, kmeans_model, scaler, pca_model, 
                        use_kmeans_override=use_kmeans
                    )
                else:
                    features = extract_all_features(img_path, None, None, None)
                X_test.append(features)
                y_test.append(class_idx)
            except Exception as e:
                print(f"\n    Error processing {img_path}: {e}")
                continue
    
    print()  # New line
    
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    print(f"\nDataset: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    
    # Make predictions
    print("\nMaking predictions...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Calculate accuracies
    top1_acc = accuracy_score(y_test, y_pred)
    top3_acc = top_k_accuracy_score(y_test, y_pred_proba, k=3)
    top5_acc = top_k_accuracy_score(y_test, y_pred_proba, k=5)
    
    # Print results
    print("\n" + "="*70)
    print("Evaluation Results")
    print("="*70)
    print(f"\nTop-1 Accuracy (Standard): {top1_acc:.4f} ({top1_acc*100:.2f}%)")
    print(f"Top-3 Accuracy:            {top3_acc:.4f} ({top3_acc*100:.2f}%)")
    print(f"Top-5 Accuracy:            {top5_acc:.4f} ({top5_acc*100:.2f}%)")
    print(f"\nImprovement from Top-1 to Top-3: {(top3_acc - top1_acc)*100:.2f}%")
    print(f"Improvement from Top-1 to Top-5: {(top5_acc - top1_acc)*100:.2f}%")
    
    # Classification report
    print(f"\nClassification Report (Top-1):")
    print(classification_report(y_test, y_pred, target_names=classes))
    
    # Per-class top-3 accuracy analysis
    print("\n" + "-"*70)
    print("Per-Class Top-3 Analysis")
    print("-"*70)
    
    class_top3_correct = {i: 0 for i in range(len(classes))}
    class_counts = {i: 0 for i in range(len(classes))}
    
    for true_label, proba in zip(y_test, y_pred_proba):
        class_counts[true_label] += 1
        top3_indices = np.argsort(proba)[-3:][::-1]  # Get top 3 predictions
        if true_label in top3_indices:
            class_top3_correct[true_label] += 1
    
    print(f"\n{'Class':<25} {'Count':<8} {'Top-3 Correct':<15} {'Top-3 Acc':<12}")
    print("-" * 70)
    
    for i, class_name in enumerate(classes):
        count = class_counts[i]
        if count > 0:
            top3_correct = class_top3_correct[i]
            top3_acc_class = top3_correct / count
            print(f"{class_name:<25} {count:<8} {top3_correct:<15} {top3_acc_class:.4f} ({top3_acc_class*100:.2f}%)")
    
    # Generate and save confusion matrix
    print("\n" + "-"*70)
    print("Generating Confusion Matrix...")
    print("-"*70)
    
    cm = confusion_matrix(y_test, y_pred)
    
    # Create figure with appropriate size for 68 classes
    fig, ax = plt.subplots(figsize=(20, 18))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', ax=ax,
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Improved Random Forest Model - Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xticks(rotation=90, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    
    # Save confusion matrix
    output_path = 'confusion_matrix_rf_improved.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {output_path}")
    
    return {
        'top1_accuracy': top1_acc,
        'top3_accuracy': top3_acc,
        'top5_accuracy': top5_acc,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Improved Random Forest Model')
    parser.add_argument('--model', type=str, default='rf_main_improved.pkl',
                        help='Path to model file')
    parser.add_argument('--data-dir', type=str, default='.',
                        help='Data directory')
    parser.add_argument('--split', type=str, default='test',
                        help='Dataset split (train or test)')
    
    args = parser.parse_args()
    
    results = evaluate_improved_model(args.model, args.data_dir, args.split)
    
    print("\n" + "="*70)
    print("Evaluation Complete!")
    print("="*70)