# app.py
#!/usr/bin/env python3
"""
Flask API for Gemstone Classification
- Accepts image uploads
- Bypasses YOLO detection (for testing)
- Uses improved RF model for classification
- Returns top 3 predictions
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import sys
import numpy as np
import pickle
from werkzeug.utils import secure_filename

# Import feature extraction
from train_hierarchical_rf import extract_all_features

# Add parent directory to path for model file
sys.path.append('..')

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
RF_MODEL_PATH = '../rf_main_improved.pkl'  # Model is in parent directory

# Create upload folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize RF model only (bypass YOLO)
print("Loading RF model...")
rf_model = None
rf_classes = None
rf_kmeans = None
rf_scaler = None
rf_pca = None
rf_use_kmeans = False

try:
    with open(RF_MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
        rf_model = model_data['model']
        rf_classes = model_data['classes']
        rf_kmeans = model_data.get('kmeans', None)
        rf_scaler = model_data.get('scaler', None)
        rf_pca = model_data.get('pca', None)
        rf_use_kmeans = model_data.get('use_kmeans', False)
    print(f"RF model loaded successfully! ({len(rf_classes)} classes)")
    print(f"Using k-means: {rf_use_kmeans}")
except Exception as e:
    print(f"Error loading RF model: {e}")
    print(f"Make sure {RF_MODEL_PATH} exists")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_top3_predictions(model, image_path, classes, kmeans_model, scaler, pca_model, use_kmeans):
    """Get top 3 predictions from RF model."""
    # Extract features
    if use_kmeans and kmeans_model is not None:
        features = extract_all_features(
            image_path, kmeans_model, scaler, pca_model,
            use_kmeans_override=use_kmeans
        )
    else:
        features = extract_all_features(image_path, None, None, None)
    
    features = features.reshape(1, -1)
    
    # Get predictions
    probabilities = model.predict_proba(features)[0]
    
    # Get top 3
    top3_indices = np.argsort(probabilities)[-3:][::-1]
    
    results = []
    for idx in top3_indices:
        results.append({
            'class': classes[idx],
            'confidence': float(probabilities[idx]),
            'percentage': float(probabilities[idx] * 100)
        })
    
    return results


@app.route('/')
def index():
    """Serve the frontend."""
    return render_template('index.html')


@app.route('/api/classify', methods=['POST'])
def classify():
    """Classify uploaded image (bypasses YOLO detection)."""
    if rf_model is None:
        return jsonify({'error': 'RF model not loaded'}), 500
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: PNG, JPG, JPEG'}), 400
    
    try:
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # BYPASS YOLO - Classify entire image directly
        print(f"Classifying image: {filename} (YOLO bypassed)")
        
        # Get top 3 predictions directly from the uploaded image
        top3 = get_top3_predictions(
            rf_model,
            filepath,
            rf_classes,
            rf_kmeans,
            rf_scaler,
            rf_pca,
            rf_use_kmeans
        )
        
        # Cleanup uploaded file
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'detections': 0,
            'predictions': top3,
            'message': 'Classification completed (YOLO bypassed)'
        })
    
    except Exception as e:
        # Cleanup on error
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        
        return jsonify({'error': f'Processing error: {str(e)}'}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'rf_model_loaded': rf_model is not None,
        'classes': len(rf_classes) if rf_classes else 0,
        'yolo_bypassed': True
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)