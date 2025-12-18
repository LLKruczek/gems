# backend_purple.py
#!/usr/bin/env python3
"""
Flask API for Gemstone Classification using Purple Stones Specialized RF Model
- Accepts image uploads
- Bypasses YOLO detection (for testing)
- Uses specialized RF model for purple stones (Amethyst, Sapphire Purple, Ametrine)
- Returns top 3 predictions
- Runs on port 5002
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
PURPLE_MODEL_PATH = '../rf_purple_model.pkl'  # Purple model is in parent directory

# Purple stone classes (subset of 68 classes)
PURPLE_STONES = [
    'Amethyst',
    'Sapphire Purple',
    'Ametrine'
]

# Create upload folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize Purple RF model only (bypass YOLO)
print("Loading Purple RF model...")
purple_model = None
purple_classes = None
purple_kmeans = None
purple_scaler = None
purple_pca = None
purple_use_kmeans = False

try:
    with open(PURPLE_MODEL_PATH, 'rb') as f:
        model_data = pickle.load(f)
        purple_model = model_data['model']
        purple_classes = model_data['classes']
        purple_kmeans = model_data.get('kmeans', None)
        purple_scaler = model_data.get('scaler', None)
        purple_pca = model_data.get('pca', None)
        purple_use_kmeans = model_data.get('use_kmeans', False)
    print(f"Purple RF model loaded successfully! ({len(purple_classes)} classes)")
    print(f"Classes: {purple_classes}")
    print(f"Using k-means: {purple_use_kmeans}")
except Exception as e:
    print(f"Error loading Purple RF model: {e}")
    print(f"Make sure {PURPLE_MODEL_PATH} exists")
    import traceback
    traceback.print_exc()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_top3_predictions(model, image_path, classes, kmeans_model, scaler, pca_model, use_kmeans):
    """Get top 3 predictions from Purple RF model."""
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
    
    # Get top 3 (or fewer if model has fewer classes)
    top_k = min(3, len(probabilities))
    top_indices = np.argsort(probabilities)[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
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


@app.route('/api/classify', methods=['GET', 'POST'])
def classify():
    """Classify uploaded image using Purple RF model (bypasses YOLO detection)."""
    # Handle GET requests (for testing or direct browser access)
    if request.method == 'GET':
        return jsonify({
            'message': 'Gemstone Classification API - Purple Stones Specialized Model',
            'method': 'POST',
            'endpoint': '/api/classify',
            'description': 'Upload an image file to get top 3 gemstone predictions (purple stones only)',
            'model': 'Random Forest - Purple Stones Specialized',
            'classes': purple_classes if purple_classes else PURPLE_STONES,
            'status': 'ready' if purple_model is not None else 'model_not_loaded',
            'example': 'Use POST with multipart/form-data containing an "image" field'
        }), 200
    
    # Handle POST requests
    if purple_model is None:
        return jsonify({'error': 'Purple RF model not loaded'}), 500
    
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
        
        # BYPASS YOLO - Classify entire image directly with Purple model
        print(f"Classifying image: {filename} (Purple RF model, YOLO bypassed)")
        
        # Get top 3 predictions directly from the uploaded image
        top3 = get_top3_predictions(
            purple_model,
            filepath,
            purple_classes,
            purple_kmeans,
            purple_scaler,
            purple_pca,
            purple_use_kmeans
        )
        
        # Cleanup uploaded file
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'detections': 0,
            'predictions': top3,
            'message': 'Classification completed (Purple RF model, YOLO bypassed)',
            'model_type': 'Purple Stones Specialized'
        })
    
    except Exception as e:
        # Cleanup on error
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error: {error_trace}")
        return jsonify({'error': f'Processing error: {str(e)}'}), 500


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'purple_model_loaded': purple_model is not None,
        'classes': purple_classes if purple_classes else [],
        'num_classes': len(purple_classes) if purple_classes else 0,
        'expected_classes': PURPLE_STONES,
        'using_kmeans': purple_use_kmeans,
        'yolo_bypassed': True,
        'model_type': 'Random Forest - Purple Stones Specialized'
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)

