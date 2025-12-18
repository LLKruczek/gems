#!/usr/bin/env python3
"""
Gemstone Detection and Classification Pipeline

This script implements a two-stage pipeline:
1. YOLO: Detect and localize gemstones in images (get bounding boxes)
2. Classification Model: Classify the cropped gemstone region

This is useful for real-world applications where images may have:
- Complex backgrounds
- Multiple gemstones
- Gemstones not centered
- Need automatic cropping before classification
"""

import torch
import cv2
import numpy as np
from PIL import Image
import os
import argparse
from pathlib import Path

# Try to import YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not available. Install with: pip install ultralytics")

# Classification model imports
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


class GemstoneDetectorClassifier:
    """Two-stage pipeline: YOLO detection + classification model."""
    
    def __init__(self, yolo_model_path, classifier_model_path, classifier_type='vit', device='cpu'):
        """
        Initialize the pipeline.
        
        Args:
            yolo_model_path: Path to trained YOLO model (.pt file)
            classifier_model_path: Path to classification model
            classifier_type: 'vit', 'resnet', or 'rf' (random forest)
            device: 'cpu' or 'cuda'
        """
        self.device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        self.classifier_type = classifier_type
        
        # Load YOLO detector
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics not installed. Install with: pip install ultralytics")
        
        print(f"Loading YOLO detector from {yolo_model_path}...")
        self.yolo_model = YOLO(yolo_model_path)
        self.yolo_model.to(self.device)
        print("YOLO detector loaded!")
        
        # Load classification model
        print(f"Loading {classifier_type} classifier from {classifier_model_path}...")
        self.classifier = self._load_classifier(classifier_model_path, classifier_type)
        print("Classifier loaded!")
    
    def _load_classifier(self, model_path, model_type):
        """Load the classification model."""
        if model_type == 'vit':
            return self._load_vit_model(model_path)
        elif model_type == 'resnet':
            return self._load_resnet_model(model_path)
        elif model_type == 'rf':
            return self._load_rf_model(model_path)
        else:
            raise ValueError(f"Unknown classifier type: {model_type}")
    
    def _load_vit_model(self, model_path):
        """Load ViT model."""
        # This is a simplified version - adjust based on your actual model
        import torch.nn as nn
        from torchvision import transforms
        
        if TIMM_AVAILABLE:
            model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=68)
        else:
            from torchvision import models
            model = models.resnet50(weights=None)
            model.fc = nn.Linear(model.fc.in_features, 68)
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        model.to(self.device)
        
        # Get transforms
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        return model
    
    def _load_resnet_model(self, model_path):
        """Load ResNet model."""
        from torchvision import models
        import torch.nn as nn
        from torchvision import transforms
        
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 68)
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Clean state dict
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                cleaned_state_dict[k[6:]] = v
            else:
                cleaned_state_dict[k] = v
        
        model.load_state_dict(cleaned_state_dict, strict=False)
        model.eval()
        model.to(self.device)
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        return model
    
    def _load_rf_model(self, model_path):
        """Load Random Forest model."""
        import pickle
        from train_hierarchical_rf import extract_all_features
        
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            model = data['model']
            self.rf_classes = data['classes']
            self.rf_kmeans = data.get('kmeans', None)
            self.rf_scaler = data.get('scaler', None)
            self.rf_pca = data.get('pca', None)
            self.rf_use_kmeans = data.get('use_kmeans', False)
        
        # Store feature extraction function
        self.extract_features = extract_all_features
        
        return model
    
    def detect_gemstones(self, image_path, conf_threshold=0.25):
        """
        Detect gemstones in image using YOLO.
        
        Returns:
            List of bounding boxes: [(x1, y1, x2, y2, confidence, class_id), ...]
        """
        results = self.yolo_model(image_path, conf=conf_threshold)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                detections.append((int(x1), int(y1), int(x2), int(y2), float(conf), cls))
        
        return detections
    
    def crop_gemstone(self, image, bbox, padding=10):
        """
        Crop gemstone region from image.
        
        Args:
            image: PIL Image or numpy array
            bbox: (x1, y1, x2, y2)
            padding: Additional padding around bounding box
        """
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        
        x1, y1, x2, y2 = bbox[:4]
        h, w = img_array.shape[:2]
        
        # Add padding
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        # Crop
        cropped = img_array[y1:y2, x1:x2]
        
        return Image.fromarray(cropped)
    
    def classify_gemstone(self, cropped_image):
        """
        Classify cropped gemstone image.
        
        Args:
            cropped_image: PIL Image of cropped gemstone
        
        Returns:
            dict with prediction and confidence
        """
        if self.classifier_type in ['vit', 'resnet']:
            # Deep learning model
            image_tensor = self.transform(cropped_image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.classifier(image_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                top_prob, top_idx = torch.topk(probs, 1)
            
            # Get class name (you'll need to load this from your model)
            # For now, return index
            return {
                'class_idx': int(top_idx[0][0].item()),
                'confidence': float(top_prob[0][0].item())
            }
        
        elif self.classifier_type == 'rf':
            # Random Forest model
            # Save image temporarily for feature extraction
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                cropped_image.save(tmp.name)
                tmp_path = tmp.name
            
            try:
                if self.rf_use_kmeans and self.rf_kmeans is not None:
                    features = self.extract_features(tmp_path, self.rf_kmeans, self.rf_scaler, self.rf_pca)
                else:
                    features = self.extract_features(tmp_path, None, None, None)
                
                features = features.reshape(1, -1)
                pred = self.classifier.predict(features)[0]
                proba = self.classifier.predict_proba(features)[0]
                
                return {
                    'class_idx': int(pred),
                    'class_name': self.rf_classes[pred],
                    'confidence': float(proba[pred])
                }
            finally:
                os.unlink(tmp_path)
    
    def process_image(self, image_path, conf_threshold=0.25, return_crops=False):
        """
        Complete pipeline: detect and classify gemstones.
        
        Args:
            image_path: Path to input image
            conf_threshold: YOLO confidence threshold
            return_crops: If True, also return cropped images
        
        Returns:
            List of results: [{'bbox': ..., 'detection_conf': ..., 'classification': ...}, ...]
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        image_array = np.array(image)
        
        # Detect gemstones
        detections = self.detect_gemstones(image_path, conf_threshold)
        
        if len(detections) == 0:
            print("No gemstones detected in image")
            return []
        
        results = []
        for bbox in detections:
            x1, y1, x2, y2, det_conf, det_cls = bbox
            
            # Crop gemstone
            cropped = self.crop_gemstone(image, (x1, y1, x2, y2))
            
            # Classify
            classification = self.classify_gemstone(cropped)
            
            result = {
                'bbox': (x1, y1, x2, y2),
                'detection_confidence': det_conf,
                'detection_class': det_cls,
                'classification': classification
            }
            
            if return_crops:
                result['cropped_image'] = cropped
            
            results.append(result)
        
        return results


def main():
    """Example usage."""
    parser = argparse.ArgumentParser(description='Detect and Classify Gemstones')
    parser.add_argument('--yolo-model', type=str, required=True,
                        help='Path to YOLO detection model')
    parser.add_argument('--classifier-model', type=str, required=True,
                        help='Path to classification model')
    parser.add_argument('--classifier-type', type=str, default='vit',
                        choices=['vit', 'resnet', 'rf'],
                        help='Type of classifier model')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--conf-threshold', type=float, default=0.25,
                        help='YOLO confidence threshold')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu or cuda)')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = GemstoneDetectorClassifier(
        args.yolo_model,
        args.classifier_model,
        args.classifier_type,
        args.device
    )
    
    # Process image
    results = pipeline.process_image(args.image, args.conf_threshold)
    
    print("\n" + "="*70)
    print("Detection and Classification Results")
    print("="*70)
    
    for i, result in enumerate(results, 1):
        print(f"\nGemstone {i}:")
        print(f"  Bounding Box: {result['bbox']}")
        print(f"  Detection Confidence: {result['detection_confidence']:.4f}")
        print(f"  Classification:")
        if 'class_name' in result['classification']:
            print(f"    Class: {result['classification']['class_name']}")
        else:
            print(f"    Class Index: {result['classification']['class_idx']}")
        print(f"    Confidence: {result['classification']['confidence']:.4f}")


if __name__ == '__main__':
    main()

