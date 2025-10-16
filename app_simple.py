"""
Simplified Flask Backend API for Medicinal Leaf Classification
Loads TensorFlow only when needed to avoid NumPy compatibility issues
"""

import os
import base64
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import json
from datetime import datetime
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Global variables for models
models = {}
class_names = []

# Model configuration
MODEL_CONFIG = {
    'vgg19': {
        'path': 'vgg19_best_model.h5',
        'name': 'VGG19',
        'description': 'VGG19 - Deep CNN with 19 layers'
    },
    'resnet50': {
        'path': 'resnet50_best_model.h5', 
        'name': 'ResNet50',
        'description': 'ResNet50 - Residual Network with 50 layers'
    },
    'densenet121': {
        'path': 'densenet121_best_model.h5',
        'name': 'DenseNet121', 
        'description': 'DenseNet121 - Densely Connected Network'
    },
    'mobilenet': {
        'path': 'mobilenet_best_model.h5',
        'name': 'MobileNet',
        'description': 'MobileNet - Lightweight CNN for mobile devices'
    }
}

def load_class_names():
    """Load class names from the dataset directory"""
    global class_names
    data_path = "removed background"
    if os.path.exists(data_path):
        class_names = sorted([d for d in os.listdir(data_path) 
                             if os.path.isdir(os.path.join(data_path, d))])
        print(f"Loaded {len(class_names)} classes: {class_names}")
    else:
        # Fallback class names based on the project structure
        class_names = [
            'Aloevera', 'Amla', 'Bamboo', 'Bhrami', 'Bringaraja', 'Castor',
            'Coffee', 'Coriender', 'Curry', 'Eucalyptus', 'Ginger', 'Guava',
            'Henna', 'Hibiscus', 'Lemon', 'Mint', 'Neem', 'Onion',
            'Palak(Spinach)', 'Papaya'
        ]
        print(f"Using fallback class names: {len(class_names)} classes")
    return class_names

def load_model(model_key):
    """Load a specific model - only import TensorFlow when needed"""
    try:
        if model_key in models:
            return models[model_key]
            
        model_config = MODEL_CONFIG[model_key]
        model_path = model_config['path']
        
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return None
            
        print(f"Loading {model_config['name']} model...")
        
        # Import TensorFlow only when needed
        import tensorflow as tf
        import numpy as np
        
        # Suppress TensorFlow warnings
        tf.get_logger().setLevel('ERROR')
        
        model = tf.keras.models.load_model(model_path)
        models[model_key] = model
        print(f"Successfully loaded {model_config['name']} model")
        return model
        
    except Exception as e:
        print(f"Error loading {model_key} model: {str(e)}")
        return None

def preprocess_image(image_data, target_size=(224, 224)):
    """Preprocess image for model inference"""
    try:
        # Convert base64 to PIL Image
        if isinstance(image_data, str):
            # Remove data URL prefix if present
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
        else:
            image_bytes = image_data
            
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Resize image
        image = image.resize(target_size)
        
        # Convert to numpy array and normalize
        import numpy as np
        img_array = np.array(image) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
        
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        return None

def predict_with_model(model, image_array, model_name):
    """Make prediction with a specific model"""
    try:
        import numpy as np
        
        start_time = datetime.now()
        predictions = model.predict(image_array, verbose=0)
        inference_time = (datetime.now() - start_time).total_seconds()
        
        # Get top 5 predictions
        top_indices = np.argsort(predictions[0])[-5:][::-1]
        top_predictions = []
        
        for idx in top_indices:
            class_name = class_names[idx]
            confidence = float(predictions[0][idx])
            top_predictions.append({
                'class': class_name,
                'confidence': confidence,
                'percentage': round(confidence * 100, 2)
            })
        
        return {
            'model_name': model_name,
            'predictions': top_predictions,
            'inference_time': round(inference_time, 3),
            'success': True
        }
        
    except Exception as e:
        print(f"Error making prediction with {model_name}: {str(e)}")
        return {
            'model_name': model_name,
            'error': str(e),
            'success': False
        }

@app.route('/')
def home():
    """Home endpoint"""
    return jsonify({
        'message': 'Medicinal Leaf Classification API',
        'version': '1.0.0',
        'endpoints': {
            'classify': '/classify',
            'models': '/models',
            'health': '/health'
        }
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    loaded_models = list(models.keys())
    return jsonify({
        'status': 'healthy',
        'loaded_models': loaded_models,
        'total_classes': len(class_names),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/models')
def get_models():
    """Get available models"""
    available_models = []
    for key, config in MODEL_CONFIG.items():
        model_info = {
            'key': key,
            'name': config['name'],
            'description': config['description'],
            'loaded': key in models,
            'path': config['path']
        }
        available_models.append(model_info)
    
    return jsonify({
        'models': available_models,
        'total_models': len(available_models),
        'loaded_models': list(models.keys())
    })

@app.route('/classify', methods=['POST'])
def classify():
    """Classify leaf image"""
    try:
        # Get request data
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
            
        image_data = data['image']
        model_key = data.get('model', 'densenet121')  # Default to DenseNet121
        
        # Validate model key
        if model_key not in MODEL_CONFIG:
            return jsonify({'error': f'Invalid model: {model_key}'}), 400
            
        # Load model if not already loaded
        model = load_model(model_key)
        if model is None:
            return jsonify({'error': f'Failed to load model: {model_key}'}), 500
            
        # Preprocess image
        image_array = preprocess_image(image_data)
        if image_array is None:
            return jsonify({'error': 'Failed to preprocess image'}), 400
            
        # Make prediction
        result = predict_with_model(model, image_array, MODEL_CONFIG[model_key]['name'])
        
        if not result['success']:
            return jsonify({'error': result.get('error', 'Prediction failed')}), 500
            
        # Add metadata
        result['timestamp'] = datetime.now().isoformat()
        result['image_size'] = image_array.shape[1:3]
        result['total_classes'] = len(class_names)
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Classification error: {str(e)}")
        return jsonify({'error': f'Classification failed: {str(e)}'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

def initialize_app():
    """Initialize the Flask application"""
    print("Initializing Medicinal Leaf Classification API...")
    
    # Load class names
    load_class_names()
    
    print("API initialization complete!")
    print(f"Available models: {list(MODEL_CONFIG.keys())}")
    print(f"Total classes: {len(class_names)}")

if __name__ == '__main__':
    initialize_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
