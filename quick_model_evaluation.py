"""
Quick Model Evaluation Script
Evaluates the trained models and shows their performance
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def evaluate_model(model_path, model_name, data_path="removed background", img_size=(224, 224)):
    """Evaluate a single model"""
    print(f"\n{'='*60}")
    print(f"EVALUATING {model_name}")
    print(f"{'='*60}")
    
    try:
        # Load the model
        model = tf.keras.models.load_model(model_path)
        print(f"[OK] Loaded {model_name} model")
        
        # Get class names
        class_names = sorted([d for d in os.listdir(data_path) 
                            if os.path.isdir(os.path.join(data_path, d))])
        
        # Create validation generator
        val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
        
        val_generator = val_datagen.flow_from_directory(
            data_path,
            target_size=img_size,
            batch_size=32,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        # Evaluate model
        print("Evaluating model on validation set...")
        val_loss, val_accuracy = model.evaluate(val_generator, verbose=0)
        
        # Get predictions
        predictions = model.predict(val_generator, verbose=0)
        y_pred = np.argmax(predictions, axis=1)
        y_true = val_generator.classes
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        print(f"\n[RESULTS] {model_name}:")
        print(f"  Validation Accuracy: {val_accuracy:.4f}")
        print(f"  Validation Loss: {val_loss:.4f}")
        print(f"  Accuracy Score: {accuracy:.4f}")
        
        return {
            'model_name': model_name,
            'validation_accuracy': val_accuracy,
            'validation_loss': val_loss,
            'accuracy_score': accuracy,
            'class_names': class_names,
            'predictions': y_pred,
            'true_labels': y_true
        }
        
    except Exception as e:
        print(f"[ERROR] Failed to evaluate {model_name}: {e}")
        return None

def main():
    """Main evaluation function"""
    print("="*80)
    print("QUICK MODEL EVALUATION")
    print("="*80)
    
    # Check if data directory exists
    if not os.path.exists("removed background"):
        print("[ERROR] 'removed background' directory not found!")
        print("Please make sure the dataset is prepared.")
        return
    
    # Define models to evaluate
    models = [
        ("vgg19_best_model.h5", "VGG19"),
        ("resnet50_best_model.h5", "ResNet50"),
        ("densenet121_best_model.h5", "DenseNet121"),
        ("mobilenet_best_model.h5", "MobileNet")
    ]
    
    results = {}
    
    # Evaluate each model
    for model_path, model_name in models:
        if os.path.exists(model_path):
            result = evaluate_model(model_path, model_name)
            if result:
                results[model_name] = result
        else:
            print(f"[WARNING] {model_path} not found!")
    
    # Create comparison summary
    if results:
        print(f"\n{'='*80}")
        print("MODEL COMPARISON SUMMARY")
        print(f"{'='*80}")
        
        # Sort by validation accuracy
        sorted_results = sorted(results.items(), key=lambda x: x[1]['validation_accuracy'], reverse=True)
        
        print(f"{'Rank':<4} {'Model':<12} {'Val Accuracy':<15} {'Val Loss':<12} {'Accuracy':<12}")
        print("-" * 70)
        
        for i, (model_name, result) in enumerate(sorted_results, 1):
            print(f"{i:<4} {model_name:<12} {result['validation_accuracy']:<15.4f} {result['validation_loss']:<12.4f} {result['accuracy_score']:<12.4f}")
        
        # Find best model
        best_model = sorted_results[0]
        print(f"\n🏆 BEST MODEL: {best_model[0]}")
        print(f"   Validation Accuracy: {best_model[1]['validation_accuracy']:.4f}")
        
        # Check for overfitting
        print(f"\n{'='*80}")
        print("OVERFITTING ANALYSIS")
        print(f"{'='*80}")
        
        for model_name, result in results.items():
            acc = result['validation_accuracy']
            if acc > 0.9:
                print(f"✅ {model_name}: Excellent performance ({acc:.4f})")
            elif acc > 0.8:
                print(f"✅ {model_name}: Good performance ({acc:.4f})")
            elif acc > 0.7:
                print(f"⚠️  {model_name}: Moderate performance ({acc:.4f})")
            else:
                print(f"❌ {model_name}: Poor performance ({acc:.4f})")
        
        print(f"\n{'='*80}")
        print("EVALUATION COMPLETED!")
        print(f"{'='*80}")
        
    else:
        print("[ERROR] No models could be evaluated!")

if __name__ == "__main__":
    main()
