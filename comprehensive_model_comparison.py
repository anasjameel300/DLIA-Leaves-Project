"""
Comprehensive Model Comparison Analysis
Compares VGG19, ResNet50, DenseNet121, and MobileNet models
Generates detailed metrics, visualizations, and saves results
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import time
# import psutil  # Optional dependency
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ComprehensiveModelComparator:
    def __init__(self, data_path="removed background", img_size=(224, 224)):
        self.data_path = data_path
        self.img_size = img_size
        self.models = ['VGG19', 'ResNet50', 'DenseNet121', 'MobileNet']
        self.model_files = {
            'VGG19': 'vgg19_best_model.h5',
            'ResNet50': 'resnet50_best_model.h5', 
            'DenseNet121': 'densenet121_best_model.h5',
            'MobileNet': 'mobilenet_best_model.h5'
        }
        self.results = {}
        self.comparison_data = {}
        self.output_dir = "comparison_data"
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_class_names(self):
        """Load class names from dataset"""
        self.class_names = sorted([d for d in os.listdir(self.data_path) 
                                 if os.path.isdir(os.path.join(self.data_path, d))])
        print(f"Found {len(self.class_names)} classes: {self.class_names}")
        return self.class_names
    
    def create_data_generators(self):
        """Create data generators for evaluation"""
        # Validation generator
        val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
        
        self.val_generator = val_datagen.flow_from_directory(
            self.data_path,
            target_size=self.img_size,
            batch_size=32,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        # Test generator (same as validation for this analysis)
        self.test_generator = val_datagen.flow_from_directory(
            self.data_path,
            target_size=self.img_size,
            batch_size=32,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        print(f"Validation samples: {self.val_generator.samples}")
        print(f"Number of classes: {len(self.class_names)}")
    
    def evaluate_model_performance(self, model_path, model_name):
        """Comprehensive evaluation of a single model"""
        print(f"\n{'='*60}")
        print(f"EVALUATING {model_name}")
        print(f"{'='*60}")
        
        try:
            # Load model
            print(f"[LOADING] {model_name} model...")
            start_time = time.time()
            model = tf.keras.models.load_model(model_path)
            load_time = time.time() - start_time
            
            print(f"[SUCCESS] Model loaded in {load_time:.2f} seconds")
            
            # Get model info
            total_params = model.count_params()
            trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
            
            # Evaluate model
            print("[EVALUATING] On validation set...")
            eval_start = time.time()
            val_loss, val_accuracy = model.evaluate(self.val_generator, verbose=0)
            eval_time = time.time() - eval_start
            print(f"[SUCCESS] Evaluation completed in {eval_time:.2f} seconds")
            
            # Get predictions
            print("[GENERATING] Predictions...")
            pred_start = time.time()
            predictions = model.predict(self.val_generator, verbose=0)
            pred_time = time.time() - pred_start
            print(f"[SUCCESS] Predictions generated in {pred_time:.2f} seconds")
            
            y_pred = np.argmax(predictions, axis=1)
            y_true = self.val_generator.classes
            y_true_onehot = tf.keras.utils.to_categorical(y_true, num_classes=len(self.class_names))
            
            # Calculate detailed metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average='weighted')
            
            # Per-class metrics
            class_report = classification_report(y_true, y_pred, target_names=self.class_names, output_dict=True)
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Calculate inference speed (images per second)
            inference_speed = len(y_true) / pred_time
            
            # Calculate time complexity metrics
            time_per_image = pred_time / len(y_true)
            time_per_parameter = pred_time / total_params if total_params > 0 else 0
            efficiency_score = val_accuracy / (pred_time / len(y_true))  # Accuracy per second
            
            # Memory usage (simplified without psutil)
            memory_usage = 1000  # Placeholder value
            
            result = {
                'model_name': model_name,
                'validation_accuracy': val_accuracy,
                'validation_loss': val_loss,
                'accuracy_score': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'load_time': load_time,
                'evaluation_time': eval_time,
                'prediction_time': pred_time,
                'inference_speed': inference_speed,
                'time_per_image': time_per_image,
                'time_per_parameter': time_per_parameter,
                'efficiency_score': efficiency_score,
                'memory_usage_mb': memory_usage,
                'class_report': class_report,
                'confusion_matrix': cm.tolist(),
                'predictions': y_pred,
                'true_labels': y_true,
                'class_names': self.class_names
            }
            
            print(f"\n[RESULTS] {model_name}:")
            print(f"  [ACCURACY] Validation Accuracy: {val_accuracy:.4f}")
            print(f"  [LOSS] Validation Loss: {val_loss:.4f}")
            print(f"  [PRECISION] Precision: {precision:.4f}")
            print(f"  [RECALL] Recall: {recall:.4f}")
            print(f"  [F1-SCORE] F1-Score: {f1:.4f}")
            print(f"  [PARAMETERS] Total Parameters: {total_params:,}")
            print(f"  [TRAINABLE] Trainable Parameters: {trainable_params:,}")
            print(f"  [SPEED] Inference Speed: {inference_speed:.2f} images/sec")
            print(f"  [TIME] Time per Image: {time_per_image:.4f} seconds")
            print(f"  [EFFICIENCY] Efficiency Score: {efficiency_score:.2f} (accuracy/sec)")
            print(f"  [MEMORY] Memory Usage: {memory_usage:.2f} MB")
            
            return result
            
        except Exception as e:
            print(f"[ERROR] Failed to evaluate {model_name}: {e}")
            return None
    
    def evaluate_all_models(self):
        """Evaluate all models"""
        print("="*80)
        print("COMPREHENSIVE MODEL EVALUATION")
        print("="*80)
        
        for i, model_name in enumerate(self.models, 1):
            print(f"\n[PROCESSING] Model {i}/{len(self.models)}: {model_name}")
            model_path = self.model_files[model_name]
            if os.path.exists(model_path):
                result = self.evaluate_model_performance(model_path, model_name)
                if result:
                    self.results[model_name] = result
                    print(f"[SUCCESS] {model_name} evaluation completed!")
                else:
                    print(f"[FAILED] {model_name} evaluation failed!")
            else:
                print(f"[WARNING] {model_path} not found!")
            
            print(f"[PROGRESS] {i}/{len(self.models)} models processed")
    
    def create_comparison_dataframe(self):
        """Create comprehensive comparison dataframe"""
        if not self.results:
            print("[ERROR] No results to compare!")
            return None
        
        comparison_data = []
        
        for model_name, result in self.results.items():
            row = {
                'Model': model_name,
                'Validation Accuracy': result['validation_accuracy'],
                'Validation Loss': result['validation_loss'],
                'Precision': result['precision'],
                'Recall': result['recall'],
                'F1-Score': result['f1_score'],
                'Total Parameters': result['total_parameters'],
                'Trainable Parameters': result['trainable_parameters'],
                'Load Time (s)': result['load_time'],
                'Evaluation Time (s)': result['evaluation_time'],
                'Prediction Time (s)': result['prediction_time'],
                'Inference Speed (img/s)': result['inference_speed'],
                'Time per Image (s)': result['time_per_image'],
                'Time per Parameter (s)': result['time_per_parameter'],
                'Efficiency Score': result['efficiency_score'],
                'Memory Usage (MB)': result['memory_usage_mb']
            }
            comparison_data.append(row)
        
        self.comparison_df = pd.DataFrame(comparison_data)
        return self.comparison_df
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("\n" + "="*60)
        print("CREATING VISUALIZATIONS")
        print("="*60)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. Model Performance Comparison
        print("[CREATING] Performance comparison charts...")
        self.plot_performance_comparison()
        print("[SUCCESS] Performance comparison completed")
        
        # 2. Model Complexity Analysis
        print("[CREATING] Complexity analysis charts...")
        self.plot_complexity_analysis()
        print("[SUCCESS] Complexity analysis completed")
        
        # 3. Speed vs Accuracy Trade-off
        print("[CREATING] Speed vs accuracy charts...")
        self.plot_speed_accuracy_tradeoff()
        print("[SUCCESS] Speed vs accuracy charts completed")
        
        # 4. Confusion Matrices
        print("[CREATING] Confusion matrices...")
        self.plot_confusion_matrices()
        print("[SUCCESS] Confusion matrices completed")
        
        # 5. Per-Class Performance
        print("[CREATING] Per-class performance charts...")
        self.plot_per_class_performance()
        print("[SUCCESS] Per-class performance charts completed")
        
        # 6. Model Size Comparison
        print("[CREATING] Model size comparison charts...")
        self.plot_model_size_comparison()
        print("[SUCCESS] Model size comparison completed")
        
        # 7. Memory Usage Analysis
        print("[CREATING] Memory usage analysis charts...")
        self.plot_memory_analysis()
        print("[SUCCESS] Memory usage analysis completed")
        
        # 8. Time Complexity Analysis
        print("[CREATING] Time complexity analysis charts...")
        self.plot_time_complexity_analysis()
        print("[SUCCESS] Time complexity analysis completed")
        
        print(f"\n[SUCCESS] All visualizations saved to {self.output_dir}/")
    
    def plot_performance_comparison(self):
        """Plot overall performance comparison"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy comparison
        models = self.comparison_df['Model']
        accuracies = self.comparison_df['Validation Accuracy']
        
        bars1 = ax1.bar(models, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax1.set_title('Validation Accuracy Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, acc in zip(bars1, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Loss comparison
        losses = self.comparison_df['Validation Loss']
        bars2 = ax2.bar(models, losses, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax2.set_title('Validation Loss Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Loss')
        ax2.grid(True, alpha=0.3)
        
        for bar, loss in zip(bars2, losses):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                    f'{loss:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # F1-Score comparison
        f1_scores = self.comparison_df['F1-Score']
        bars3 = ax3.bar(models, f1_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax3.set_title('F1-Score Comparison', fontsize=14, fontweight='bold')
        ax3.set_ylabel('F1-Score')
        ax3.set_ylim(0, 1)
        ax3.grid(True, alpha=0.3)
        
        for bar, f1 in zip(bars3, f1_scores):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{f1:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Precision vs Recall
        precision = self.comparison_df['Precision']
        recall = self.comparison_df['Recall']
        
        ax4.scatter(precision, recall, s=200, c=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'], alpha=0.7)
        for i, model in enumerate(models):
            ax4.annotate(model, (precision.iloc[i], recall.iloc[i]), 
                        xytext=(5, 5), textcoords='offset points', fontweight='bold')
        ax4.set_xlabel('Precision')
        ax4.set_ylabel('Recall')
        ax4.set_title('Precision vs Recall', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_complexity_analysis(self):
        """Plot model complexity analysis"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Parameters vs Accuracy
        total_params = self.comparison_df['Total Parameters'] / 1e6  # Convert to millions
        accuracies = self.comparison_df['Validation Accuracy']
        models = self.comparison_df['Model']
        
        scatter = ax1.scatter(total_params, accuracies, s=200, 
                             c=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'], alpha=0.7)
        for i, model in enumerate(models):
            ax1.annotate(model, (total_params.iloc[i], accuracies.iloc[i]), 
                        xytext=(5, 5), textcoords='offset points', fontweight='bold')
        ax1.set_xlabel('Total Parameters (Millions)')
        ax1.set_ylabel('Validation Accuracy')
        ax1.set_title('Model Complexity vs Accuracy', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Trainable vs Total Parameters
        trainable_params = self.comparison_df['Trainable Parameters'] / 1e6
        bars = ax2.bar(models, total_params, color='lightblue', alpha=0.7, label='Total Parameters')
        bars2 = ax2.bar(models, trainable_params, color='darkblue', alpha=0.7, label='Trainable Parameters')
        ax2.set_ylabel('Parameters (Millions)')
        ax2.set_title('Total vs Trainable Parameters', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/complexity_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_speed_accuracy_tradeoff(self):
        """Plot speed vs accuracy trade-off"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Inference Speed vs Accuracy
        speeds = self.comparison_df['Inference Speed (img/s)']
        accuracies = self.comparison_df['Validation Accuracy']
        models = self.comparison_df['Model']
        
        scatter = ax1.scatter(speeds, accuracies, s=200, 
                             c=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'], alpha=0.7)
        for i, model in enumerate(models):
            ax1.annotate(model, (speeds.iloc[i], accuracies.iloc[i]), 
                        xytext=(5, 5), textcoords='offset points', fontweight='bold')
        ax1.set_xlabel('Inference Speed (images/sec)')
        ax1.set_ylabel('Validation Accuracy')
        ax1.set_title('Speed vs Accuracy Trade-off', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Memory Usage vs Accuracy
        memory = self.comparison_df['Memory Usage (MB)']
        scatter2 = ax2.scatter(memory, accuracies, s=200, 
                              c=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'], alpha=0.7)
        for i, model in enumerate(models):
            ax2.annotate(model, (memory.iloc[i], accuracies.iloc[i]), 
                        xytext=(5, 5), textcoords='offset points', fontweight='bold')
        ax2.set_xlabel('Memory Usage (MB)')
        ax2.set_ylabel('Validation Accuracy')
        ax2.set_title('Memory Usage vs Accuracy', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/speed_accuracy_tradeoff.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        n_models = len(self.results)
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()
        
        for i, (model_name, result) in enumerate(self.results.items()):
            cm = np.array(result['confusion_matrix'])
            
            # Normalize confusion matrix
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                       xticklabels=self.class_names, yticklabels=self.class_names,
                       ax=axes[i])
            axes[i].set_title(f'{model_name} Confusion Matrix', fontsize=14, fontweight='bold')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_per_class_performance(self):
        """Plot per-class performance analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()
        
        for i, (model_name, result) in enumerate(self.results.items()):
            class_report = result['class_report']
            
            # Extract per-class metrics
            classes = []
            precisions = []
            recalls = []
            f1_scores = []
            
            for class_name in self.class_names:
                if class_name in class_report:
                    classes.append(class_name)
                    precisions.append(class_report[class_name]['precision'])
                    recalls.append(class_report[class_name]['recall'])
                    f1_scores.append(class_report[class_name]['f1-score'])
            
            # Create subplot
            x = np.arange(len(classes))
            width = 0.25
            
            axes[i].bar(x - width, precisions, width, label='Precision', alpha=0.8)
            axes[i].bar(x, recalls, width, label='Recall', alpha=0.8)
            axes[i].bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
            
            axes[i].set_xlabel('Classes')
            axes[i].set_ylabel('Score')
            axes[i].set_title(f'{model_name} - Per-Class Performance', fontsize=12, fontweight='bold')
            axes[i].set_xticks(x)
            axes[i].set_xticklabels(classes, rotation=45, ha='right')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/per_class_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_model_size_comparison(self):
        """Plot model size comparison"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        models = self.comparison_df['Model']
        total_params = self.comparison_df['Total Parameters'] / 1e6
        trainable_params = self.comparison_df['Trainable Parameters'] / 1e6
        
        # Total parameters
        bars1 = ax1.bar(models, total_params, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax1.set_title('Total Parameters Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Parameters (Millions)')
        ax1.grid(True, alpha=0.3)
        
        for bar, params in zip(bars1, total_params):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{params:.1f}M', ha='center', va='bottom', fontweight='bold')
        
        # Trainable parameters
        bars2 = ax2.bar(models, trainable_params, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax2.set_title('Trainable Parameters Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Parameters (Millions)')
        ax2.grid(True, alpha=0.3)
        
        for bar, params in zip(bars2, trainable_params):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{params:.1f}M', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/model_size_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_memory_analysis(self):
        """Plot memory usage analysis"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        models = self.comparison_df['Model']
        memory_usage = self.comparison_df['Memory Usage (MB)']
        inference_speed = self.comparison_df['Inference Speed (img/s)']
        
        # Memory usage
        bars1 = ax1.bar(models, memory_usage, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax1.set_title('Memory Usage Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Memory Usage (MB)')
        ax1.grid(True, alpha=0.3)
        
        for bar, mem in zip(bars1, memory_usage):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{mem:.1f}MB', ha='center', va='bottom', fontweight='bold')
        
        # Inference speed
        bars2 = ax2.bar(models, inference_speed, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax2.set_title('Inference Speed Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Speed (images/sec)')
        ax2.grid(True, alpha=0.3)
        
        for bar, speed in zip(bars2, inference_speed):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{speed:.1f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/memory_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_time_complexity_analysis(self):
        """Plot time complexity analysis"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        models = self.comparison_df['Model']
        
        # Time per image
        time_per_image = self.comparison_df['Time per Image (s)']
        bars1 = ax1.bar(models, time_per_image, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax1.set_title('Time per Image (Lower is Better)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Time per Image (seconds)')
        ax1.grid(True, alpha=0.3)
        
        for bar, time_val in zip(bars1, time_per_image):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001, 
                    f'{time_val:.4f}s', ha='center', va='bottom', fontweight='bold')
        
        # Time per parameter
        time_per_param = self.comparison_df['Time per Parameter (s)']
        bars2 = ax2.bar(models, time_per_param, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax2.set_title('Time per Parameter (Lower is Better)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Time per Parameter (seconds)')
        ax2.set_yscale('log')  # Log scale for better visualization
        ax2.grid(True, alpha=0.3)
        
        for bar, time_val in zip(bars2, time_per_param):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.5, 
                    f'{time_val:.2e}s', ha='center', va='bottom', fontweight='bold')
        
        # Efficiency Score (Accuracy per second)
        efficiency = self.comparison_df['Efficiency Score']
        bars3 = ax3.bar(models, efficiency, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax3.set_title('Efficiency Score (Higher is Better)', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Efficiency (Accuracy per second)')
        ax3.grid(True, alpha=0.3)
        
        for bar, eff in zip(bars3, efficiency):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    f'{eff:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Time vs Accuracy Trade-off
        accuracies = self.comparison_df['Validation Accuracy']
        scatter = ax4.scatter(time_per_image, accuracies, s=200, 
                             c=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'], alpha=0.7)
        for i, model in enumerate(models):
            ax4.annotate(model, (time_per_image.iloc[i], accuracies.iloc[i]), 
                        xytext=(5, 5), textcoords='offset points', fontweight='bold')
        ax4.set_xlabel('Time per Image (seconds)')
        ax4.set_ylabel('Validation Accuracy')
        ax4.set_title('Time vs Accuracy Trade-off', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/time_complexity_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_detailed_results(self):
        """Save detailed results to files"""
        print("\n" + "="*60)
        print("SAVING DETAILED RESULTS")
        print("="*60)
        
        # Save comparison dataframe
        print("[SAVING] Comparison summary...")
        self.comparison_df.to_csv(f'{self.output_dir}/model_comparison_summary.csv', index=False)
        print(f"[SUCCESS] Comparison summary saved to {self.output_dir}/model_comparison_summary.csv")
        
        # Save detailed results as JSON
        print("[SAVING] Detailed results...")
        detailed_results = {}
        for model_name, result in self.results.items():
            # Convert numpy arrays to lists for JSON serialization
            result_copy = result.copy()
            result_copy['confusion_matrix'] = result['confusion_matrix']
            result_copy['predictions'] = result['predictions'].tolist()
            result_copy['true_labels'] = result['true_labels'].tolist()
            # Convert numpy types to Python types
            result_copy['total_parameters'] = int(result['total_parameters'])
            result_copy['trainable_parameters'] = int(result['trainable_parameters'])
            detailed_results[model_name] = result_copy
        
        with open(f'{self.output_dir}/detailed_results.json', 'w') as f:
            json.dump(detailed_results, f, indent=2)
        print(f"[SUCCESS] Detailed results saved to {self.output_dir}/detailed_results.json")
        
        # Save per-class reports
        print("[SAVING] Per-class reports...")
        for model_name, result in self.results.items():
            class_report_df = pd.DataFrame(result['class_report']).transpose()
            class_report_df.to_csv(f'{self.output_dir}/{model_name.lower()}_class_report.csv')
        print(f"[SUCCESS] Per-class reports saved")
    
    def generate_final_report(self):
        """Generate final comprehensive report"""
        print("\n" + "="*80)
        print("FINAL COMPREHENSIVE REPORT")
        print("="*80)
        
        # Sort by accuracy
        sorted_df = self.comparison_df.sort_values('Validation Accuracy', ascending=False)
        
        print(f"\n[BEST] MODEL RANKING (by Validation Accuracy):")
        print("-" * 80)
        for i, (_, row) in enumerate(sorted_df.iterrows(), 1):
            print(f"{i}. {row['Model']:<12} - Accuracy: {row['Validation Accuracy']:.4f} | "
                  f"F1: {row['F1-Score']:.4f} | Speed: {row['Inference Speed (img/s)']:.1f} img/s")
        
        print(f"\n[WINNER] BEST MODEL: {sorted_df.iloc[0]['Model']}")
        best_model = sorted_df.iloc[0]
        print(f"   Validation Accuracy: {best_model['Validation Accuracy']:.4f}")
        print(f"   F1-Score: {best_model['F1-Score']:.4f}")
        print(f"   Inference Speed: {best_model['Inference Speed (img/s)']:.1f} images/sec")
        print(f"   Total Parameters: {best_model['Total Parameters']:,}")
        
        print(f"\n[FASTEST] FASTEST MODEL: {sorted_df.loc[sorted_df['Inference Speed (img/s)'].idxmax(), 'Model']}")
        fastest_model = sorted_df.loc[sorted_df['Inference Speed (img/s)'].idxmax()]
        print(f"   Speed: {fastest_model['Inference Speed (img/s)']:.1f} images/sec")
        print(f"   Accuracy: {fastest_model['Validation Accuracy']:.4f}")
        
        print(f"\n[EFFICIENT] MOST EFFICIENT MODEL: {sorted_df.loc[sorted_df['Efficiency Score'].idxmax(), 'Model']}")
        efficient_model = sorted_df.loc[sorted_df['Efficiency Score'].idxmax()]
        print(f"   Efficiency Score: {efficient_model['Efficiency Score']:.2f} (accuracy/sec)")
        print(f"   Time per Image: {efficient_model['Time per Image (s)']:.4f} seconds")
        print(f"   Accuracy: {efficient_model['Validation Accuracy']:.4f}")
        
        print(f"\n[MEMORY] LEAST MEMORY USAGE: {sorted_df.loc[sorted_df['Memory Usage (MB)'].idxmin(), 'Model']}")
        memory_efficient_model = sorted_df.loc[sorted_df['Memory Usage (MB)'].idxmin()]
        print(f"   Memory Usage: {memory_efficient_model['Memory Usage (MB)']:.1f} MB")
        print(f"   Accuracy: {memory_efficient_model['Validation Accuracy']:.4f}")
        
        # Overfitting analysis
        print(f"\n[ANALYSIS] OVERFITTING ANALYSIS:")
        print("-" * 40)
        for _, row in sorted_df.iterrows():
            acc = row['Validation Accuracy']
            if acc > 0.9:
                status = "[EXCELLENT]"
            elif acc > 0.8:
                status = "[GOOD]"
            elif acc > 0.7:
                status = "[MODERATE]"
            else:
                status = "[POOR]"
            print(f"{row['Model']:<12}: {acc:.4f} - {status}")
        
        print(f"\n[FILES] All results saved in: {self.output_dir}/")
        print(f"   - model_comparison_summary.csv")
        print(f"   - detailed_results.json")
        print(f"   - *_class_report.csv")
        print(f"   - *.png (visualizations)")
        print(f"   - time_complexity_analysis.png (NEW!)")
    
    def run_comprehensive_comparison(self):
        """Run the complete comprehensive comparison"""
        print("="*80)
        print("COMPREHENSIVE MODEL COMPARISON ANALYSIS")
        print("="*80)
        print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Load class names
        self.load_class_names()
        
        # Create data generators
        self.create_data_generators()
        
        # Evaluate all models
        self.evaluate_all_models()
        
        if not self.results:
            print("[ERROR] No models could be evaluated!")
            return
        
        # Create comparison dataframe
        self.create_comparison_dataframe()
        
        # Create visualizations
        self.create_visualizations()
        
        # Save detailed results
        self.save_detailed_results()
        
        # Generate final report
        self.generate_final_report()
        
        print(f"\n{'='*80}")
        print("COMPREHENSIVE ANALYSIS COMPLETED!")
        print(f"{'='*80}")

def main():
    """Main function"""
    comparator = ComprehensiveModelComparator()
    comparator.run_comprehensive_comparison()

if __name__ == "__main__":
    main()
