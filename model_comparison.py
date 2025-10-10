import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ModelComparator:
    def __init__(self):
        self.models = ['VGG19', 'ResNet50', 'DenseNet121', 'MobileNet']
        self.results = {}
        self.comparison_data = {}
        
    def load_all_results(self):
        """Load results from all trained models"""
        print("="*60)
        print("LOADING MODEL RESULTS")
        print("="*60)
        
        for model_name in self.models:
            results_path = f"{model_name.lower()}_results/{model_name.lower()}_results.pkl"
            if os.path.exists(results_path):
                with open(results_path, 'rb') as f:
                    self.results[model_name] = pickle.load(f)
                print(f"✓ Loaded {model_name} results")
            else:
                print(f"⚠ {model_name} results not found at {results_path}")
                
        print(f"\nLoaded results for {len(self.results)} models")
        
    def create_comparison_summary(self):
        """Create a comprehensive comparison summary"""
        print("\n" + "="*60)
        print("CREATING MODEL COMPARISON")
        print("="*60)
        
        # Extract key metrics
        comparison_data = []
        
        for model_name, results in self.results.items():
            # Basic metrics
            accuracy = results['validation_accuracy']
            loss = results['validation_loss']
            total_params = results['model_summary']['total_parameters']
            trainable_params = results['model_summary']['trainable_parameters']
            
            # Training history metrics
            history = results['training_history']
            final_train_acc = history['accuracy'][-1] if history['accuracy'] else 0
            final_train_loss = history['loss'][-1] if history['loss'] else 0
            
            # Calculate overfitting (difference between train and val accuracy)
            overfitting = final_train_acc - accuracy
            
            # Calculate convergence speed (epochs to reach 90% of final accuracy)
            convergence_epochs = self._calculate_convergence_epochs(history['val_accuracy'])
            
            comparison_data.append({
                'Model': model_name,
                'Validation Accuracy': accuracy,
                'Validation Loss': loss,
                'Training Accuracy': final_train_acc,
                'Training Loss': final_train_loss,
                'Overfitting': overfitting,
                'Total Parameters': total_params,
                'Trainable Parameters': trainable_params,
                'Convergence Epochs': convergence_epochs,
                'Model Size (MB)': total_params * 4 / (1024 * 1024)  # Assuming float32
            })
        
        self.comparison_data = pd.DataFrame(comparison_data)
        
        # Sort by validation accuracy
        self.comparison_data = self.comparison_data.sort_values('Validation Accuracy', ascending=False)
        
        print("✓ Comparison data created")
        return self.comparison_data
    
    def _calculate_convergence_epochs(self, val_acc_history):
        """Calculate epochs needed to reach 90% of final accuracy"""
        if not val_acc_history:
            return 0
        
        final_acc = val_acc_history[-1]
        target_acc = final_acc * 0.9
        
        for i, acc in enumerate(val_acc_history):
            if acc >= target_acc:
                return i + 1
        
        return len(val_acc_history)
    
    def create_visualizations(self):
        """Create comprehensive comparison visualizations"""
        print("\n" + "="*60)
        print("CREATING COMPARISON VISUALIZATIONS")
        print("="*60)
        
        # Create output directory
        os.makedirs('model_comparison_results', exist_ok=True)
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Accuracy and Loss Comparison
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # Validation Accuracy
        models = self.comparison_data['Model']
        accuracies = self.comparison_data['Validation Accuracy']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        bars1 = ax1.bar(models, accuracies, color=colors, alpha=0.8)
        ax1.set_title('Validation Accuracy Comparison', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, acc in zip(bars1, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Validation Loss
        losses = self.comparison_data['Validation Loss']
        bars2 = ax2.bar(models, losses, color=colors, alpha=0.8)
        ax2.set_title('Validation Loss Comparison', fontsize=16, fontweight='bold')
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, loss in zip(bars2, losses):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{loss:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Model Parameters
        total_params = self.comparison_data['Total Parameters'] / 1e6  # Convert to millions
        bars3 = ax3.bar(models, total_params, color=colors, alpha=0.8)
        ax3.set_title('Model Parameters (Millions)', fontsize=16, fontweight='bold')
        ax3.set_ylabel('Parameters (M)', fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, params in zip(bars3, total_params):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{params:.1f}M', ha='center', va='bottom', fontweight='bold')
        
        # Overfitting Analysis
        overfitting = self.comparison_data['Overfitting']
        bars4 = ax4.bar(models, overfitting, color=colors, alpha=0.8)
        ax4.set_title('Overfitting Analysis (Train - Val Accuracy)', fontsize=16, fontweight='bold')
        ax4.set_ylabel('Overfitting', fontsize=12)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, overfit in zip(bars4, overfitting):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{overfit:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('model_comparison_results/accuracy_loss_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Training History Comparison
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # Training Accuracy History
        for i, model_name in enumerate(self.models):
            if model_name in self.results:
                history = self.results[model_name]['training_history']
                epochs = range(1, len(history['accuracy']) + 1)
                ax1.plot(epochs, history['accuracy'], label=f'{model_name} (Train)', 
                        color=colors[i], linewidth=2)
                ax1.plot(epochs, history['val_accuracy'], label=f'{model_name} (Val)', 
                        color=colors[i], linestyle='--', linewidth=2)
        
        ax1.set_title('Training vs Validation Accuracy', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Epochs', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Training Loss History
        for i, model_name in enumerate(self.models):
            if model_name in self.results:
                history = self.results[model_name]['training_history']
                epochs = range(1, len(history['loss']) + 1)
                ax2.plot(epochs, history['loss'], label=f'{model_name} (Train)', 
                        color=colors[i], linewidth=2)
                ax2.plot(epochs, history['val_loss'], label=f'{model_name} (Val)', 
                        color=colors[i], linestyle='--', linewidth=2)
        
        ax2.set_title('Training vs Validation Loss', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Epochs', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Model Efficiency (Accuracy vs Parameters)
        ax3.scatter(total_params, accuracies, s=200, c=colors, alpha=0.8)
        for i, model in enumerate(models):
            ax3.annotate(model, (total_params.iloc[i], accuracies.iloc[i]), 
                        xytext=(5, 5), textcoords='offset points', fontweight='bold')
        
        ax3.set_title('Model Efficiency: Accuracy vs Parameters', fontsize=16, fontweight='bold')
        ax3.set_xlabel('Parameters (Millions)', fontsize=12)
        ax3.set_ylabel('Validation Accuracy', fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        # Convergence Speed
        convergence = self.comparison_data['Convergence Epochs']
        bars_conv = ax4.bar(models, convergence, color=colors, alpha=0.8)
        ax4.set_title('Convergence Speed (Epochs to 90% Accuracy)', fontsize=16, fontweight='bold')
        ax4.set_ylabel('Epochs', fontsize=12)
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, conv in zip(bars_conv, convergence):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{conv}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('model_comparison_results/training_history_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Confusion Matrix Comparison
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.ravel()
        
        for i, model_name in enumerate(self.models):
            if model_name in self.results:
                cm = np.array(self.results[model_name]['confusion_matrix'])
                class_names = self.results[model_name]['class_names']
                
                # Normalize confusion matrix
                cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                
                sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                           xticklabels=class_names, yticklabels=class_names, 
                           ax=axes[i], cbar_kws={'shrink': 0.8})
                axes[i].set_title(f'{model_name} Confusion Matrix', fontsize=14, fontweight='bold')
                axes[i].set_xlabel('Predicted', fontsize=10)
                axes[i].set_ylabel('Actual', fontsize=10)
                
                # Rotate x-axis labels for better readability
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].tick_params(axis='y', rotation=0)
        
        plt.tight_layout()
        plt.savefig('model_comparison_results/confusion_matrices_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ All visualizations created and saved")
        
    def create_detailed_report(self):
        """Create a detailed comparison report"""
        print("\n" + "="*60)
        print("CREATING DETAILED REPORT")
        print("="*60)
        
        # Save comparison data
        self.comparison_data.to_csv('model_comparison_results/model_comparison_summary.csv', index=False)
        print("✓ Comparison summary saved to CSV")
        
        # Create detailed report
        report = []
        report.append("="*80)
        report.append("MEDICINAL PLANT LEAF CLASSIFICATION - MODEL COMPARISON REPORT")
        report.append("="*80)
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Overall Summary
        report.append("OVERALL SUMMARY")
        report.append("-" * 40)
        best_model = self.comparison_data.iloc[0]
        report.append(f"Best Model: {best_model['Model']}")
        report.append(f"Best Validation Accuracy: {best_model['Validation Accuracy']:.4f}")
        report.append(f"Best Validation Loss: {best_model['Validation Loss']:.4f}")
        report.append("")
        
        # Detailed Model Analysis
        report.append("DETAILED MODEL ANALYSIS")
        report.append("-" * 40)
        
        for _, row in self.comparison_data.iterrows():
            report.append(f"\n{row['Model']}:")
            report.append(f"  Validation Accuracy: {row['Validation Accuracy']:.4f}")
            report.append(f"  Validation Loss: {row['Validation Loss']:.4f}")
            report.append(f"  Training Accuracy: {row['Training Accuracy']:.4f}")
            report.append(f"  Overfitting: {row['Overfitting']:.4f}")
            report.append(f"  Total Parameters: {row['Total Parameters']:,}")
            report.append(f"  Model Size: {row['Model Size (MB)']:.1f} MB")
            report.append(f"  Convergence Epochs: {row['Convergence Epochs']}")
        
        # Recommendations
        report.append("\n" + "="*80)
        report.append("RECOMMENDATIONS")
        report.append("="*80)
        
        # Best for accuracy
        best_acc_model = self.comparison_data.iloc[0]
        report.append(f"🏆 Best for Accuracy: {best_acc_model['Model']}")
        report.append(f"   - Validation Accuracy: {best_acc_model['Validation Accuracy']:.4f}")
        report.append(f"   - Use case: Production systems where accuracy is critical")
        
        # Best for efficiency
        efficiency_scores = self.comparison_data['Validation Accuracy'] / (self.comparison_data['Total Parameters'] / 1e6)
        best_efficiency_idx = efficiency_scores.idxmax()
        best_efficiency_model = self.comparison_data.iloc[best_efficiency_idx]
        report.append(f"\n⚡ Best for Efficiency: {best_efficiency_model['Model']}")
        report.append(f"   - Efficiency Score: {efficiency_scores.iloc[best_efficiency_idx]:.4f}")
        report.append(f"   - Use case: Resource-constrained environments")
        
        # Best for speed
        fastest_convergence_idx = self.comparison_data['Convergence Epochs'].idxmin()
        fastest_model = self.comparison_data.iloc[fastest_convergence_idx]
        report.append(f"\n🚀 Fastest Convergence: {fastest_model['Model']}")
        report.append(f"   - Convergence Epochs: {fastest_model['Convergence Epochs']}")
        report.append(f"   - Use case: Rapid prototyping and development")
        
        # Best for mobile/edge
        smallest_model_idx = self.comparison_data['Model Size (MB)'].idxmin()
        smallest_model = self.comparison_data.iloc[smallest_model_idx]
        report.append(f"\n📱 Best for Mobile/Edge: {smallest_model['Model']}")
        report.append(f"   - Model Size: {smallest_model['Model Size (MB)']:.1f} MB")
        report.append(f"   - Use case: Mobile applications and edge devices")
        
        # Save report
        with open('model_comparison_results/comparison_report.txt', 'w') as f:
            f.write('\n'.join(report))
        
        print("✓ Detailed report saved")
        
        # Print summary to console
        print("\n" + "="*80)
        print("MODEL COMPARISON SUMMARY")
        print("="*80)
        print(self.comparison_data.to_string(index=False))
        
    def run_comparison(self):
        """Run the complete comparison analysis"""
        print("Starting model comparison analysis...")
        
        self.load_all_results()
        self.create_comparison_summary()
        self.create_visualizations()
        self.create_detailed_report()
        
        print("\n" + "="*60)
        print("MODEL COMPARISON COMPLETED!")
        print("="*60)
        print("Results saved in: model_comparison_results/")
        print("Files created:")
        print("  - model_comparison_summary.csv")
        print("  - comparison_report.txt")
        print("  - accuracy_loss_comparison.png")
        print("  - training_history_comparison.png")
        print("  - confusion_matrices_comparison.png")

if __name__ == "__main__":
    comparator = ModelComparator()
    comparator.run_comparison()
