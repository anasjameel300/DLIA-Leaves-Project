import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard, LearningRateScheduler
from tensorflow.keras.utils import to_categorical
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DenseNet121Trainer:
    def __init__(self, data_path="removed background", img_size=(224, 224), batch_size=32):
        self.data_path = data_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.history = None
        self.history_finetune = None
        self.class_names = []
        self.results = {}
        self.class_weights = None
        
        # Check GPU availability
        self.check_gpu()
        
    def check_gpu(self):
        """Check and configure GPU settings"""
        print("="*60)
        print("GPU CONFIGURATION")
        print("="*60)
        
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"[OK] Found {len(gpus)} GPU(s):")
            for gpu in gpus:
                print(f"  - {gpu.name}")
            
            # Enable memory growth to avoid OOM errors
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("[OK] GPU memory growth enabled")
            
            # Set mixed precision for faster training
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            print("[OK] Mixed precision enabled for faster training")
        else:
            print("[WARNING] No GPU found, using CPU")
            
        print(f"TensorFlow version: {tf.__version__}")
        print("="*60)
    
    def load_data(self):
        """Load and prepare the dataset"""
        print("\n" + "="*60)
        print("DATA LOADING AND PREPARATION")
        print("="*60)
        
        # Get class names (plant categories)
        self.class_names = sorted([d for d in os.listdir(self.data_path) 
                                  if os.path.isdir(os.path.join(self.data_path, d))])
        
        print(f"Found {len(self.class_names)} plant categories:")
        for i, class_name in enumerate(self.class_names):
            class_path = os.path.join(self.data_path, class_name)
            num_images = len([f for f in os.listdir(class_path) if f.endswith('.png')])
            print(f"  {i+1:2d}. {class_name:<20} - {num_images:4d} images")
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.2
        )
        
        # Only rescaling for validation
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
        )
        
        # Load training data
        self.train_generator = train_datagen.flow_from_directory(
            self.data_path,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        # Load validation data
        self.val_generator = val_datagen.flow_from_directory(
            self.data_path,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        print(f"\nTraining samples: {self.train_generator.samples}")
        print(f"Validation samples: {self.val_generator.samples}")
        print(f"Total samples: {self.train_generator.samples + self.val_generator.samples}")
        
        # Store class indices
        self.class_indices = self.train_generator.class_indices
        print(f"\nClass indices: {self.class_indices}")
        
        # Calculate class weights for imbalanced data
        self.calculate_class_weights()
        
    def build_model(self):
        """Build DenseNet121 model with transfer learning"""
        print("\n" + "="*60)
        print("BUILDING DENSENET121 MODEL")
        print("="*60)
        
        # Load pre-trained DenseNet121
        base_model = DenseNet121(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.img_size, 3)
        )
        
        print(f"[OK] Loaded pre-trained DenseNet121")
        print(f"[OK] Base model parameters: {base_model.count_params():,}")
        
        # Freeze base model layers
        base_model.trainable = False
        print("[OK] Frozen base model layers")
        
        # Create new model
        inputs = tf.keras.Input(shape=(*self.img_size, 3))
        x = base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        outputs = Dense(len(self.class_names), activation='softmax')(x)
        
        self.model = Model(inputs, outputs)
        
        print(f"[OK] Model built successfully")
        print(f"[OK] Total parameters: {self.model.count_params():,}")
        print(f"[OK] Trainable parameters: {sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights]):,}")
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("[OK] Model compiled successfully")
    
    def calculate_class_weights(self):
        """Calculate class weights to handle imbalanced datasets"""
        from sklearn.utils.class_weight import compute_class_weight
        
        # Get class distribution
        class_counts = {}
        for class_name in self.class_names:
            class_path = os.path.join(self.data_path, class_name)
            count = len([f for f in os.listdir(class_path) if f.endswith('.png')])
            class_counts[class_name] = count
        
        # Compute class weights
        class_indices_list = list(range(len(self.class_names)))
        class_counts_array = np.array([class_counts[name] for name in self.class_names])
        
        weights = compute_class_weight(
            class_weight='balanced',
            classes=np.array(class_indices_list),
            y=np.repeat(class_indices_list, class_counts_array)
        )
        
        self.class_weights = dict(enumerate(weights))
        print(f"\n[OK] Class weights calculated for balanced training")
        print(f"  Weight range: {min(weights):.2f} - {max(weights):.2f}")
        
    def train_model(self, epochs=50, fine_tune_epochs=15):
        """Train the model with two-phase approach: frozen base + fine-tuning"""
        print("\n" + "="*60)
        print("PHASE 1: TRAINING DENSENET121 MODEL (FROZEN BASE)")
        print("="*60)
        
        # Learning rate schedule for warmup
        def lr_schedule(epoch, lr):
            if epoch < 5:
                return 0.0001 * (epoch + 1) / 5  # Warmup
            return lr
        
        # Create callbacks for phase 1
        callbacks_phase1 = [
            ModelCheckpoint(
                'densenet121_best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=12,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            LearningRateScheduler(lr_schedule, verbose=0),
            TensorBoard(
                log_dir=f'densenet121_results/logs',
                histogram_freq=0,
                write_graph=False
            )
        ]
        
        # Phase 1: Train with frozen base
        print("Starting Phase 1 training (frozen base layers)...")
        self.history = self.model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.val_generator,
            callbacks=callbacks_phase1,
            class_weight=self.class_weights,
            verbose=1
        )
        
        print("[OK] Phase 1 training completed!")
        
        # Phase 2: Fine-tuning
        print("\n" + "="*60)
        print("PHASE 2: FINE-TUNING DENSENET121 MODEL")
        print("="*60)
        
        # Unfreeze the last dense block
        base_model = self.model.layers[1]
        base_model.trainable = True
        
        # Freeze all layers except the last 12 (last dense block)
        for layer in base_model.layers[:-12]:
            layer.trainable = False
        
        print(f"[OK] Unfroze last 12 layers of base model for fine-tuning")
        print(f"[OK] Trainable parameters: {sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights]):,}")
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=Adam(learning_rate=0.00005, clipnorm=1.0),  # Lower LR + gradient clipping
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("[OK] Model recompiled with lower learning rate (0.00005) and gradient clipping")
        
        # Callbacks for phase 2
        callbacks_phase2 = [
            ModelCheckpoint(
                'densenet121_best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=8,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-8,
                verbose=1
            ),
            TensorBoard(
                log_dir=f'densenet121_results/logs_finetune',
                histogram_freq=0,
                write_graph=False
            )
        ]
        
        # Fine-tune the model
        print(f"Starting Phase 2 fine-tuning ({fine_tune_epochs} epochs)...")
        self.history_finetune = self.model.fit(
            self.train_generator,
            epochs=fine_tune_epochs,
            validation_data=self.val_generator,
            callbacks=callbacks_phase2,
            class_weight=self.class_weights,
            verbose=1
        )
        
        print("[OK] Phase 2 fine-tuning completed!")
        
        # Combine histories
        for key in self.history.history.keys():
            self.history.history[key].extend(self.history_finetune.history[key])
        
    def evaluate_model(self):
        """Evaluate the model and generate metrics"""
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        # Evaluate on validation set
        val_loss, val_accuracy = self.model.evaluate(self.val_generator, verbose=0)
        
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        
        # Get predictions
        val_generator_no_shuffle = ImageDataGenerator(rescale=1./255).flow_from_directory(
            self.data_path,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        predictions = self.model.predict(val_generator_no_shuffle, verbose=0)
        y_pred = np.argmax(predictions, axis=1)
        y_true = val_generator_no_shuffle.classes
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Classification report
        report = classification_report(
            y_true, y_pred, 
            target_names=self.class_names,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Store results
        self.results = {
            'model_name': 'DenseNet121',
            'validation_accuracy': val_accuracy,
            'validation_loss': val_loss,
            'accuracy_score': accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'class_names': self.class_names,
            'training_history': {
                'loss': [float(x) for x in self.history.history['loss']],
                'accuracy': [float(x) for x in self.history.history['accuracy']],
                'val_loss': [float(x) for x in self.history.history['val_loss']],
                'val_accuracy': [float(x) for x in self.history.history['val_accuracy']]
            },
            'class_weights': {int(k): float(v) for k, v in self.class_weights.items()},
            'model_summary': {
                'total_parameters': self.model.count_params(),
                'trainable_parameters': sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights]),
                'input_shape': self.img_size,
                'num_classes': len(self.class_names)
            }
        }
        
        print(f"[OK] Model evaluation completed")
        print(f"[OK] Final Validation Accuracy: {val_accuracy:.4f}")
        
    def save_results(self):
        """Save all results and visualizations"""
        print("\n" + "="*60)
        print("SAVING RESULTS")
        print("="*60)
        
        # Create results directory
        os.makedirs('densenet121_results', exist_ok=True)
        
        # Save model
        self.model.save('densenet121_results/densenet121_final_model.h5')
        print("[OK] Model saved")
        
        # Save results dictionary
        with open('densenet121_results/densenet121_results.pkl', 'wb') as f:
            pickle.dump(self.results, f)
        print("[OK] Results dictionary saved")
        
        # Save results as JSON (for easy reading)
        json_results = self.results.copy()
        # Convert numpy arrays to lists for JSON serialization
        json_results['confusion_matrix'] = self.results['confusion_matrix']
        with open('densenet121_results/densenet121_results.json', 'w') as f:
            json.dump(json_results, f, indent=2)
        print("[OK] Results JSON saved")
        
        # Create visualizations
        self.create_visualizations()
        
        print("[OK] All results saved in 'densenet121_results' folder")
        
    def create_visualizations(self):
        """Create and save training visualizations"""
        # Training history plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        ax1.plot(self.history.history['loss'], label='Training Loss')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax2.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        # Confusion matrix
        cm = np.array(self.results['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, 
                   yticklabels=self.class_names, ax=ax3)
        ax3.set_title('Confusion Matrix')
        ax3.set_xlabel('Predicted')
        ax3.set_ylabel('Actual')
        
        # Class accuracy
        class_accuracy = []
        for i in range(len(self.class_names)):
            if cm[i].sum() > 0:
                class_accuracy.append(cm[i][i] / cm[i].sum())
            else:
                class_accuracy.append(0)
        
        ax4.bar(range(len(self.class_names)), class_accuracy)
        ax4.set_title('Per-Class Accuracy')
        ax4.set_xlabel('Class Index')
        ax4.set_ylabel('Accuracy')
        ax4.set_xticks(range(len(self.class_names)))
        ax4.set_xticklabels(range(len(self.class_names)), rotation=45)
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig('densenet121_results/densenet121_training_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save detailed classification report
        report_df = pd.DataFrame(self.results['classification_report']).transpose()
        report_df.to_csv('densenet121_results/densenet121_classification_report.csv')
        
        print("[OK] Visualizations created and saved")
        
    def run_training_pipeline(self, epochs=50):
        """Run the complete training pipeline"""
        start_time = datetime.now()
        print(f"Starting DenseNet121 training pipeline at {start_time}")
        
        try:
            self.load_data()
            self.build_model()
            self.train_model(epochs)
            self.evaluate_model()
            self.save_results()
            
            end_time = datetime.now()
            training_time = end_time - start_time
            
            print("\n" + "="*60)
            print("DENSENET121 TRAINING PIPELINE COMPLETED!")
            print("="*60)
            print(f"Training time: {training_time}")
            print(f"Final validation accuracy: {self.results['validation_accuracy']:.4f}")
            print(f"Results saved in: densenet121_results/")
            
        except Exception as e:
            print(f"Error during training: {e}")
            raise

if __name__ == "__main__":
    # Initialize trainer
    trainer = DenseNet121Trainer(
        data_path="removed background",
        img_size=(224, 224),
        batch_size=32  # Optimized batch size for DenseNet121
    )
    
    # Run training pipeline with two-phase training
    trainer.run_training_pipeline(epochs=35)  # Reduced epochs due to two-phase training
