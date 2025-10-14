"""
Advanced Regularization Techniques for Deep Learning
This module provides additional regularization methods to prevent overfitting.
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import to_categorical
import random

class LabelSmoothing(Layer):
    """
    Label Smoothing Regularization Layer
    Reduces overconfidence in predictions by smoothing the target labels
    """
    def __init__(self, smoothing=0.1, **kwargs):
        super(LabelSmoothing, self).__init__(**kwargs)
        self.smoothing = smoothing
    
    def call(self, inputs):
        # inputs should be one-hot encoded labels
        num_classes = tf.shape(inputs)[-1]
        smooth_labels = (1 - self.smoothing) * inputs + self.smoothing / num_classes
        return smooth_labels

class MixUp:
    """
    MixUp Data Augmentation
    Creates virtual training examples by mixing pairs of examples and their labels
    """
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def mixup_batch(self, x_batch, y_batch):
        """Apply mixup to a batch of data"""
        batch_size = tf.shape(x_batch)[0]
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Create random permutation
        perm = tf.random.shuffle(tf.range(batch_size))
        
        # Mix images
        mixed_x = lam * x_batch + (1 - lam) * tf.gather(x_batch, perm)
        
        # Mix labels
        mixed_y = lam * y_batch + (1 - lam) * tf.gather(y_batch, perm)
        
        return mixed_x, mixed_y

class CutMix:
    """
    CutMix Data Augmentation
    Cuts and pastes patches from one image to another
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def cutmix_batch(self, x_batch, y_batch):
        """Apply CutMix to a batch of data"""
        batch_size = tf.shape(x_batch)[0]
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Create random permutation
        perm = tf.random.shuffle(tf.range(batch_size))
        
        # Get image dimensions
        H, W = tf.shape(x_batch)[1], tf.shape(x_batch)[2]
        
        # Calculate cut region
        cut_rat = tf.sqrt(1. - lam)
        cut_w = tf.cast(W * cut_rat, tf.int32)
        cut_h = tf.cast(H * cut_rat, tf.int32)
        
        # Random center
        cx = tf.random.uniform([], 0, W, dtype=tf.int32)
        cy = tf.random.uniform([], 0, H, dtype=tf.int32)
        
        # Bounding box
        bbx1 = tf.clip_by_value(cx - cut_w // 2, 0, W)
        bby1 = tf.clip_by_value(cy - cut_h // 2, 0, H)
        bbx2 = tf.clip_by_value(cx + cut_w // 2, 0, W)
        bby2 = tf.clip_by_value(cy + cut_h // 2, 0, H)
        
        # Create masks
        mask = tf.zeros([H, W])
        mask = tf.tensor_scatter_nd_update(
            mask, 
            tf.stack([tf.meshgrid(tf.range(bby1, bby2), tf.range(bbx1, bbx2), indexing='ij')], axis=-1),
            tf.ones([bby2-bby1, bbx2-bbx1])
        )
        mask = tf.expand_dims(mask, axis=-1)
        mask = tf.expand_dims(mask, axis=0)
        
        # Apply cutmix
        mixed_x = x_batch * (1 - mask) + tf.gather(x_batch, perm) * mask
        
        # Adjust lambda
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        mixed_y = lam * y_batch + (1 - lam) * tf.gather(y_batch, perm)
        
        return mixed_x, mixed_y

class FocalLoss:
    """
    Focal Loss for addressing class imbalance
    Reduces the loss contribution from easy examples
    """
    def __init__(self, alpha=1.0, gamma=2.0):
        self.alpha = alpha
        self.gamma = gamma
    
    def __call__(self, y_true, y_pred):
        # Compute cross entropy
        ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        
        # Compute focal loss
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1)
        alpha_t = self.alpha
        focal_loss = alpha_t * tf.pow(1 - p_t, self.gamma) * ce
        
        return tf.reduce_mean(focal_loss)

class CosineAnnealingScheduler:
    """
    Cosine Annealing Learning Rate Scheduler
    Reduces learning rate following a cosine curve
    """
    def __init__(self, T_max, eta_min=0):
        self.T_max = T_max
        self.eta_min = eta_min
    
    def __call__(self, epoch, lr):
        return self.eta_min + (lr - self.eta_min) * (1 + np.cos(np.pi * epoch / self.T_max)) / 2

class StochasticWeightAveraging:
    """
    Stochastic Weight Averaging (SWA)
    Averages model weights during training to improve generalization
    """
    def __init__(self, model, start_epoch=10):
        self.model = model
        self.start_epoch = start_epoch
        self.swa_weights = None
        self.swa_count = 0
    
    def update(self, epoch):
        """Update SWA weights"""
        if epoch >= self.start_epoch:
            if self.swa_weights is None:
                self.swa_weights = [tf.Variable(w) for w in self.model.get_weights()]
            else:
                for swa_w, w in zip(self.swa_weights, self.model.get_weights()):
                    swa_w.assign(swa_w * self.swa_count / (self.swa_count + 1) + w / (self.swa_count + 1))
            self.swa_count += 1
    
    def apply_swa(self):
        """Apply SWA weights to model"""
        if self.swa_weights is not None:
            self.model.set_weights([w.numpy() for w in self.swa_weights])

def create_advanced_callbacks(model_name, use_swa=True, use_cosine_annealing=True):
    """
    Create advanced callbacks for training
    """
    callbacks = []
    
    # Model checkpoint
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(
        f'{model_name}_best_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ))
    
    # Early stopping with reduced patience
    callbacks.append(tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=6,  # Reduced patience
        restore_best_weights=True,
        verbose=1
    ))
    
    # Learning rate reduction
    callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,  # More aggressive reduction
        patience=3,
        min_lr=1e-8,
        verbose=1
    ))
    
    # Cosine annealing scheduler
    if use_cosine_annealing:
        callbacks.append(tf.keras.callbacks.LearningRateScheduler(
            CosineAnnealingScheduler(T_max=50, eta_min=1e-6),
            verbose=0
        ))
    
    # TensorBoard
    callbacks.append(tf.keras.callbacks.TensorBoard(
        log_dir=f'{model_name}_results/logs',
        histogram_freq=0,
        write_graph=False
    ))
    
    return callbacks

def apply_advanced_regularization(model, use_label_smoothing=True, smoothing=0.1):
    """
    Apply advanced regularization techniques to a model
    """
    if use_label_smoothing:
        # Add label smoothing layer
        model.add(LabelSmoothing(smoothing=smoothing))
    
    return model

def create_regularized_optimizer(learning_rate=0.001, weight_decay=1e-4):
    """
    Create optimizer with weight decay
    """
    return tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        decay=weight_decay,
        clipnorm=1.0  # Gradient clipping
    )
