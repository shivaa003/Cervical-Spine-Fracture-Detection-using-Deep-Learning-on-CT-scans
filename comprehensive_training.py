"""
Comprehensive Cervical Spine Fracture Detection Training Script
Trains both MobileNetV2 and EfficientNetB4 models with extensive metrics
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, auc, 
                            precision_recall_curve, average_precision_score, f1_score,
                            cohen_kappa_score, matthews_corrcoef)
import pandas as pd
import os
import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Paths
    TRAIN_PATH = 'cervical fracture/train'
    VAL_PATH = 'cervical fracture/val'
    OUTPUT_DIR = 'training_output'
    
    # Training parameters
    TARGET_SIZE = (256, 256)
    BATCH_SIZE = 32  # Will be reduced for EfficientNet if needed
    EPOCHS_PHASE1 = 15
    EPOCHS_PHASE2 = 15
    LEARNING_RATE = 0.001
    FINE_TUNE_LR = 1e-5
    
    # Model selection
    AVAILABLE_MODELS = ['mobilenetv2', 'efficientnetb4']
    
    @classmethod
    def create_dirs(cls):
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(f'{cls.OUTPUT_DIR}/models', exist_ok=True)
        os.makedirs(f'{cls.OUTPUT_DIR}/plots', exist_ok=True)
        os.makedirs(f'{cls.OUTPUT_DIR}/reports', exist_ok=True)

Config.create_dirs()

# ============================================================================
# DATA PREPARATION
# ============================================================================

class DataManager:
    def __init__(self, config):
        self.config = config
        self.train_generator = None
        self.val_generator = None
        self.class_indices = None
        
    def create_generators(self, model_type='mobilenetv2'):
        """Create data generators with appropriate augmentation"""
        
        # Common augmentation for both models
        if model_type == 'mobilenetv2':
            # MobileNetV2 augmentation (moderate)
            train_datagen = image.ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest',
                width_shift_range=0.1,
                height_shift_range=0.1
            )
        else:
            # EfficientNetB4 augmentation (more aggressive)
            train_datagen = image.ImageDataGenerator(
                rescale=1./255,
                rotation_range=25,
                shear_range=0.25,
                zoom_range=0.25,
                horizontal_flip=True,
                vertical_flip=True,
                fill_mode='nearest',
                width_shift_range=0.15,
                height_shift_range=0.15,
                brightness_range=[0.8, 1.2],
                channel_shift_range=30
            )
        
        val_datagen = image.ImageDataGenerator(rescale=1./255)
        
        # Adjust batch size for EfficientNet
        batch_size = self.config.BATCH_SIZE
        if model_type == 'efficientnetb4':
            batch_size = 8  # Smaller batch size for EfficientNet
        
        # Create generators
        self.train_generator = train_datagen.flow_from_directory(
            self.config.TRAIN_PATH,
            target_size=self.config.TARGET_SIZE,
            batch_size=batch_size,
            class_mode='binary',
            shuffle=True,
            seed=42
        )
        
        self.val_generator = val_datagen.flow_from_directory(
            self.config.VAL_PATH,
            target_size=self.config.TARGET_SIZE,
            batch_size=batch_size,
            class_mode='binary',
            shuffle=False,
            seed=42
        )
        
        self.class_indices = self.train_generator.class_indices
        print(f"\n📊 Class Indices: {self.class_indices}")
        print(f"Training samples: {self.train_generator.samples}")
        print(f"Validation samples: {self.val_generator.samples}")
        
        return self.train_generator, self.val_generator

# ============================================================================
# MODEL BUILDERS
# ============================================================================

class ModelBuilder:
    @staticmethod
    def build_mobilenetv2(input_shape=(256, 256, 3)):
        """Build MobileNetV2 model (from train_model.py)"""
        
        # Base model
        base_model = tf.keras.applications.MobileNetV2(
            weights='imagenet',
            input_shape=input_shape,
            include_top=False
        )
        
        # Freeze base model initially
        for layer in base_model.layers:
            layer.trainable = False
        
        # Build model
        model = tf.keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])
        
        return model, base_model
    
    @staticmethod
    def build_efficientnetb4(input_shape=(256, 256, 3)):
        """Build EfficientNetB4 model (from model_training.py)"""
        
        # Base model
        base_model = tf.keras.applications.EfficientNetB4(
            weights='imagenet',
            input_shape=input_shape,
            include_top=False
        )
        
        # Unfreeze last 20 layers
        for layer in base_model.layers[-20:]:
            layer.trainable = True
        
        # Build model
        model = tf.keras.Sequential([
            base_model,
            layers.GaussianNoise(0.2),
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='swish', kernel_regularizer='l2'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(256, activation='swish', kernel_regularizer='l2'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        
        return model, base_model

# ============================================================================
# METRICS AND VISUALIZATION
# ============================================================================

class MetricsVisualizer:
    def __init__(self, model_name, output_dir):
        self.model_name = model_name
        self.output_dir = output_dir
        self.metrics = {}
        
    def calculate_all_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate comprehensive metrics"""
        
        # Convert to binary
        y_pred_binary = (y_pred_proba > 0.5).astype(int)
        
        # Basic metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        accuracy = accuracy_score(y_true, y_pred_binary)
        precision = precision_score(y_true, y_pred_binary)
        recall = recall_score(y_true, y_pred_binary)
        f1 = f1_score(y_true, y_pred_binary)
        
        # Advanced metrics
        specificity = recall_score(y_true, y_pred_binary, pos_label=0)
        npv = precision_score(y_true, y_pred_binary, pos_label=0)
        
        # ROC and PR curves
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = average_precision_score(y_true, y_pred_proba)
        
        # Other metrics
        kappa = cohen_kappa_score(y_true, y_pred_binary)
        mcc = matthews_corrcoef(y_true, y_pred_binary)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred_binary)
        tn, fp, fn, tp = cm.ravel()
        
        # Derived metrics
        tpr = tp / (tp + fn)  # Sensitivity/Recall
        tnr = tn / (tn + fp)  # Specificity
        ppv = tp / (tp + fp)  # Precision
        npv = tn / (tn + fn)  # Negative Predictive Value
        fpr_rate = fp / (fp + tn)
        fnr = fn / (fn + tp)
        
        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'npv': npv,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'kappa': kappa,
            'mcc': mcc,
            'tpr': tpr,
            'tnr': tnr,
            'ppv': ppv,
            'npv_detail': npv,
            'fpr': fpr_rate,
            'fnr': fnr,
            'confusion_matrix': cm.tolist()
        }
        
        return metrics_dict
    
    def plot_training_history(self, history, history_fine=None):
        """Plot comprehensive training history"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'{self.model_name} - Training History', fontsize=16, fontweight='bold')
        
        # Combine histories if fine-tuning was done
        if history_fine:
            full_history = {}
            for key in history.history.keys():
                full_history[key] = history.history[key] + history_fine.history[key]
            epochs = range(1, len(full_history['accuracy']) + 1)
            split_epoch = len(history.history['accuracy'])
        else:
            full_history = history.history
            epochs = range(1, len(full_history['accuracy']) + 1)
            split_epoch = None
        
        # 1. Accuracy
        ax = axes[0, 0]
        ax.plot(epochs, full_history['accuracy'], 'b-', label='Train Accuracy', linewidth=2)
        ax.plot(epochs, full_history['val_accuracy'], 'r-', label='Val Accuracy', linewidth=2)
        if split_epoch:
            ax.axvline(x=split_epoch, color='g', linestyle='--', label='Fine-tuning start')
        ax.set_title('Model Accuracy', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Loss
        ax = axes[0, 1]
        ax.plot(epochs, full_history['loss'], 'b-', label='Train Loss', linewidth=2)
        ax.plot(epochs, full_history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        if split_epoch:
            ax.axvline(x=split_epoch, color='g', linestyle='--', label='Fine-tuning start')
        ax.set_title('Model Loss', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Learning Rate (if available)
        ax = axes[0, 2]
        if 'lr' in full_history:
            ax.plot(epochs, full_history['lr'], 'purple', linewidth=2)
            ax.set_title('Learning Rate', fontsize=12, fontweight='bold')
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Learning Rate')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
        
        # 4. Precision
        ax = axes[1, 0]
        if 'precision' in full_history:
            ax.plot(epochs, full_history['precision'], 'b-', label='Train Precision', linewidth=2)
            ax.plot(epochs, full_history['val_precision'], 'r-', label='Val Precision', linewidth=2)
        elif 'Precision' in full_history:
            ax.plot(epochs, full_history['Precision'], 'b-', label='Train Precision', linewidth=2)
            ax.plot(epochs, full_history['val_Precision'], 'r-', label='Val Precision', linewidth=2)
        ax.set_title('Precision', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Precision')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Recall
        ax = axes[1, 1]
        if 'recall' in full_history:
            ax.plot(epochs, full_history['recall'], 'b-', label='Train Recall', linewidth=2)
            ax.plot(epochs, full_history['val_recall'], 'r-', label='Val Recall', linewidth=2)
        elif 'Recall' in full_history:
            ax.plot(epochs, full_history['Recall'], 'b-', label='Train Recall', linewidth=2)
            ax.plot(epochs, full_history['val_Recall'], 'r-', label='Val Recall', linewidth=2)
        ax.set_title('Recall', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Recall')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. AUC
        ax = axes[1, 2]
        if 'auc' in full_history:
            ax.plot(epochs, full_history['auc'], 'b-', label='Train AUC', linewidth=2)
            ax.plot(epochs, full_history['val_auc'], 'r-', label='Val AUC', linewidth=2)
        elif 'AUC' in full_history:
            ax.plot(epochs, full_history['AUC'], 'b-', label='Train AUC', linewidth=2)
            ax.plot(epochs, full_history['val_AUC'], 'r-', label='Val AUC', linewidth=2)
        ax.set_title('AUC', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('AUC')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/plots/{self.model_name}_training_history.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_evaluation_curves(self, y_true, y_pred_proba):
        """Plot ROC and Precision-Recall curves"""
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'{self.model_name} - Evaluation Curves', fontsize=16, fontweight='bold')
        
        # ROC Curve
        ax = axes[0]
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random Classifier')
        ax.fill_between(fpr, tpr, alpha=0.2, color='blue')
        ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
        ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=11)
        ax.set_title('ROC Curve', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        # Precision-Recall Curve
        ax = axes[1]
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = average_precision_score(y_true, y_pred_proba)
        
        ax.plot(recall, precision, 'g-', linewidth=2, label=f'PR Curve (AP = {pr_auc:.3f})')
        ax.fill_between(recall, precision, alpha=0.2, color='green')
        ax.set_xlabel('Recall', fontsize=11)
        ax.set_ylabel('Precision', fontsize=11)
        ax.set_title('Precision-Recall Curve', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/plots/{self.model_name}_evaluation_curves.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return roc_auc, pr_auc
    
    def plot_confusion_matrix(self, y_true, y_pred_proba):
        """Plot confusion matrix with annotations"""
        
        y_pred = (y_pred_proba > 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'{self.model_name} - Confusion Matrix', fontsize=16, fontweight='bold')
        
        # Absolute numbers
        ax = axes[0]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Normal', 'Fracture'],
                   yticklabels=['Normal', 'Fracture'],
                   cbar_kws={'label': 'Count'})
        ax.set_xlabel('Predicted Label', fontsize=11)
        ax.set_ylabel('True Label', fontsize=11)
        ax.set_title('Confusion Matrix (Absolute)', fontsize=12, fontweight='bold')
        
        # Percentages
        ax = axes[1]
        sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Greens', ax=ax,
                   xticklabels=['Normal', 'Fracture'],
                   yticklabels=['Normal', 'Fracture'],
                   cbar_kws={'label': 'Percentage (%)'})
        ax.set_xlabel('Predicted Label', fontsize=11)
        ax.set_ylabel('True Label', fontsize=11)
        ax.set_title('Confusion Matrix (Percentage)', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/plots/{self.model_name}_confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Extract metrics from confusion matrix
        tn, fp, fn, tp = cm.ravel()
        
        metrics = {
            'True Negatives': tn,
            'False Positives': fp,
            'False Negatives': fn,
            'True Positives': tp,
            'Sensitivity (TPR)': tp / (tp + fn),
            'Specificity (TNR)': tn / (tn + fp),
            'Precision (PPV)': tp / (tp + fp),
            'NPV': tn / (tn + fn),
            'False Positive Rate': fp / (fp + tn),
            'False Negative Rate': fn / (fn + tp)
        }
        
        return metrics
    
    def plot_prediction_distribution(self, y_true, y_pred_proba):
        """Plot distribution of predictions"""
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'{self.model_name} - Prediction Distribution', fontsize=16, fontweight='bold')
        
        # Separate predictions by true class
        pred_normal = y_pred_proba[y_true == 0]
        pred_fracture = y_pred_proba[y_true == 1]
        
        # Histogram
        ax = axes[0]
        ax.hist(pred_normal, bins=30, alpha=0.7, color='blue', label='Normal', density=True)
        ax.hist(pred_fracture, bins=30, alpha=0.7, color='red', label='Fracture', density=True)
        ax.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold (0.5)')
        ax.set_xlabel('Prediction Probability')
        ax.set_ylabel('Density')
        ax.set_title('Prediction Distribution by Class')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Box plot
        ax = axes[1]
        data = [pred_normal, pred_fracture]
        bp = ax.boxplot(data, labels=['Normal', 'Fracture'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        ax.axhline(y=0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
        ax.set_ylabel('Prediction Probability')
        ax.set_title('Prediction Distribution Box Plot')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/plots/{self.model_name}_prediction_distribution.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def plot_metrics_comparison(self, metrics_dict):
        """Plot bar chart of key metrics"""
        
        # Select key metrics for visualization
        key_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'specificity', 'roc_auc', 'pr_auc', 'kappa', 'mcc']
        values = [metrics_dict.get(m, 0) for m in key_metrics]
        
        # Create color map based on values
        colors = plt.cm.RdYlGn(values)
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(key_metrics, values, color=colors, edgecolor='black', linewidth=1)
        plt.ylim(0, 1.1)
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title(f'{self.model_name} - Performance Metrics', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/plots/{self.model_name}_metrics_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()

# ============================================================================
# TRAINER CLASS
# ============================================================================

class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.data_manager = DataManager(config)
        self.results = {}
        
    def train_model(self, model_type='mobilenetv2'):
        """Train a specific model type"""
        
        print("\n" + "="*70)
        print(f"🚀 TRAINING {model_type.upper()} MODEL")
        print("="*70)
        
        # Get data generators
        train_gen, val_gen = self.data_manager.create_generators(model_type)
        
        # Build model
        if model_type == 'mobilenetv2':
            model, base_model = ModelBuilder.build_mobilenetv2()
        else:
            model, base_model = ModelBuilder.build_efficientnetb4()
        
        print(f"\n📋 Model Architecture Summary:")
        model.summary()
        
        # Create visualizer
        visualizer = MetricsVisualizer(model_type, self.config.OUTPUT_DIR)
        
        # Phase 1: Feature Extraction
        print(f"\n{'='*50}")
        print(f"📌 PHASE 1: Feature Extraction")
        print(f"{'='*50}")
        
        # Compile for phase 1
        model.compile(
            loss='binary_crossentropy',
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.LEARNING_RATE),
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
        )
        
        # Callbacks for phase 1
        callbacks = [
            ModelCheckpoint(
                f'{self.config.OUTPUT_DIR}/models/{model_type}_phase1.h5',
                monitor='val_accuracy',
                mode='max',
                save_best_only=True,
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
                factor=0.2,
                patience=4,
                min_lr=1e-7,
                verbose=1
            ),
            TensorBoard(
                log_dir=f'{self.config.OUTPUT_DIR}/logs/{model_type}_phase1',
                histogram_freq=1
            )
        ]
        
        # Train phase 1
        history1 = model.fit(
            train_gen,
            epochs=self.config.EPOCHS_PHASE1,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        # Phase 2: Fine-tuning (only for MobileNetV2)
        history2 = None
        if model_type == 'mobilenetv2':
            print(f"\n{'='*50}")
            print(f"📌 PHASE 2: Fine-Tuning")
            print(f"{'='*50}")
            
            # Unfreeze base model
            base_model.trainable = True
            
            # Freeze early layers
            for layer in base_model.layers[:100]:
                layer.trainable = False
            
            # Recompile with lower learning rate
            model.compile(
                loss='binary_crossentropy',
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.FINE_TUNE_LR),
                metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
            )
            
            # Callbacks for phase 2
            callbacks2 = [
                ModelCheckpoint(
                    f'{self.config.OUTPUT_DIR}/models/{model_type}_best.h5',
                    monitor='val_accuracy',
                    mode='max',
                    save_best_only=True,
                    verbose=1
                ),
                EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,
                    patience=3,
                    min_lr=1e-8,
                    verbose=1
                )
            ]
            
            # Train phase 2
            history2 = model.fit(
                train_gen,
                epochs=self.config.EPOCHS_PHASE2,
                validation_data=val_gen,
                callbacks=callbacks2,
                verbose=1
            )
            
            # Save final model
            model.save(f'{self.config.OUTPUT_DIR}/models/{model_type}_final.h5')
        else:
            # For EfficientNet, just save the model
            model.save(f'{self.config.OUTPUT_DIR}/models/{model_type}_best.h5')
        
        # Copy best model to root for Streamlit app
        import shutil
        best_model_path = f'{self.config.OUTPUT_DIR}/models/{model_type}_best.h5'
        if os.path.exists(best_model_path):
            shutil.copy(best_model_path, 'best_model.h5')
            print(f"\n✅ Best model copied to 'best_model.h5' for Streamlit app")
        
        # Plot training history
        visualizer.plot_training_history(history1, history2)
        
        # Evaluate model
        print(f"\n{'='*50}")
        print(f"📊 EVALUATING {model_type.upper()} MODEL")
        print(f"{'='*50}")
        
        # Get predictions
        y_true = val_gen.classes
        y_pred_proba = model.predict(val_gen).flatten()
        
        # Calculate all metrics
        metrics = visualizer.calculate_all_metrics(y_true, y_pred_proba, y_pred_proba)
        
        # Plot evaluation curves
        roc_auc, pr_auc = visualizer.plot_evaluation_curves(y_true, y_pred_proba)
        
        # Plot confusion matrix
        cm_metrics = visualizer.plot_confusion_matrix(y_true, y_pred_proba)
        
        # Plot prediction distribution
        visualizer.plot_prediction_distribution(y_true, y_pred_proba)
        
        # Plot metrics comparison
        visualizer.plot_metrics_comparison(metrics)
        
        # Store results
        self.results[model_type] = {
            'model': model,
            'metrics': metrics,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'confusion_matrix_metrics': cm_metrics
        }
        
        # Print detailed metrics
        print(f"\n📈 Detailed Metrics for {model_type.upper()}:")
        print("-" * 40)
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key:20s}: {value:.4f}")
        
        return model, metrics
    
    def compare_models(self):
        """Compare all trained models"""
        
        if len(self.results) < 2:
            print("\n⚠️ Need at least 2 models for comparison")
            return
        
        print("\n" + "="*70)
        print("🏆 MODEL COMPARISON")
        print("="*70)
        
        # Create comparison dataframe
        comparison_data = []
        for model_name, result in self.results.items():
            metrics = result['metrics']
            row = {
                'Model': model_name.upper(),
                'Accuracy': metrics.get('accuracy', 0),
                'Precision': metrics.get('precision', 0),
                'Recall': metrics.get('recall', 0),
                'F1-Score': metrics.get('f1_score', 0),
                'Specificity': metrics.get('specificity', 0),
                'ROC-AUC': metrics.get('roc_auc', 0),
                'PR-AUC': metrics.get('pr_auc', 0),
                'Kappa': metrics.get('kappa', 0),
                'MCC': metrics.get('mcc', 0)
            }
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        df.set_index('Model', inplace=True)
        
        # Display comparison
        print("\n📊 Model Performance Comparison:")
        print("-" * 80)
        print(df.to_string(float_format='%.4f'))
        
        # Save to CSV
        df.to_csv(f'{self.config.OUTPUT_DIR}/reports/model_comparison.csv')
        
        # Plot comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # Bar plot comparison
        ax = axes[0, 0]
        df.T.plot(kind='bar', ax=ax, colormap='Set2', edgecolor='black')
        ax.set_title('All Metrics Comparison')
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Score')
        ax.legend(title='Model')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.1)
        
        # Radar chart
        ax = axes[0, 1]
        metrics_for_radar = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity', 'ROC-AUC']
        angles = np.linspace(0, 2 * np.pi, len(metrics_for_radar), endpoint=False).tolist()
        angles += angles[:1]
        
        for model_name in df.index:
            values = df.loc[model_name, metrics_for_radar].values.tolist()
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=model_name)
            ax.fill(angles, values, alpha=0.1)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_for_radar)
        ax.set_title('Radar Chart Comparison')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        # ROC curves comparison
        ax = axes[1, 0]
        colors = ['blue', 'red']
        for i, (model_name, result) in enumerate(self.results.items()):
            # Get predictions for ROC curve
            val_gen = self.data_manager.val_generator
            y_true = val_gen.classes
            y_pred = result['model'].predict(val_gen).flatten()
            fpr, tpr, _ = roc_curve(y_true, y_pred)
            ax.plot(fpr, tpr, color=colors[i], linewidth=2, 
                   label=f"{model_name.upper()} (AUC = {result['roc_auc']:.3f})")
        
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # PR curves comparison
        ax = axes[1, 1]
        for i, (model_name, result) in enumerate(self.results.items()):
            val_gen = self.data_manager.val_generator
            y_true = val_gen.classes
            y_pred = result['model'].predict(val_gen).flatten()
            precision, recall, _ = precision_recall_curve(y_true, y_pred)
            ax.plot(recall, precision, color=colors[i], linewidth=2,
                   label=f"{model_name.upper()} (AP = {result['pr_auc']:.3f})")
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curves Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.config.OUTPUT_DIR}/plots/model_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Determine best model
        best_model = df['F1-Score'].idxmax()
        print(f"\n🏆 Best Model based on F1-Score: {best_model}")
        
        # Save best model separately
        best_model_path = f'{self.config.OUTPUT_DIR}/models/{best_model.lower()}_best.h5'
        if os.path.exists(best_model_path):
            import shutil
            shutil.copy(best_model_path, 'best_model.h5')
            shutil.copy(best_model_path, f'{self.config.OUTPUT_DIR}/models/best_overall_model.h5')
            print(f"✅ Best model saved as 'best_model.h5' for Streamlit app")
        
        return df

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main training function"""
    
    print("\n" + "="*80)
    print("🔬 CERVICAL SPINE FRACTURE DETECTION - COMPREHENSIVE TRAINING")
    print("="*80)
    
    # Initialize trainer
    trainer = ModelTrainer(Config)
    
    # Train both models
    for model_type in Config.AVAILABLE_MODELS:
        try:
            trainer.train_model(model_type)
        except Exception as e:
            print(f"\n❌ Error training {model_type}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Compare models
    comparison_df = trainer.compare_models()
    
    # Generate final report
    print("\n" + "="*80)
    print("📋 FINAL REPORT")
    print("="*80)
    
    report = {
        'timestamp': datetime.datetime.now().isoformat(),
        'configuration': {
            'target_size': Config.TARGET_SIZE,
            'batch_size': Config.BATCH_SIZE,
            'epochs_phase1': Config.EPOCHS_PHASE1,
            'epochs_phase2': Config.EPOCHS_PHASE2,
            'learning_rate': Config.LEARNING_RATE,
            'fine_tune_lr': Config.FINE_TUNE_LR
        },
        'results': {}
    }
    
    for model_name, result in trainer.results.items():
        report['results'][model_name] = {
            'metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v 
                       for k, v in result['metrics'].items() if isinstance(v, (float, np.floating))},
            'roc_auc': float(result['roc_auc']),
            'pr_auc': float(result['pr_auc'])
        }
    
    # Save report as JSON
    with open(f'{Config.OUTPUT_DIR}/reports/training_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n✅ Training complete! All outputs saved to 'training_output/' directory")
    print("\n📁 Output directory structure:")
    print("   training_output/")
    print("   ├── models/          - Saved model files")
    print("   ├── plots/           - Training and evaluation plots")
    print("   ├── reports/         - Performance reports and comparisons")
    print("   └── logs/            - TensorBoard logs")
    
    print("\n📊 Best model saved as 'best_model.h5' for Streamlit app")
    print("\n" + "="*80)

if __name__ == "__main__":
    main()