import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.models import * 
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import load_img, img_to_array
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import cv2

# Configuration
use_grayscale = False  # Define this before using it
target_size = (256, 256)
batch_size = 8

# Enhanced data generators with more augmentation
train_datagen = image.ImageDataGenerator(
    rotation_range=20,
    shear_range=0.25,
    zoom_range=0.25,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    width_shift_range=0.15,
    height_shift_range=0.15,
    brightness_range=[0.8, 1.2],
    channel_shift_range=50
)

val_datagen = image.ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
    'cervical fracture/train',
    target_size=target_size,
    batch_size=batch_size,
    class_mode='binary',
    color_mode='grayscale' if use_grayscale else 'rgb'
)

validation_generator = val_datagen.flow_from_directory(
    'cervical fracture/val',
    target_size=target_size,
    batch_size=batch_size,
    shuffle=False,
    class_mode='binary',
    color_mode='grayscale' if use_grayscale else 'rgb'
)

# Enhanced model architecture
def build_model():
    base_model = tf.keras.applications.EfficientNetB4(
        weights='imagenet', 
        input_shape=(target_size[0], target_size[1], 3), 
        include_top=False
    )
    
    # Unfreeze some layers for fine-tuning
    for layer in base_model.layers[-20:]:
        layer.trainable = True
        
    model = Sequential([
        base_model,
        GaussianNoise(0.2),
        GlobalAveragePooling2D(),
        Dense(512, activation='swish', kernel_regularizer='l2'),
        BatchNormalization(),
        Dropout(0.4),
        Dense(256, activation='swish', kernel_regularizer='l2'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.AUC(name='prc', curve='PR')  # Precision-Recall curve
        ]
    )
    
    return model

# Enhanced callbacks
callbacks = [
    ModelCheckpoint('models/best_model.h5', monitor='val_prc', mode='max', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_prc', patience=5, mode='max', restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6),
    tf.keras.callbacks.TensorBoard(log_dir='./logs')
]

# Create models directory if it doesn't exist
import os
os.makedirs('models', exist_ok=True)

# Train the model
model = build_model()
history = model.fit(
    train_generator,
    epochs=1,
    validation_data=validation_generator,
    callbacks=callbacks
)

# Post-training analysis
def evaluate_model():
    # Generate predictions
    y_pred = model.predict(validation_generator)
    y_pred = (y_pred > 0.5).astype(int)
    y_true = validation_generator.classes
    
    # Classification report
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Fracture']))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Fracture'],
                yticklabels=['Normal', 'Fracture'])
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    
    # Plot training history
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.legend()
    plt.savefig('training_history.png')

evaluate_model()