import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.preprocessing import image
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import os

# --- 1. Define Data Paths ---
TRAIN_PATH = 'cervical fracture/train'
VAL_PATH = 'cervical fracture/val'

# --- 2. Setup Data Augmentation and Generators ---
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

val_datagen = image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=(256, 256), # Ensure this matches your app's target_size
    batch_size=32,
    class_mode='binary'
)

validation_generator = val_datagen.flow_from_directory(
    VAL_PATH,
    target_size=(256, 256), # Ensure this matches your app's target_size
    batch_size=32,
    shuffle=False,
    class_mode='binary'
)

print("Data Generators created successfully.")
print("Class Indices:", train_generator.class_indices)


# --- 3. Build the Model using Transfer Learning (MobileNetV2) ---
base_model = tf.keras.applications.MobileNetV2(
    weights='imagenet',
    input_shape=(256, 256, 3), # Ensure this matches target_size
    include_top=False
)

for layer in base_model.layers:
    layer.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

print("Model Architecture:")
model.summary()

# --- 4. Compile the Model for Phase 1 ---
model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy', 'Precision', 'Recall', 'AUC']
)

# --- 5. Define Callbacks ---
checkpoint = ModelCheckpoint(
    'best_model.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max'
)
early_stopping = EarlyStopping(
    monitor='val_loss', patience=10, verbose=1, restore_best_weights=True # Increased patience
)
lrp = ReduceLROnPlateau(
    monitor="val_loss", factor=0.2, patience=4, verbose=1
)
callbacks = [checkpoint, early_stopping, lrp]

# --- 6. Run Training Phase 1: Feature Extraction ---
print("\n--- Starting Model Training (Phase 1: Feature Extraction) ---")
# Train only the custom top layers
history = model.fit(
    train_generator,
    epochs=15, # Train for a solid number of epochs to warm up the new layers
    validation_data=validation_generator,
    steps_per_epoch=len(train_generator),
    validation_steps=len(validation_generator),
    callbacks=callbacks
)

# --- 7. Run Training Phase 2: Fine-Tuning ---
print("\n--- Starting Model Training (Phase 2: Fine-Tuning) ---")
base_model.trainable = True

# Freeze the early layers and unfreeze the later ones
for layer in base_model.layers[:100]:
    layer.trainable = False

# Re-compile the model with a very low learning rate for fine-tuning
model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), # 100x smaller LR
    metrics=['accuracy', 'Precision', 'Recall', 'AUC']
)

# Continue training with the fine-tuning configuration
history_fine_tune = model.fit(
    train_generator,
    epochs=15, # Train for more epochs to fine-tune
    validation_data=validation_generator,
    steps_per_epoch=len(train_generator),
    validation_steps=len(validation_generator),
    callbacks=callbacks
)

# --- 8. Evaluate the Final Model ---
print("--- Final Model Evaluation ---")
print("Evaluating on Training Data:")
model.evaluate(train_generator)
print("\nEvaluating on Validation Data:")
model.evaluate(validation_generator)


# --- 9. Plot Combined Training History ---
def combine_history(h1, h2):
    """Combine two Keras history objects."""
    history = {}
    for key in h1.history.keys():
        history[key] = h1.history[key] + h2.history[key]
    return history

full_history = combine_history(history, history_fine_tune)

plt.figure(figsize=(12, 5))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(full_history['accuracy'], label='Train Accuracy')
plt.plot(full_history['val_accuracy'], label='Validation Accuracy')
plt.axvline(x=len(history.history['accuracy'])-1, color='r', linestyle='--', label='Start of Fine-Tuning')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(full_history['loss'], label='Train Loss')
plt.plot(full_history['val_loss'], label='Validation Loss')
plt.axvline(x=len(history.history['loss'])-1, color='r', linestyle='--', label='Start of Fine-Tuning')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_history_with_finetuning.png')
plt.show()

print("\nTraining complete. A new model 'best_model.h5' and history plot have been saved.")