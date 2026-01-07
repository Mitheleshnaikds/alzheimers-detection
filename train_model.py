"""
Alzheimer's Disease Detection - Model Training Script
Trains a CNN model with data augmentation and early stopping
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os
import numpy as np

# Configuration
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 100
DATASET_PATH = './dataset/AugmentedAlzheimerDataset'  # Using augmented dataset for better accuracy
MODEL_SAVE_PATH = './App/model.h5'
VALIDATION_SPLIT = 0.2  # 20% for validation

print("=" * 60)
print("Alzheimer's Disease Detection - Training Script")
print("=" * 60)

# Check if dataset exists
if not os.path.exists(DATASET_PATH):
    print(f"\n❌ ERROR: Dataset not found at {DATASET_PATH}")
    print("Please ensure the dataset is placed in: dataset/ folder")
    exit()

print("\n✅ Dataset found!")

# Step 1: Data Augmentation
print("\n📊 Setting up data augmentation...")

train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=VALIDATION_SPLIT
)

# Load training data
print(f"Loading training data from {DATASET_PATH}...")
train_ds = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    subset='training'
)

# Load validation data
print(f"Loading validation data from {DATASET_PATH}...")
val_ds = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False,
    subset='validation'
)

print(f"✅ Classes: {train_ds.class_indices}")
print(f"✅ Training samples: {train_ds.samples}")
print(f"✅ Validation samples: {val_ds.samples}")

# Step 2: Build Model
print("\n🏗️  Building CNN model...")

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(4, activation='softmax')  # 4 classes
])

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(model.summary())

# Step 3: Setup Callbacks
print("\n⚙️  Setting up training callbacks...")

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=10,
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
    ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# Step 4: Train Model
print("\n🚀 Starting training...")
print("=" * 60)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# Step 5: Save Final Model
print("\n💾 Saving model...")
model.save(MODEL_SAVE_PATH)
print(f"✅ Model saved to {MODEL_SAVE_PATH}")

# Step 6: Evaluate
print("\n📈 Evaluating model...")
test_loss, test_accuracy = model.evaluate(val_ds, verbose=0)
print(f"✅ Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"✅ Test Loss: {test_loss:.4f}")

print("\n" + "=" * 60)
print("🎉 Training completed successfully!")
print(f"📊 Model saved: {MODEL_SAVE_PATH}")
print("You can now use it with the Streamlit app!")
print("=" * 60)
