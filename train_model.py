"""
Alzheimer's Disease Detection - Model Training Script
Supports training a baseline CNN or transfer learning models.

Architectures:
- cnn (baseline)
- resnet50 (ImageNet pretrained)
- efficientnetb0 (ImageNet pretrained)
"""

import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Optional imports for transfer learning
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess

# For hybrid CNN+SVM
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib

# For confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Default configuration
DEFAULT_IMG_SIZE_CNN = 128
DEFAULT_IMG_SIZE_TL = 224
BATCH_SIZE = 32
EPOCHS_DEFAULT = 100
DATASET_PATH_DEFAULT = './dataset/AugmentedAlzheimerDataset'  # Using augmented dataset for better accuracy
MODEL_SAVE_DIR = './App'
VALIDATION_SPLIT = 0.2  # 20% for validation

print("=" * 60)
print("Alzheimer's Disease Detection - Training Script")
print("=" * 60)

# CLI args
parser = argparse.ArgumentParser(description="Train Alzheimer's model")
parser.add_argument('--model', choices=['cnn', 'resnet50', 'efficientnetb0', 'cnn_svm'], default='cnn', help='Model architecture to train')
parser.add_argument('--dataset', default=DATASET_PATH_DEFAULT, help='Path to dataset root directory')
parser.add_argument('--epochs', type=int, default=EPOCHS_DEFAULT, help='Number of training epochs')
parser.add_argument('--img-size', type=int, default=None, help='Image size (square) to use')
parser.add_argument('--output', default=None, help='Custom output model path (h5)')
args = parser.parse_args()

ARCH = args.model
DATASET_PATH = args.dataset
EPOCHS = args.epochs
IMG_SIZE = args.img_size or (DEFAULT_IMG_SIZE_CNN if ARCH == 'cnn' else DEFAULT_IMG_SIZE_TL)
MODEL_SAVE_PATH = args.output or os.path.join(MODEL_SAVE_DIR, f"model_{ARCH}.h5" if ARCH != 'cnn' else 'model.h5')

# Check if dataset exists
if not os.path.exists(DATASET_PATH):
    print(f"\n❌ ERROR: Dataset not found at {DATASET_PATH}")
    print("Please ensure the dataset is placed in: dataset/ folder")
    exit()

print("\n✅ Dataset found!")

def get_datagen(arch: str):
    """Create ImageDataGenerator with appropriate preprocessing for the architecture."""
    base_kwargs = dict(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=VALIDATION_SPLIT
    )

    if arch == 'resnet50':
        return ImageDataGenerator(preprocessing_function=resnet_preprocess, **base_kwargs)
    elif arch == 'efficientnetb0':
        return ImageDataGenerator(preprocessing_function=effnet_preprocess, **base_kwargs)
    else:  # cnn or cnn_svm
        return ImageDataGenerator(rescale=1.0/255.0, **base_kwargs)

print("\n📊 Setting up data augmentation...")
train_datagen = get_datagen(ARCH)

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

def build_model(arch: str, img_size: int, num_classes: int = 4):
    """Build selected model architecture."""
    if arch == 'resnet50':
        print("\n🏗️  Building ResNet50 transfer learning model...")
        base = ResNet50(include_top=False, weights='imagenet', input_shape=(img_size, img_size, 3))
        base.trainable = False  # start with feature extractor frozen
        x = layers.GlobalAveragePooling2D()(base.output)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        model = models.Model(inputs=base.input, outputs=outputs)
        optimizer = keras.optimizers.Adam(learning_rate=1e-3)
        return model, optimizer
    elif arch == 'efficientnetb0':
        print("\n🏗️  Building EfficientNetB0 transfer learning model...")
        base = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(img_size, img_size, 3))
        base.trainable = False
        x = layers.GlobalAveragePooling2D()(base.output)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        model = models.Model(inputs=base.input, outputs=outputs)
        optimizer = keras.optimizers.Adam(learning_rate=1e-3)
        return model, optimizer
    else:
        print("\n🏗️  Building baseline CNN model...")
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
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
            layers.Dense(128, activation='relu', name='feat_dense'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        optimizer = keras.optimizers.Adam(learning_rate=1e-3)
        return model, optimizer

model, optimizer = build_model(ARCH, IMG_SIZE)

# Compile model
model.compile(
    optimizer=optimizer,
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
print(f"🏷️  Architecture: {ARCH}")
print(f"🖼️  Image size: {IMG_SIZE}x{IMG_SIZE}")
print(f"💾 Output: {MODEL_SAVE_PATH}")
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
print(f"✅ Validation Accuracy: {test_accuracy * 100:.2f}%")
print(f"✅ Validation Loss: {test_loss:.4f}")

# Step 7: Generate Confusion Matrix
print("\n📊 Generating confusion matrix...")
val_ds.reset()
y_true = val_ds.classes
y_pred = model.predict(val_ds, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)

# Get class names
class_names = list(val_ds.class_indices.keys())

# Create confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title(f'Confusion Matrix - {ARCH.upper()}\nValidation Accuracy: {test_accuracy * 100:.2f}%')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()

# Save confusion matrix
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
cm_path = os.path.join(MODEL_SAVE_DIR, f'confusion_matrix_{ARCH}.png')
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
print(f"✅ Confusion matrix saved to {cm_path}")
plt.close()

# Print classification report
print("\n📋 Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=class_names))

# Optional: Hybrid CNN+SVM training
if ARCH == 'cnn_svm':
    print("\n🔧 Preparing features for CNN+SVM classifier...")
    # Deterministic feature extraction (no augmentation, no shuffle)
    feat_datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=VALIDATION_SPLIT)
    feat_train = feat_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
        subset='training'
    )
    feat_val = feat_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
        subset='validation'
    )

    # Build feature model from saved CNN to ensure inputs are defined
    try:
        base_model = keras.models.load_model(MODEL_SAVE_PATH)
        feature_layer = base_model.get_layer('feat_dense').output
        feature_model = models.Model(inputs=base_model.input, outputs=feature_layer)
    except Exception as e:
        raise RuntimeError(f"Failed to build feature extractor from saved model: {e}")

    steps_train = int(np.ceil(feat_train.samples / feat_train.batch_size))
    steps_val = int(np.ceil(feat_val.samples / feat_val.batch_size))
    print("\n🔍 Extracting train features...")
    X_train = feature_model.predict(feat_train, steps=steps_train, verbose=1)
    y_train = feat_train.classes[:X_train.shape[0]]
    print("🔍 Extracting val features...")
    X_val = feature_model.predict(feat_val, steps=steps_val, verbose=1)
    y_val = feat_val.classes[:X_val.shape[0]]

    print("\n🚀 Training SVM classifier...")
    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(kernel='rbf', probability=True, C=2.0, gamma='scale'))
    ])
    clf.fit(X_train, y_train)
    val_acc = clf.score(X_val, y_val)
    print(f"✅ CNN+SVM Validation Accuracy: {val_acc * 100:.2f}%")

    # Generate confusion matrix for CNN+SVM
    print("\n📊 Generating CNN+SVM confusion matrix...")
    y_pred_svm = clf.predict(X_val)
    cm_svm = confusion_matrix(y_val, y_pred_svm)
    class_names_svm = list(feat_train.class_indices.keys())
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Greens', xticklabels=class_names_svm, yticklabels=class_names_svm)
    plt.title(f'Confusion Matrix - CNN+SVM\\nValidation Accuracy: {val_acc * 100:.2f}%')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    cm_svm_path = os.path.join(MODEL_SAVE_DIR, 'confusion_matrix_cnn_svm.png')
    plt.savefig(cm_svm_path, dpi=300, bbox_inches='tight')
    print(f"✅ CNN+SVM confusion matrix saved to {cm_svm_path}")
    plt.close()
    
    print("\n📋 CNN+SVM Classification Report:")
    print(classification_report(y_val, y_pred_svm, target_names=class_names_svm))

    # Save classifier and metadata
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    base = os.path.join(MODEL_SAVE_DIR, 'hybrid_cnn_svm')
    clf_path = f"{base}.pkl"
    meta_path = f"{base}.json"
    joblib.dump(clf, clf_path)
    meta = {
        'backbone': 'cnn',
        'classifier': 'svm',
        'img_size': IMG_SIZE,
        'class_indices': feat_train.class_indices,
        'feature_layer': 'feat_dense',
        'normalization': 'rescale_1/255'
    }
    with open(meta_path, 'w') as f:
        import json
        json.dump(meta, f)
    print(f"💾 Saved CNN+SVM: {clf_path}")
    print(f"💾 Saved metadata: {meta_path}")

print("\n" + "=" * 60)
print("🎉 Training completed successfully!")
print(f"📊 Model saved: {MODEL_SAVE_PATH}")
print("You can now use it with the Streamlit app!")
print("=" * 60)
