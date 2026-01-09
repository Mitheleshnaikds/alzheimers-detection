import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime
import pandas as pd

# --- Set paths ---
MODEL_PATH = './App/model.h5'  # adjust if needed
DATASET_PATH = './dataset/AugmentedAlzheimerDataset'  # adjust if needed
IMG_SIZE = 128  # use the same as during training
BATCH_SIZE = 32

# --- Load model ---
model = tf.keras.models.load_model(MODEL_PATH)

# --- Prepare data generator (no augmentation, just rescale) ---
datagen = ImageDataGenerator(rescale=1.0/255.0)
val_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# --- Predict ---
y_true = val_gen.classes
y_pred = model.predict(val_gen, verbose=1)
y_pred_classes = np.argmax(y_pred, axis=1)
class_names = list(val_gen.class_indices.keys())

# --- Evaluate (loss / metrics) on the whole generator ---
eval_results = None
try:
    eval_results = model.evaluate(val_gen, verbose=1)
    metric_names = getattr(model, 'metrics_names', None)
except Exception:
    eval_results = None
    metric_names = None

# --- Confusion Matrix ---
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix_local.png')
plt.show()

# --- ROC Curve ---
n_classes = len(class_names)
y_true_bin = label_binarize(y_true, classes=range(n_classes))
fpr = dict()
tpr = dict()
roc_auc = dict()
plt.figure(figsize=(8, 6))
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], lw=2, label=f'ROC curve for {class_names[i]} (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('roc_curves_local.png')
plt.show()

# --- Save textual evaluation summary ---
out_lines = []
out_lines.append(f"Model evaluation - {datetime.datetime.now().isoformat()}")
out_lines.append('\n-- Model Summary --')
try:
    # capture model summary
    import io
    buf = io.StringIO()
    model.summary(print_fn=lambda s: buf.write(s + "\n"))
    out_lines.append(buf.getvalue())
except Exception as e:
    out_lines.append(f"Could not get model summary: {e}")

if eval_results is not None and metric_names is not None:
    out_lines.append('\n-- Evaluation Metrics --')
    if isinstance(eval_results, (list, tuple)):
        for name, val in zip(metric_names, eval_results):
            out_lines.append(f"{name}: {val}")
    else:
        out_lines.append(f"loss: {eval_results}")
else:
    out_lines.append('\nEvaluation metrics not available via model.evaluate()')

# Classification report
out_lines.append('\n-- Classification Report --')
try:
    report = classification_report(y_true, y_pred_classes, target_names=class_names)
    out_lines.append(report)
except Exception as e:
    out_lines.append(f"Could not compute classification report: {e}")

# Confusion matrix (as text and CSV)
out_lines.append('\n-- Confusion Matrix --')
out_lines.append(np.array2string(cm))
try:
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv('confusion_matrix_local.csv')
    out_lines.append('\nConfusion matrix saved to confusion_matrix_local.csv')
except Exception as e:
    out_lines.append(f"Could not save confusion matrix CSV: {e}")

# ROC AUC per class
out_lines.append('\n-- ROC AUC per class --')
try:
    for i in range(n_classes):
        out_lines.append(f"{class_names[i]}: AUC = {roc_auc[i]:.4f}")
except Exception as e:
    out_lines.append(f"Could not compute ROC AUC per class: {e}")

# Save overall report to text file
with open('model_evaluation.txt', 'w', encoding='utf-8') as fh:
    fh.write('\n'.join(out_lines))

print('Saved evaluation summary to model_evaluation.txt')
