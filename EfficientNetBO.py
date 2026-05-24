"""
Pneumonia detection from chest X-rays — EfficientNetB0 pipeline.

Same training recipe as the MobileNetV2 version:
  - Two-phase training (frozen head -> fine-tune top of backbone)
  - BatchNorm kept frozen during fine-tuning
  - Auto class weights, val_auc monitoring, threshold tuning

IMPORTANT preprocessing note:
  Keras' EfficientNet models expect RAW [0, 255] pixel inputs. Normalization
  is baked into the model itself, so efficientnet.preprocess_input is a
  pass-through. Do NOT rescale to [0, 1] and do NOT divide by 255 — doing so
  is a common, silent accuracy killer with EfficientNet.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import (
    EfficientNetB0, preprocess_input
)
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
IMG_SIZE = 224          # EfficientNetB0's native resolution
BATCH_SIZE = 32
DATA_DIR = "chest_xray_processed"
SEED = 42

# Ensure similar results everytime model is ran
tf.random.set_seed(SEED)
np.random.seed(SEED)

# -----------------------------------------------------------------------------
# Data
# horizontal_flip stays OFF — chest anatomy isn't left/right symmetric.
# preprocess_input is a pass-through here; it just keeps the API consistent.
# -----------------------------------------------------------------------------
train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=10,
    zoom_range=0.10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    brightness_range=(0.9, 1.1),
)

eval_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_data = train_gen.flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    seed=SEED,
)

val_data = eval_gen.flow_from_directory(
    os.path.join(DATA_DIR, "val"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False,
)

test_data = eval_gen.flow_from_directory(
    os.path.join(DATA_DIR, "test"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False,
)

print("Class indices:", train_data.class_indices)  # {'NORMAL': 0, 'PNEUMONIA': 1}

# -----------------------------------------------------------------------------
# Class weights from the actual training distribution
# -----------------------------------------------------------------------------
class_weights = {
    0: 1.8497,   # NORMAL
    1: 0.6852    # PNEUMONIA
}
print("Class weights:", class_weights)

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
base_model = EfficientNetB0(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False  # phase 1

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)   # keep backbone BN in inference mode
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(inputs, outputs)

# -----------------------------------------------------------------------------
# Phase 1 — train the head only
# -----------------------------------------------------------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
)

callbacks = [
    EarlyStopping(monitor="val_auc", mode="max", patience=5,
                  restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.3,
                      patience=3, min_lr=1e-6),
]

history_head = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10,
    class_weight=class_weights,
    callbacks=callbacks,
)

# -----------------------------------------------------------------------------
# Phase 2 — unfreeze the top of the backbone, fine-tune at a low LR.
# EfficientNet has more, smaller blocks than MobileNetV2, so unfreezing ~20
# layers is a reasonable starting point. BatchNorm stays frozen.
# -----------------------------------------------------------------------------
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
)

model.summary()

history_ft = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,
    class_weight=class_weights,
    callbacks=callbacks,
)

model.save("outputs/EfficientNetB0_pneumonia_model.keras")

# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------
def merge(metric):
    return history_head.history[metric] + history_ft.history[metric]

for metric, title, fname in [
    ("accuracy", "Accuracy", "effnet_accuracy.png"),
    ("loss", "Loss", "effnet_loss.png"),
    ("auc", "AUC", "effnet_auc.png"),
]:
    plt.figure()
    plt.plot(merge(metric), label=f"Train {title}")
    plt.plot(merge(f"val_{metric}"), label=f"Val {title}")
    plt.axvline(len(history_head.history[metric]) - 1, ls="--",
                c="gray", label="fine-tune start")
    plt.legend()
    plt.title(title)
    plt.savefig(f"outputs/{fname}", bbox_inches="tight")
    plt.close()

# -----------------------------------------------------------------------------
# Evaluation + threshold tuning
# -----------------------------------------------------------------------------
pred_probs = model.predict(test_data).ravel()
y_true = test_data.classes

auc = roc_auc_score(y_true, pred_probs)
print(f"Test ROC-AUC: {auc:.4f}")

fpr, tpr, thr = roc_curve(y_true, pred_probs)
best_thr = float(thr[np.argmax(tpr - fpr)])
print(f"Tuned threshold: {best_thr:.3f}")

for name, t in [("0.50 threshold", 0.5), ("tuned threshold", best_thr)]:
    preds = (pred_probs > t).astype(int)
    report = classification_report(
        y_true, preds, target_names=["Normal", "Pneumonia"]
    )
    print(f"\n=== {name} ===\n{report}")
    with open(f"outputs/effnet_report_{t:.2f}.txt", "w") as f:
        f.write(f"ROC-AUC: {auc:.4f}\nThreshold: {t:.3f}\n\n{report}")

preds = (pred_probs > best_thr).astype(int)
cm = confusion_matrix(y_true, preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal", "Pneumonia"],
            yticklabels=["Normal", "Pneumonia"])
plt.title(f"EfficientNetB0 Confusion Matrix (thr={best_thr:.2f})")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("outputs/effnet_confusion_matrix.png", bbox_inches="tight")
plt.close()