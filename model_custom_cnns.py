# ================================================================
# ACML Project — Pneumonia Detection from Chest X-Rays
# Variation 1: Custom CNN with Batch Normalisation
#
# Team Members:
#   Amogelang Moenyane (2708897)
#   Paballo Moloantoa  (2451669)
#   Lanet
#   Aphiwe
#
# How to run:
#   1. Make sure chest_xray_processed/ is in the same folder
#      as this script
#   2. Install dependencies:
#      pip install tensorflow numpy matplotlib seaborn scikit-learn
#   3. Run:
#      python model_variation1.py
#
# To run on Google Colab:
#   1. Mount Google Drive
#   2. Copy chest_xray_processed/ to /content/
#   3. Change DATA_DIR below to '/content/chest_xray_processed'
# ================================================================


# ---------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)

print("TensorFlow version:", tf.__version__)


# ---------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------

DATA_DIR   = 'chest_xray_processed'   # path to the processed dataset
IMG_SIZE   = (224, 224)               # image size set by preprocess.py
BATCH_SIZE = 32                       # images per training batch
EPOCHS     = 30                       # maximum training epochs
SEED       = 42                       # for reproducibility

# Class weights computed by class_weights.py
# Penalises the model more for errors on NORMAL (minority class)
CLASS_WEIGHTS = {
    0: 1.8497,   # NORMAL
    1: 0.6852    # PNEUMONIA
}

TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR   = os.path.join(DATA_DIR, 'val')
TEST_DIR  = os.path.join(DATA_DIR, 'test')


# ---------------------------------------------------------------
# DATA LOADING AND AUGMENTATION
#
# Augmentation is applied ONLY to the training set to artificially
# increase variety and reduce overfitting. The validation and test
# sets are only rescaled — no augmentation — so we evaluate on
# realistic, unmodified images.
#
# Rescaling divides pixel values by 255 so all values are in [0,1].
# ---------------------------------------------------------------

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.1,
    horizontal_flip=True,
)

val_test_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

# flow_from_directory reads folder structure and assigns labels:
# NORMAL = 0, PNEUMONIA = 1 (alphabetical order)
train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    color_mode='rgb',
    seed=SEED
)

val_gen = val_test_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    color_mode='rgb',
    seed=SEED
)

test_gen = val_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    color_mode='rgb',
    shuffle=False,   # keep order for evaluation
    seed=SEED
)


# ---------------------------------------------------------------
# MODEL ARCHITECTURE
#
# Variation 1: Custom CNN with Batch Normalisation
#
#   Input (224 x 224 x 3)
#   |
#   Conv Block 1: Conv(32, 3x3) -> BatchNorm -> ReLU -> MaxPool(2x2)
#   Conv Block 2: Conv(64, 3x3) -> BatchNorm -> ReLU -> MaxPool(2x2)
#   Conv Block 3: Conv(128,3x3) -> BatchNorm -> ReLU -> MaxPool(2x2)
#   Conv Block 4: Conv(256,3x3) -> BatchNorm -> ReLU -> MaxPool(2x2)
#   |
#   Flatten
#   Dense(256) -> ReLU -> Dropout(0.5)
#   Dense(1)   -> Sigmoid
#   |
#   Output: probability of PNEUMONIA
#
# Batch Normalisation (added after each Conv layer):
#   Rescales the outputs of each convolutional layer to a stable
#   range during training. This makes training faster and more
#   stable, which is especially important with a small dataset.
#
# Dropout(0.5):
#   Randomly switches off 50% of neurons in the fully connected
#   layer during training. This prevents overfitting by stopping
#   the model from relying too heavily on any single neuron.
#
# Filter count doubles each block (32->64->128->256):
#   Early layers detect simple features (edges, curves) and need
#   fewer kernels. Deeper layers combine these into complex
#   patterns and need more kernels to represent them.
# ---------------------------------------------------------------

def build_model():
    model = models.Sequential([

        layers.Input(shape=(224, 224, 3)),

        # Conv Block 1
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),

        # Conv Block 2
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),

        # Conv Block 3
        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),

        # Conv Block 4
        layers.Conv2D(256, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),

        # Flatten: unrolls 3D feature maps into a 1D array
        layers.Flatten(),

        # Fully connected layer
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),

        # Output: single node, sigmoid for binary classification
        # > 0.5 = PNEUMONIA, <= 0.5 = NORMAL
        layers.Dense(1, activation='sigmoid')
    ])

    return model


model = build_model()

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()


# ---------------------------------------------------------------
# CALLBACKS
#
# EarlyStopping:
#   Stops training if validation loss does not improve for 5
#   consecutive epochs. Prevents overfitting.
#
# ModelCheckpoint:
#   Saves the best version of the model (lowest validation loss)
#   automatically during training.
#
# ReduceLROnPlateau:
#   Halves the learning rate if validation loss does not improve
#   for 3 epochs. Helps fine-tune the model near convergence.
# ---------------------------------------------------------------

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        filepath='best_model_variation1.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        verbose=1
    )
]


# ---------------------------------------------------------------
# TRAINING
#
# class_weight handles the imbalance between NORMAL and PNEUMONIA.
# validation_data monitors overfitting during training but is NOT
# used for final evaluation — that is done on the test set.
# ---------------------------------------------------------------

print("\nStarting training...\n")

history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    class_weight=CLASS_WEIGHTS,
    callbacks=callbacks
)

print("\nTraining complete.")


# ---------------------------------------------------------------
# TRAINING HISTORY PLOTS
#
# Both curves should move together. If training accuracy is much
# higher than validation accuracy the model is overfitting
# (Lecture 6, Section 4.4).
# ---------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['accuracy'], label='Training Accuracy')
axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
axes[0].set_title('Model Accuracy over Epochs')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(history.history['loss'], label='Training Loss')
axes[1].plot(history.history['val_loss'], label='Validation Loss')
axes[1].set_title('Model Loss over Epochs')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('training_history_variation1.png', dpi=150)
plt.show()


# ---------------------------------------------------------------
# EVALUATION ON TEST SET
#
# The test set is data the model has NEVER seen during training
# or validation. These are the metrics that matter.
#
# Metrics used (Lecture 6):
#   Accuracy   = (TP + TN) / total
#   Precision  = TP / (TP + FP)
#   Recall     = TP / (TP + FN)
#   F1 Score   = harmonic mean of Precision and Recall
#
# Recall is the most important metric for this problem —
# we must not miss pneumonia cases (false negatives).
# ---------------------------------------------------------------

y_true     = test_gen.classes
y_pred_prob = model.predict(test_gen, verbose=1).flatten()
y_pred     = (y_pred_prob > 0.5).astype(int)

accuracy = np.mean(y_true == y_pred)
print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=['NORMAL', 'PNEUMONIA']
))


# ---------------------------------------------------------------
# CONFUSION MATRIX (Lecture 6, Section 4.1)
#
#   TP: PNEUMONIA correctly predicted as PNEUMONIA
#   TN: NORMAL correctly predicted as NORMAL
#   FP: NORMAL incorrectly predicted as PNEUMONIA
#   FN: PNEUMONIA incorrectly predicted as NORMAL (most dangerous)
# ---------------------------------------------------------------

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['NORMAL', 'PNEUMONIA'],
    yticklabels=['NORMAL', 'PNEUMONIA']
)
plt.title('Confusion Matrix — Variation 1 (Custom CNN)')
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')
plt.tight_layout()
plt.savefig('confusion_matrix_variation1.png', dpi=150)
plt.show()

tn, fp, fn, tp = cm.ravel()
print(f"\nTrue Positives  (TP): {tp}  — Pneumonia correctly identified")
print(f"True Negatives  (TN): {tn}  — Normal correctly identified")
print(f"False Positives (FP): {fp}  — Normal incorrectly flagged as Pneumonia")
print(f"False Negatives (FN): {fn}  — Pneumonia missed (most dangerous error)")

precision = tp / (tp + fp)
recall    = tp / (tp + fn)
f1        = 2 * (precision * recall) / (precision + recall)

print(f"\nPrecision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")


# ---------------------------------------------------------------
# ROC CURVE AND AUC (Lecture 6, Section 4.2)
#
# Plots True Positive Rate vs False Positive Rate for all
# possible threshold values.
#
# AUC = 1.0 → perfect model
# AUC = 0.5 → no better than random guessing
#
# Use AUC to compare this variation against the other two.
# ---------------------------------------------------------------

fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2,
         label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='grey', lw=1,
         linestyle='--', label='Random Classifier (AUC = 0.5)')
plt.scatter(fpr, tpr, color='blue', s=10)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve — Variation 1 (Custom CNN)')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig('roc_curve_variation1.png', dpi=150)
plt.show()

print(f"\nAUC Score: {roc_auc:.4f}")


# ---------------------------------------------------------------
# FINAL SUMMARY
# ---------------------------------------------------------------

print("=" * 55)
print("  VARIATION 1 — CUSTOM CNN — FINAL RESULTS SUMMARY")
print("=" * 55)
print(f"  Accuracy:  {accuracy*100:.2f}%")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1 Score:  {f1:.4f}")
print(f"  AUC:       {roc_auc:.4f}")
print("=" * 55)
