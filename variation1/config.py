# ================================================================
# ACML Project — Pneumonia Detection from Chest X-Rays
# Variation 1: Custom CNN with Batch Normalisation
#
# config.py — All settings and paths in one place.
# Edit this file to change any settings for the model.
# ================================================================

# ---------------------------------------------------------------
# DATASET PATH
#
# Default assumes chest_xray_processed/ is in the same folder
# as this script.
#
# If running on Google Colab, change to:
#   DATA_DIR = '/content/chest_xray_processed'
# ---------------------------------------------------------------
DATA_DIR = 'chest_xray_processed'

# ---------------------------------------------------------------
# IMAGE SETTINGS
# Must match what preprocess.py produced
# ---------------------------------------------------------------
IMG_SIZE   = (224, 224)   # width x height in pixels
IMG_CHANNELS = 3          # RGB

# ---------------------------------------------------------------
# TRAINING SETTINGS
# ---------------------------------------------------------------
BATCH_SIZE = 32           # number of images processed at a time
EPOCHS     = 30           # maximum number of training epochs
SEED       = 42           # for reproducibility

# ---------------------------------------------------------------
# CLASS WEIGHTS
#
# Computed by class_weights.py to handle class imbalance.
# Penalises the model more for errors on NORMAL (minority class).
#
#   NORMAL    : 1108 images
#   PNEUMONIA : 2991 images
# ---------------------------------------------------------------
CLASS_WEIGHTS = {
    0: 1.8497,   # NORMAL
    1: 0.6852    # PNEUMONIA
}

# ---------------------------------------------------------------
# OUTPUT PATHS
# All saved files go here
# ---------------------------------------------------------------
MODEL_SAVE_PATH       = 'best_model_variation1.keras'
HISTORY_PLOT_PATH     = 'training_history_variation1.png'
CONFUSION_MATRIX_PATH = 'confusion_matrix_variation1.png'
ROC_CURVE_PATH        = 'roc_curve_variation1.png'
