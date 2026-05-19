import os
import numpy as np
import matplotlib.pyplot as plt

# ── Config ──────────────────────────────────────────────
DATASET_PATH = "chest_xray_processed"
CLASSES      = ["NORMAL", "PNEUMONIA"]

# ── 1. Count training images ─────────────────────────────
print("=" * 50)
print("COMPUTING CLASS WEIGHTS")
print("=" * 50)

normal_count    = len(os.listdir(os.path.join(DATASET_PATH, "train", "NORMAL")))
pneumonia_count = len(os.listdir(os.path.join(DATASET_PATH, "train", "PNEUMONIA")))
total           = normal_count + pneumonia_count
num_classes     = 2

print(f"\n  NORMAL    : {normal_count} images")
print(f"  PNEUMONIA : {pneumonia_count} images")
print(f"  Total     : {total} images")

# ── 2. Compute weights ───────────────────────────────────
weight_normal    = total / (num_classes * normal_count)
weight_pneumonia = total / (num_classes * pneumonia_count)

print(f"\n  Weight for NORMAL    : {weight_normal:.4f}")
print(f"  Weight for PNEUMONIA : {weight_pneumonia:.4f}")

# ── 3. Print ready-to-use format ────────────────────────
print()
print("=" * 50)
print("COPY THIS INTO YOUR TRAINING SCRIPT")
print("=" * 50)
print(f"""
  # Class weights (paste into model training)
  class_weights = {{
      0: {weight_normal:.4f},   # NORMAL
      1: {weight_pneumonia:.4f}    # PNEUMONIA
  }}
""")

# ── 4. Visualise the weights ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Class Imbalance & Weights", fontsize=14, fontweight="bold")

# Left — image counts
bars = axes[0].bar(CLASSES,
                   [normal_count, pneumonia_count],
                   color=["steelblue", "tomato"],
                   edgecolor="black")
axes[0].set_title("Image Counts (Train Set)")
axes[0].set_ylabel("Number of Images")
for bar, val in zip(bars, [normal_count, pneumonia_count]):
    axes[0].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 20,
                 str(val), ha="center", fontweight="bold")

# Right — class weights
bars2 = axes[1].bar(CLASSES,
                    [weight_normal, weight_pneumonia],
                    color=["steelblue", "tomato"],
                    edgecolor="black")
axes[1].set_title("Class Weights Assigned")
axes[1].set_ylabel("Weight Value")
for bar, val in zip(bars2, [weight_normal, weight_pneumonia]):
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.01,
                 f"{val:.4f}", ha="center", fontweight="bold")

plt.tight_layout()
plt.savefig("outputs/class_weights.png", dpi=150)
plt.show()
print("✅ Chart saved to outputs/class_weights.png")