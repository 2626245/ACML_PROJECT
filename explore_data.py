import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

# ── Config ──────────────────────────────────────────────
DATASET_PATH = "chest_xray"
SPLITS = ["train", "val", "test"]
CLASSES = ["NORMAL", "PNEUMONIA"]
OUTPUT_PATH = "outputs"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# ── 1. Count images per split per class ─────────────────
print("=" * 50)
print("IMAGE COUNTS PER SPLIT")
print("=" * 50)

counts = {}
for split in SPLITS:
    counts[split] = {}
    for cls in CLASSES:
        folder = os.path.join(DATASET_PATH, split, cls)
        n = len(os.listdir(folder))
        counts[split][cls] = n
        print(f"  {split}/{cls}: {n} images")
    print()

# ── 2. Check for corrupted images ───────────────────────
print("=" * 50)
print("CHECKING FOR CORRUPTED IMAGES")
print("=" * 50)

corrupted = []
for split in SPLITS:
    for cls in CLASSES:
        folder = os.path.join(DATASET_PATH, split, cls)
        for fname in os.listdir(folder):
            fpath = os.path.join(folder, fname)
            try:
                img = Image.open(fpath)
                img.verify()
            except Exception as e:
                corrupted.append(fpath)
                print(f"  Corrupted: {fpath} — {e}")

if not corrupted:
    print("  ✅ No corrupted images found!")
print()

# ── 3. Check image sizes and modes ──────────────────────
print("=" * 50)
print("IMAGE SIZES AND MODES (sample of 5 per class)")
print("=" * 50)

for split in ["train"]:
    for cls in CLASSES:
        folder = os.path.join(DATASET_PATH, split, cls)
        files = os.listdir(folder)[:5]
        print(f"\n  {split}/{cls}:")
        for fname in files:
            fpath = os.path.join(folder, fname)
            img = Image.open(fpath)
            print(f"    {fname} — size: {img.size}, mode: {img.mode}")

# ── 4. Plot class distribution ───────────────────────────
print("\n")
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Class Distribution Per Split", fontsize=14, fontweight="bold")

for i, split in enumerate(SPLITS):
    normal = counts[split]["NORMAL"]
    pneumonia = counts[split]["PNEUMONIA"]
    bars = axes[i].bar(CLASSES, [normal, pneumonia],
                       color=["steelblue", "tomato"], edgecolor="black")
    axes[i].set_title(split.upper())
    axes[i].set_ylabel("Number of Images")
    for bar, val in zip(bars, [normal, pneumonia]):
        axes[i].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 10,
                     str(val), ha="center", fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "class_distribution.png"), dpi=150)
plt.show()
print("✅ Chart saved to outputs/class_distribution.png")

# ── 5. Show sample images ────────────────────────────────
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle("Sample Images — NORMAL (top) vs PNEUMONIA (bottom)",
             fontsize=13, fontweight="bold")

for col, cls in enumerate(CLASSES):
    folder = os.path.join(DATASET_PATH, "train", cls)
    files = os.listdir(folder)[:5]
    row = CLASSES.index(cls)
    for col_idx, fname in enumerate(files):
        fpath = os.path.join(folder, fname)
        img = Image.open(fpath).convert("RGB")
        img = img.resize((224, 224))
        axes[row][col_idx].imshow(img)
        axes[row][col_idx].axis("off")
        axes[row][col_idx].set_title(cls, fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_PATH, "sample_images.png"), dpi=150)
plt.show()
print("✅ Sample images saved to outputs/sample_images.png")