import os
import shutil
from sklearn.model_selection import train_test_split
from collections import Counter

# ── Config ──────────────────────────────────────────────
DATASET_PATH = "chest_xray"
OUTPUT_PATH = "chest_xray_split"
SPLITS = ["train", "val", "test"]
CLASSES = ["NORMAL", "PNEUMONIA"]
RANDOM_STATE = 42

# ── 1. Pool all images ───────────────────────────────────
print("=" * 50)
print("POOLING ALL IMAGES")
print("=" * 50)

all_images = []
all_labels = []

for split in SPLITS:
    for cls in CLASSES:
        folder = os.path.join(DATASET_PATH, split, cls)
        for fname in os.listdir(folder):
            fpath = os.path.join(folder, fname)
            all_images.append(fpath)
            all_labels.append(cls)

print(f"  Total images pooled: {len(all_images)}")
print(f"  Class counts: {Counter(all_labels)}")
print()

# ── 2. Stratified split 70/15/15 ────────────────────────
print("=" * 50)
print("SPLITTING DATA 70 / 15 / 15")
print("=" * 50)

# First split: 70% train, 30% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    all_images, all_labels,
    test_size=0.30,
    stratify=all_labels,
    random_state=RANDOM_STATE
)

# Second split: 50% of temp = 15% val, 15% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=RANDOM_STATE
)

print(f"  Train : {len(X_train)} images — {Counter(y_train)}")
print(f"  Val   : {len(X_val)} images — {Counter(y_val)}")
print(f"  Test  : {len(X_test)} images — {Counter(y_test)}")
print()

# ── 3. Copy images into new folder structure ─────────────
print("=" * 50)
print("COPYING IMAGES TO NEW FOLDERS")
print("=" * 50)

splits_data = {
    "train": (X_train, y_train),
    "val":   (X_val,   y_val),
    "test":  (X_test,  y_test)
}

for split, (images, labels) in splits_data.items():
    for cls in CLASSES:
        folder = os.path.join(OUTPUT_PATH, split, cls)
        os.makedirs(folder, exist_ok=True)

    for fpath, label in zip(images, labels):
        fname = os.path.basename(fpath)
        dest = os.path.join(OUTPUT_PATH, split, label, fname)
        shutil.copy2(fpath, dest)

    print(f"  ✅ {split} done")

# ── 4. Verify final counts ───────────────────────────────
print()
print("=" * 50)
print("FINAL COUNTS VERIFICATION")
print("=" * 50)

for split in SPLITS:
    for cls in CLASSES:
        folder = os.path.join(OUTPUT_PATH, split, cls)
        n = len(os.listdir(folder))
        print(f"  {split}/{cls}: {n} images")
    print()

print("✅ Resplit complete! New dataset saved to:", OUTPUT_PATH)