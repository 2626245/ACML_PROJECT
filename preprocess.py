import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

# ── Config ──────────────────────────────────────────────
INPUT_PATH  = "chest_xray_split"
OUTPUT_PATH = "chest_xray_processed"
SPLITS      = ["train", "val", "test"]
CLASSES     = ["NORMAL", "PNEUMONIA"]
IMG_SIZE    = (224, 224)

# ── 1. Preprocess & save all images ─────────────────────
print("=" * 50)
print("PREPROCESSING IMAGES")
print("=" * 50)

for split in SPLITS:
    for cls in CLASSES:
        in_folder  = os.path.join(INPUT_PATH, split, cls)
        out_folder = os.path.join(OUTPUT_PATH, split, cls)
        os.makedirs(out_folder, exist_ok=True)

        files = os.listdir(in_folder)
        print(f"\n  Processing {split}/{cls} ({len(files)} images)...")

        for fname in tqdm(files, desc=f"  {split}/{cls}"):
            in_path  = os.path.join(in_folder, fname)
            out_path = os.path.join(out_folder, fname)

            img = Image.open(in_path)
            img = img.convert("RGB")       # grayscale → RGB
            img = img.resize(IMG_SIZE,     # resize to 224x224
                             Image.LANCZOS)
            img.save(out_path)

# ── 2. Verify processed images ───────────────────────────
print("\n")
print("=" * 50)
print("VERIFYING PROCESSED IMAGES")
print("=" * 50)

all_good = True
for split in SPLITS:
    for cls in CLASSES:
        folder = os.path.join(OUTPUT_PATH, split, cls)
        files  = os.listdir(folder)
        # Check a sample of 10 images
        for fname in files[:10]:
            fpath = os.path.join(folder, fname)
            img   = Image.open(fpath)
            if img.size != IMG_SIZE or img.mode != "RGB":
                print(f"  ❌ Problem: {fpath} — size:{img.size} mode:{img.mode}")
                all_good = False

if all_good:
    print("  ✅ All sampled images are 224x224 RGB!")

# ── 3. Print final counts ────────────────────────────────
print()
print("=" * 50)
print("FINAL PROCESSED COUNTS")
print("=" * 50)

for split in SPLITS:
    for cls in CLASSES:
        folder = os.path.join(OUTPUT_PATH, split, cls)
        n      = len(os.listdir(folder))
        print(f"  {split}/{cls}: {n} images")
    print()

# ── 4. Plot before vs after sample ──────────────────────
print("Generating before/after comparison chart...")

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle("Before vs After Preprocessing",
             fontsize=14, fontweight="bold")

for col, cls in enumerate(CLASSES):
    # Before
    in_folder  = os.path.join(INPUT_PATH,  "train", cls)
    out_folder = os.path.join(OUTPUT_PATH, "train", cls)
    files      = os.listdir(in_folder)[:2]

    for row, fname in enumerate(files):
        # Original
        orig     = Image.open(os.path.join(in_folder, fname))
        axes[row][col * 2].imshow(orig, cmap="gray")
        axes[row][col * 2].set_title(
            f"ORIGINAL\n{cls}\n{orig.size} {orig.mode}", fontsize=8)
        axes[row][col * 2].axis("off")

        # Processed
        proc     = Image.open(os.path.join(out_folder, fname))
        axes[row][col * 2 + 1].imshow(proc)
        axes[row][col * 2 + 1].set_title(
            f"PROCESSED\n{cls}\n{proc.size} {proc.mode}", fontsize=8)
        axes[row][col * 2 + 1].axis("off")

plt.tight_layout()
plt.savefig("outputs/before_after_preprocessing.png", dpi=150)
plt.show()
print("✅ Chart saved to outputs/before_after_preprocessing.png")
print("\n✅ Preprocessing complete! Ready at:", OUTPUT_PATH)