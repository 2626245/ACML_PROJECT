# ACML Project — Pneumonia Detection from Chest X-Rays
### Adaptive Computation and Machine Learning (COMS 4030A / COMS 7047A)

---

## 👥 Team Members
| Name | Student Number |
|---|---|
| Paballo Moloantoa | 2451669 |
| Lanet | [Student Number] |
| Aphiwe | [Student Number] |
| Amogelang Moenyane | 2708897 |

---

## 📌 Project Overview
This project uses deep learning to classify chest X-ray images as either
**NORMAL** or **PNEUMONIA**. The dataset was obtained from Kaggle:

🔗 https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

---

## 📁 Project Structure

acml_project/
├── explore_data.py       # Step 1 - Data exploration & sanity check
├── resplit_data.py       # Step 2 - Resplit data 70/15/15
├── preprocess.py         # Step 3 - Resize & convert images
├── class_weights.py      # Step 4 - Compute class weights
├── outputs/              # Generated charts & graphs
└── README.md             # You are here


> ⚠️ Image folders (the dataset) are NOT included in this repo (too large for GitHub).
> Follow the setup instructions below to generate them yourself.

---

## ⚙️ Setup Instructions

### 1. Clone the Repo
```bash
git clone https://github.com/2626245/ACML_PROJECT 
cd acml_project
```

### 2. Install Required Libraries
```bash
pip install numpy matplotlib scikit-learn Pillow tqdm
```

### 3. Download the Dataset
- Go to: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- Download and unzip the dataset (delete the second chest_xray folder inside and the MACOSX folder. leave the 3 folders, test train , val)
- Place the inner `chest_xray/` folder inside `acml_project/`
- Delete the `__MACOSX` folder if it exists

Your structure should look like:
acml_project/
├── chest_xray/
│   ├── train/
│   ├── val/
│   └── test/
└── ...(and the .py files)

### 4. Run the Scripts in Order
```bash
python explore_data.py      # Step 1 - Explore & inspect
python resplit_data.py      # Step 2 - Resplit the data
python preprocess.py        # Step 3 - Preprocess images
python class_weights.py     # Step 4 - Compute class weights
```

---

## 📊 Dataset Summary

| Split | NORMAL | PNEUMONIA | Total |
|---|---|---|---|
| Train | 1,108 | 2,991 | 4,099 |
| Val | 237 | 641 | 878 |
| Test | 238 | 641 | 879 |
| **Total** | **1,583** | **4,273** | **5,856** |

---

## ⚖️ Class Weights
Used during model training to handle class imbalance:
```python
class_weights = {
    0: 1.8497,   # NORMAL
    1: 0.6852    # PNEUMONIA
}
```
Pass this into `model.fit(..., class_weight=class_weights)`

---

## 🖼️ Model Input Specification
| Property | Value |
|---|---|
| Image size | 224 × 224 |
| Colour mode | RGB (3 channels) |
| Dataset path | `chest_xray_processed/` |
| Normalisation | Divide by 255 (or ImageNet mean/std) |

---

## 📈 Output Charts
After running all scripts, the `outputs/` folder will contain:

| File | Description |
|---|---|
| `class_distribution.png` | Class counts per split |
| `sample_images.png` | Sample X-ray images |
| `before_after_preprocessing.png` | Preprocessing comparison |
| `class_weights.png` | Imbalance vs weights chart |

---

## 🧠 Model Training
> To be completed by the modelling team.
> Use `chest_xray_processed/` as the dataset.
> Remember to pass `class_weights` into `model.fit()`.

THE CLEANED DATASET IS THE chest_xray_processed. This is the one the models will be using.
