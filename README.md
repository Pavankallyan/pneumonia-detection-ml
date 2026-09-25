# Pneumonia Detection from Chest X-rays

Master's Final Project — DATA 6250 Machine Learning, Wentworth Institute of Technology

## Overview
Medical image classification system that detects pneumonia from chest X-ray images,
using both classical machine learning and deep learning approaches.

## Results

| Model | Accuracy | AUC-ROC |
|---|---|---|
| Decision Tree | 82.37% | — |
| Random Forest | 82.05% | 0.9499 |
| **ResNet50 CNN** | **91.67%** | **0.9627** |

Best model: ResNet50 fine-tuned with transfer learning.
Pneumonia recall: **97%** (catches 97% of all pneumonia cases).

## Dataset
- **Name:** Chest X-Ray Images (Pneumonia) — [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) (CC BY 4.0)
- **Size:** 5,216 training images, 624 test images
- **Classes:** NORMAL (1,342) vs PNEUMONIA (3,876)

> Dataset not included (2.3 GB). Download from the Kaggle link and place in `data/raw/chest_xray/chest_xray/`.

## Project structure
```
├── app/            # Streamlit web app (app.py)
├── models/         # config.json, results.json, trained weights
├── notebooks/      # eda, preprocessing, baseline_cnn, transfer_learning, evaluation, classical_ml
├── src/
└── requirements.txt
```

## How to run
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip -d data/raw/
jupyter notebook          # run notebooks/ in order
cd app && streamlit run app.py
```

## Methods
EDA, data augmentation (flip/rotate/color jitter), PyTorch DataLoaders with class
weighting, scratch CNN baseline, ResNet50 transfer learning + fine-tuning,
Decision Tree, Random Forest with GridSearchCV (36 fits, 3-fold CV), Grad-CAM
explainability, Streamlit deployment. Metrics: accuracy, precision, recall, F1,
AUC-ROC, confusion matrix.

*Random seeds:* classical models use `random_state=42`; PyTorch uses deterministic MPS on Apple Silicon.
