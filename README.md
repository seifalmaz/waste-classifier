# ♻️ Waste Classifier — Intelligent Recycling System

An Intelligent Waste Classification and Recycling Assistance System built using Deep Learning.  
This project classifies waste images into 5 categories to support smart recycling and environmental sustainability.

---

## 📌 Project Overview

This project is part of a **Neural Networks and Deep Learning  course**.

We aim to build and compare multiple deep learning models for classifying waste images into:

- 🟦 Plastic  
- 📄 Paper  
- 🪙 Metal  
- 🍾 Glass  
- 🍃 Organic  

The system explores traditional CNNs, improved architectures, transfer learning, and Vision Transformers.

---

## 👥 Team Information

- Team Size: 5 members  
- Development Style: Branch-based collaboration  
- Repository: https://github.com/seifalmaz/waste-classifier  

### Branch Strategy
- `main` → stable, tested code only  
- Each member works on a separate feature branch  
- Changes are merged via Pull Requests after review  

---

## 📁 Project Structure

waste_classifier/
├── data/
│   ├── raw/                     # Original cleaned dataset (6,166 images)
│   └── processed/               # Train/Val/Test split
│
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 01b_data_collection_ddg.ipynb
│   ├── 02_preprocessing_eda.ipynb
│   ├── 03_baseline_cnn.ipynb
│   ├── 04_improved_cnn.ipynb
│   ├── 05_transfer_learning.ipynb
│   ├── 06_vit.ipynb
│   └── 07_evaluation_comparison.ipynb
│
├── models/
├── results/
├── reports/
├── requirements.txt
└── .gitignore

---

## 📊 Dataset Information

Total Images: 6,166

Plastic: 1033  
Paper: 1285  
Metal: 1102  
Glass: 1440  
Organic: 1306  

---

## 🧹 Data Processing Steps

- Web scraping (Bing + DuckDuckGo)
- Kaggle dataset merging
- Duplicate removal (imagehash)
- Manual cleaning
- Train/Val/Test split (70/15/15)
- Seed = 42

---

## ⚙️ Technical Setup

- Framework: PyTorch  
- Image Size: 224×224  
- RGB images  
- Normalization: ImageNet stats  
- Batch Size: 32  
- Optimizer: Adam  
- Device: CUDA  

---

## 🤖 Models

1. Baseline CNN  
2. Improved CNN (BN + Dropout + Augmentation)  
3. Transfer Learning (ResNet50, EfficientNetB3)  
4. Vision Transformer (ViT)  

---

## 📈 Evaluation

- Accuracy, Precision, Recall, F1-score  
- Confusion matrices  
- Training curves comparison  
- Grad-CAM visualizations  
- ViT attention maps  

---

## 🚀 Installation

pip install -r requirements.txt
