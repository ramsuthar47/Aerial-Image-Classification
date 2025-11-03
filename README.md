# Aerial Image Classification

This project focuses on classifying aerial images using both Machine Learning (ML) and Deep Learning (DL) models. It demonstrates a complete end-to-end pipeline — from data preparation and feature extraction to model training, evaluation, and comparison.

---

## Models Implemented

### Machine Learning
- Random Forest Classifier  
- Support Vector Machine (SVM)

### Deep Learning
- Custom CNN Model  
- Fine-Tuned CNN (Transfer Learning using pre-trained architectures)

---

## Dataset Information

The dataset used in this project is derived from the **UC Merced Land Use Dataset**, a publicly available aerial image dataset containing 21 land-use classes such as agricultural, beach, forest, runway, and more.  
All images are of size **256×256 pixels** with balanced class distribution.  

Dataset source: [UC Merced Land Use Dataset](http://weegee.vision.ucmerced.edu/datasets/landuse.html)

---

## Workflow Overview

1. **Data Preparation**  
   - Load and preprocess aerial image dataset  
   - Resize, normalize, and split into training and testing sets  
   - Extract structured features for ML & DL pipelines  

2. **Model Training**  
   - Train ML models (Random Forest, SVM) on extracted features  
   - Train DL models (Custom CNN and Fine-Tuned CNN) on raw image data  

3. **Evaluation**  
   - Compute metrics including accuracy, precision, recall, and F1-score  
   - Visualize performance comparisons across models  

4. **Reporting**  
   - Results are saved in the `reports/` directory as `.csv`, `.png`, and `.md` summaries  

---

## Results Summary

| Model | Type | Accuracy |
|--------|------|----------|
| Random Forest | ML | — |
| SVM | ML | — |
| Custom CNN | DL | — |
| Fine-Tuned CNN | DL | — |

(Detailed metrics are available in `reports/model_comparison_summary.md`)

---

## Requirements

Install all required dependencies using:

```bash
pip install -r requirements.txt
