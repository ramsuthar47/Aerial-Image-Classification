Aerial Image Classification using Deep Learning and Machine Learning

This project classifies aerial (satellite) images into multiple land-use and land-cover categories using both Deep Learning (DL) and Machine Learning (ML) techniques. It demonstrates a comparative approach between computational efficiency and model accuracy through CNNs, Transfer Learning, Random Forest, and SVM models.

Overview

The dataset consists of labeled aerial images across 21 classes such as agricultural, airplane, baseballdiamond, beach, buildings, chaparral, denseresidential, forest, freeway, golfcourse, harbor, intersection, mediumresidential, mobilehomepark, overpass, parkinglot, river, runway, sparseresidential, storagetanks, and tenniscourt.

The project includes:

Data preprocessing and feature extraction

Multiple model implementations (CNNs, Transfer Learning, Random Forest, SVM)

Automated model comparison with accuracy reports and visualizations

Project Structure
Aerial Image Classification/
│
├── Dataset/
│   ├── features/
│   ├── processed/
│   └── raw/
│
├── models/
│   ├── *.keras / *.h5   # Deep learning models
│   └── *.pkl            # Machine learning models
│
├── notebooks/
│   ├── model_comparison.ipynb
│   └── reports/
│       ├── model_accuracy_comparison.png
│       ├── model_comparison_results.csv
│       └── model_comparison_summary.md
│
├── results/
│   ├── cnn_predictions.npy
│   └── custom_cnn_model.h5
│
├── src/
│   ├── data_preparation.py
│   ├── feature_extraction.py
│   ├── model_dl_customfit.py
│   ├── model_dl_finetune.py
│   ├── model_ml_rf.py
│   └── model_ml_svm.py
│
├── requirements.txt
└── README.md

Models Implemented
Type	Model	Description
Deep Learning	Custom CNN	Lightweight convolutional model trained from scratch on aerial data
Deep Learning	MobileNetV2 / VGG16 (Fine-Tuned)	Transfer learning with optimized layers for aerial scenes
Machine Learning	Random Forest	Ensemble classifier trained on extracted image features
Machine Learning	SVM (Lightweight)	Optimized Support Vector Machine with feature scaling
Results Summary

All final comparison metrics are automatically generated in
notebooks/reports/model_comparison_summary.md

Example performance snapshot:

Model	Accuracy	Type
Custom CNN	0.85	Deep Learning
MobileNetV2 Fine-Tuned	0.88	Deep Learning
Random Forest	0.72	Machine Learning
SVM (Lightweight)	0.34	Machine Learning

Accuracy plot:
notebooks/reports/model_accuracy_comparison.png

Usage Guide
1. Install dependencies
pip install -r requirements.txt

2. Prepare and process dataset

Ensure the dataset is available under:

Dataset/processed/aerial_data.npz

3. Train or load models

Run any model script from the src/ folder, for example:

python src/model_dl_customfit.py
python src/model_ml_rf.py

4. Compare models

Open the notebook:

notebooks/model_comparison.ipynb


Run all cells to generate accuracy reports and summaries.

Outputs

Model Comparison Report: notebooks/reports/model_comparison_summary.md

Accuracy Plot: notebooks/reports/model_accuracy_comparison.png

Predictions (DL): results/cnn_predictions.npy

Trained Models: stored in models/

Technologies Used

Python 3.12

TensorFlow / Keras

Scikit-learn

OpenCV

NumPy / Pandas / Matplotlib

Seaborn / Tabulate

License

MIT License © 2025 Duke

Author

Duke
Aspiring Computer Science Engineer | UAV and AI Enthusiast
Focused on building intelligent systems that bridge aerial imaging and machine learning.

Highlights

Covers both ML and DL approaches

Real-world dataset with 21 aerial categories

Modular and reproducible code

Suitable for research presentation or portfolio use