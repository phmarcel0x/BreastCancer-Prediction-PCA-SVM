# Breast Cancer Prediction with PCA and SVM

This project explores how machine learning techniques can be applied to real-world medical datasets, particularly using **Principal Component Analysis (PCA)** for feature reduction and **Support Vector Machines (SVM)** for classification. The dataset I’m using is the well-known **Wisconsin Diagnostic Breast Cancer (WDBC)** [dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data).

## Project Overview

This project focuses on detecting whether a tumor is benign or malignant based on 30 features extracted from breast cancer diagnostic data. The idea is to simplify the data using PCA and then classify it using SVM models with both linear and radial basis function (RBF) kernels. By doing this, I aim to make predictions more efficient without losing critical information.

Breakdown of the steps:
- **Data Visualization**: I begin by plotting the data to understand how the different features relate to each other.
- **Dimensionality Reduction**: PCA is used to reduce the number of features while preserving as much variance in the data as possible.
- **Classification**: I train several SVM models with different kernels to classify the tumors as either benign or malignant.
- **Evaluation**: I assess the models' performance using metrics such as accuracy, confusion matrices, and classification reports to determine the best-performing model.

### Results

Through PCA, I reduced the complexity of the dataset while keeping most of the important variance intact. The SVM models performed well, with the RBF kernel proving to be particularly effective for this dataset. The classification results gave strong accuracy rates and insightful confusion matrices, showing which cases were most difficult to classify.

### Tech Stack

![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white&style=for-the-badge)
![NumPy](https://img.shields.io/badge/-NumPy-013243?logo=numpy&logoColor=white&style=for-the-badge)
![Pandas](https://img.shields.io/badge/-Pandas-150458?logo=pandas&logoColor=white&style=for-the-badge)
![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557C?logo=matplotlib&logoColor=white&style=for-the-badge)
![Scikit-Learn](https://img.shields.io/badge/-Scikit%20Learn-F7931E?logo=scikit-learn&logoColor=white&style=for-the-badge)
![Jupyter](https://img.shields.io/badge/-Jupyter-F37626?logo=jupyter&logoColor=white&style=for-the-badge)


#### Have a great day :)

