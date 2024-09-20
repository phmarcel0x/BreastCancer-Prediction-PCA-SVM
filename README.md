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
The results of this project demonstrated that SVM with an RBF kernel outperformed the linear kernel models, particularly for small values of the hyperparameter C and gamma.

* **Linear Kernel Results**: The linear models performed well, especially with smaller C values. For instance, with **C=2**, the accuracy was **96%**, and precision/recall metrics were high for both benign and malignant classifications.

* **RBF Kernel Results**: The best performance came from the **RBF kernel** with **C=2** and **gamma=0.01**, which achieved **98% accuracy** on the validation set. In contrast, higher C and gamma values led to poorer performance due to overfitting, particularly with **C=52** and **gamma=12** where accuracy dropped to **51%**.

* **Testing on Full Dataset**: The best-performing model from the RBF kernel was tested on the full dataset, resulting in a **70% accuracy** and an insightful confusion matrix showing **114 true positives** but also some misclassifications. However, after applying PCA for feature reduction, the accuracy improved to **78%** with fewer features, indicating that dimensionality reduction can indeed enhance performance by reducing noise in the dataset.

These results indicate that **careful tuning of hyperparameters** and **feature reduction** via PCA can significantly improve model performance, particularly for complex datasets like breast cancer diagnosis.

### Potential Future Improvements
While 78% accuracy shows progress, there's room for improvement in medical diagnostics. Potential avenues for enhancing performance include:

- **Deep Learning**: Exploring neural networks, particularly Convolutional Neural Networks (CNNs), which have shown promise in medical image analysis.
- **Ensemble Methods**: Combining multiple models (e.g., Random Forests, Gradient Boosting) with our SVM approach could potentially improve overall accuracy.
- **Cross-Validation**: Implementing k-fold cross-validation could provide a more reliable estimate of model performance and help in fine-tuning hyperparameters.

### Tech Stack

![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white&style=for-the-badge)
![NumPy](https://img.shields.io/badge/-NumPy-013243?logo=numpy&logoColor=white&style=for-the-badge)
![Pandas](https://img.shields.io/badge/-Pandas-150458?logo=pandas&logoColor=white&style=for-the-badge)
![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557C?logo=matplotlib&logoColor=white&style=for-the-badge)
![Scikit-Learn](https://img.shields.io/badge/-Scikit%20Learn-F7931E?logo=scikit-learn&logoColor=white&style=for-the-badge)
![Jupyter](https://img.shields.io/badge/-Jupyter-F37626?logo=jupyter&logoColor=white&style=for-the-badge)


#### Have a great day :)

