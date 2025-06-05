# TASK-7

SVM Classifier on Breast Cancer Dataset

Introduction:

This project implements a Support Vector Machine (SVM) classifier to detect breast cancer (malignant vs. benign) using the scikit-learn dataset. It supports both linear and RBF kernels, includes hyperparameter tuning, and visualizes model decision boundaries and feature importance.

CODE WORK FLOW:

1.Initialize Class: Sets up scaler, PCA, and SVM models.

2.Load Dataset: Uses load_breast_cancer() and prints basic info.

3.Split Data: Into train and test sets (80/20 stratified split).

4.Scale Features: Applies StandardScaler to normalize inputs.

5.Train Models:

*   Trains Linear SVM

*   Trains RBF SVM

6.Evaluate Models:

*   Predicts on train/test

*   Prints accuracy, classification report

*   Performs 5-fold cross-validation

7.Hyperparameter Tuning:

*   Uses GridSearchCV for C and gamma

*   Finds best estimator

8.Evaluate Best Model:

*   Tests best model on test data

*   Shows accuracy and parameters

9.Visualize Decision Boundaries:

*   Transforms data to 2D via PCA

*   Plots decision boundaries for linear and RBF SVM

10.Confusion Matrix:

*   For best model using seaborn heatmap

11.Feature Importance (if linear):

*   Displays top 10 important features based on coefficient
