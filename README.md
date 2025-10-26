# Machine Learning Project with Scikit-Learn

This repository is for the projects of the professional machine learning course with Scikit-Learn.

## Projects

### 001_pca
This project implements Principal Component Analysis (PCA) and Incremental PCA (IPCA) for dimensionality reduction on the heart disease dataset. It compares the performance of a logistic regression model trained on the PCA-transformed data versus the IPCA-transformed data.

### 002_kpca
This project applies Kernel PCA (KPCA) with a polynomial kernel to the heart disease dataset. It visualizes the transformed data and evaluates the performance of a logistic regression model.

### 003_regularization
This project explores different regularization techniques for linear regression, including Lasso, Ridge, and ElasticNet. It uses the world happiness dataset to predict happiness scores and compares the mean squared error and model coefficients for each regularization method.

### 004_robust
This project demonstrates how to use robust regression models (RANSAC, Huber Regressor, and SVR) to handle outliers in a corrupted version of the world happiness dataset. It compares the mean squared error of each model.

## Project Structure

- `data/`: Contains the datasets for the projects.
  - `raw/`: The original, immutable data.
  - `processed/`: The final, canonical data sets for modeling.
- `notebooks/`: Jupyter notebooks for exploration and analysis.
- `projects/`: Individual projects from the course.
- `references/`: Documentation and reference materials for the datasets.
- `src/`: Source code for the project.
  - `__init__.py`: Makes `src` a Python module.
  - `data_processing.py`: Scripts to process the data.
  - `train.py`: Scripts to train the models.
  - `predict.py`: Scripts to make predictions with the trained models.
- `reports/`: Generated analysis, reports, and figures.
- `tests/`: Unit tests for the source code.
  - `test_data.py`: Tests for the data processing scripts.
  - `test_model.py`: Tests for the model training and prediction scripts.
- `requirements.txt`: The requirements file for reproducing the analysis environment.
