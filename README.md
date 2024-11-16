# CKD Prediction Model

This repository contains a machine learning model for predicting Chronic Kidney Disease (CKD) using a neural network built with Keras. The model is trained on a dataset and deployed using Flask to provide a web interface for predictions.

## Table of Contents
- [Overview](#overview)
- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [License](#license)

## Overview

The project implements a binary classification model to predict the likelihood of kidney disease (CKD) in individuals. The neural network is trained on a dataset with 24 features and evaluates the prediction using various metrics like accuracy, precision, F1 score, and log loss. 

The model is packaged into a Flask web application, allowing users to input their features through a form and get a prediction about whether they have kidney disease or not.

### Dataset Description

The data used in this project was collected over a 2-month period in India and contains 25 features such as red blood cell count, white blood cell count, and other medical indicators. The target variable, `classification`, contains two classes: `ckd` (Chronic Kidney Disease) and `notckd` (no Chronic Kidney Disease). The dataset consists of 400 rows of data.

The dataset can be accessed from Kaggle here: [CKD Disease Dataset on Kaggle](https://www.kaggle.com/datasets/mansoordaku/ckdisease).

## Getting Started

To get started with this project, clone the repository and follow the installation instructions below.

### Clone the repository:
```bash
git clone https://github.com/muqtadanajm/CKD-Prediction.git
cd CKD-Prediction
```
## Prerequisites

Make sure you have Python 3.6 or higher installed. You will also need the following Python packages:

- Flask
- Keras
- TensorFlow
- Scikit-learn
- Pandas
- Numpy

## Installation

1. Install the required dependencies using `pip`:
   ```bash
   pip install -r requirements.txt
2. Ensure you have the necessary data files (e.g., the `kidney_disease.csv` dataset) in the correct directory.

## Usage

To run the application locally, follow these steps:

1. Start the Flask server:
   ```bash
   python app.py
2. Open your browser and go to `http://127.0.0.1:5000` to access the web application.

### Web Application Interface

Once you access the web application, you will see a form where you can input various features related to kidney disease. The input fields include:

- Age
- Blood Pressure
- Specific Gravity
- Albumin
- Sugar
- Red Blood Cells
- Pus Cell
- Polyarithmias
- And other relevant medical indicators

### Predicting CKD

To make a prediction:
1. Fill in the values for the input fields based on the person's medical data.
2. Click the "Predict" button.
3. The application will process the input data using the trained machine learning model and display whether the person is likely to have Chronic Kidney Disease (CKD) or not.

### Example Result

Once the prediction is made, the result will appear as:

- `CKD`: If the person is predicted to have Chronic Kidney Disease.
- `Not CKD`: If the person is predicted to be free of Chronic Kidney Disease.

## Model Training

If you'd like to retrain the model or experiment with the dataset, follow these steps:

1. Ensure that the dataset `kidney_disease.csv` is available in the directory `data/`.
2. Run the `train_model.py` script to retrain the model:
   ```bash
   python model/train_model.py
3. After training, the model will be saved as `model/ckd_model.h5`. This model file will be used by the Flask application to make predictions.

### Model Evaluation

To evaluate the model's performance, the following metrics will be displayed after training:

- **Accuracy**: The percentage of correct predictions.
- **Precision**: The ratio of correctly predicted positive observations to the total predicted positives.
- **F1 Score**: The weighted average of Precision and Recall, useful for imbalanced classes.
- **Confusion Matrix**: A table showing the number of correct and incorrect predictions classified by type.
- **Log Loss**: The logistic loss function, useful for assessing the model's uncertainty in predictions.

These metrics will help you understand how well the model is performing and whether it can make accurate predictions on new data.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

