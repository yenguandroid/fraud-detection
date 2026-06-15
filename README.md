# Fraud Detection Project - Task 1

##  Project Overview
This project focuses on data preprocessing and feature engineering for a fraud detection system. The goal of Task 1 is to clean the dataset, engineer meaningful features, and prepare the data for machine learning modeling.

---

## Dataset Description
The dataset contains e-commerce transaction records with the following key features:

- `user_id` – Unique identifier for each user  
- `signup_time` – Time when the user signed up  
- `purchase_time` – Time of transaction  
- `purchase_value` – Amount spent in the transaction  
- `device_id` – Device used for the transaction  
- `source` – Traffic source (SEO, Ads, Direct)  
- `browser` – Browser used  
- `sex` – Gender of user  
- `age` – Age of user  
- `ip_address` – IP address of transaction  
- `class` – Target variable (0 = Not Fraud, 1 = Fraud)

---

## Task 1: Data Preprocessing Steps

### 1. Data Cleaning
- Checked missing values
- Verified data types
- Ensured dataset consistency

---

### 2. Feature Engineering

#### Time-Based Features
- Extracted `hour_of_day` from `purchase_time`
- Extracted `day_of_week`
- Computed `time_since_signup`

---

####  Geolocation Feature
- Converted IP addresses to integer format
- Mapped IP ranges to country
- Created `country` feature

---

#### Behavioral Feature
- Created `transaction_count` feature based on frequency of transactions per entity (e.g., user/device/IP)

---

### 3. Encoding & Transformation
- Encoded categorical variables (source, browser, sex, country)
- Scaled numerical features where necessary

---

## Final Processed Dataset
After preprocessing, the dataset includes:

- Cleaned numerical features
- Encoded categorical variables
- Engineered behavioral and time-based features
- Target variable: `class`

---

## Key Finding: Class Imbalance

The dataset is highly imbalanced:

- Non-fraud (0): ~90.5%
- Fraud (1): ~9.5%

This imbalance must be handled during modeling using:
- Stratified sampling
- Class weighting
- Oversampling techniques (e.g., SMOTE)

---
## Transaction Count Analysis

The engineered `transaction_count` feature shows variation across users/devices:

- Majority of entities appear once
- Some entities appear up to 20 times

This feature helps detect abnormal repeated behavior which may indicate fraud.
# Fraud Detection Project

## Overview

This project develops machine learning models to detect fraudulent transactions using transaction, user, and device information. The workflow includes data preprocessing, feature engineering, class imbalance handling, model training, evaluation, and explainability analysis.

## Project Structure

```text
fraud-detection/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── src/
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── model_utils.py
│
├── models/
│   ├── logistic_regression.pkl
│   └── random_forest.pkl
│
├── requirements.txt
└── README.md
```

## Quickstart

### 1. Clone Repository

```bash
git clone https://github.com/yenguandroid/fraud-detection.git
cd fraud-detection
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

### 3. Activate Environment

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run Notebook

```bash
jupyter notebook
```

Open the notebook inside the `notebooks/` directory and execute the cells sequentially.

## Models Implemented

* Logistic Regression
* Random Forest

## Evaluation Metrics

* Precision-Recall AUC (AUC-PR)
* F1 Score
* Confusion Matrix

## Results

Random Forest achieved the best overall performance and was selected as the final fraud detection model due to its superior AUC-PR and F1 Score.

## Explainability

The project includes:

* Feature Importance Analysis
* SHAP Summary Plot
* SHAP Force Plot
* Business Recommendations

## Model Persistence

Trained models are saved in the `models/` directory using Joblib for future inference and deployment.
