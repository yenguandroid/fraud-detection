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
