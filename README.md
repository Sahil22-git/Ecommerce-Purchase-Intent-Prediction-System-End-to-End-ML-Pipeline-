# Ecommerce-Purchase-Intent-Prediction-System-End-to-End-ML-Pipeline-

📌 E-Commerce Purchase Intent Prediction System
📖 Project Overview

This project builds an end-to-end machine learning system to predict whether a website session will result in a purchase. The goal is to help e-commerce platforms identify high-intent users and trigger targeted marketing strategies such as discounts, recommendations, or retargeting campaigns.

The dataset contains session-level behavioral data including page visits, time spent, bounce rate, and visitor type.

🎯 Business Problem

Only ~15% of website sessions result in a purchase.
Predicting high-purchase intent sessions enables:

Targeted discount campaigns

Improved conversion rates

Optimized marketing spend

Personalized user engagement

📊 Dataset Information

Source: Online Shoppers Purchasing Intention Dataset (Kaggle)

Rows: ~12,000 sessions

Target Variable: Revenue (True/False)

Class Imbalance: 15.47% positive class

🧠 Feature Engineering

The following engineered features were added:

engagement_score = ProductRelated × ProductRelated_Duration

value_intensity = PageValues × Administrative_Duration

returning_user flag (encoded visitor type)

These features enhanced behavioral signal detection.

⚙️ Model Development
Step 1: Data Preprocessing

Missing value check

Standard scaling for numerical features

One-hot encoding for categorical features

Stratified train-test split

Step 2: Handling Class Imbalance

Applied class_weight="balanced"

Step 3: Model Comparison

Logistic Regression (baseline)

Random Forest (non-linear model)

📈 Model Performance
Model	ROC-AUC	Recall (Purchase)
Logistic Regression	0.896	0.75
Random Forest	0.922	0.77

After threshold optimization (0.35):

Recall improved to 0.85

Increased high-intent session detection

🔍 Feature Importance Insights

Top predictors:

PageValues (38%)

value_intensity (14%)

ExitRates (6%)

These insights align with expected user purchasing behavior.

🏗 System Pipeline

User Session Data
→ Feature Engineering
→ Preprocessing Pipeline
→ Random Forest Model
→ Probability Output
→ Threshold Decision

The final model was serialized using joblib for deployment readiness.

🛠 Technologies Used

Python

Pandas

NumPy

Scikit-learn

XGBoost

Matplotlib / Seaborn

VS Code + Jupyter Notebook

🚀 Future Improvements

Hyperparameter tuning using advanced search

SHAP explainability

Streamlit deployment

Real-time API integration

📂 Project Structure
ecommerce_purchase_prediction/
│
├── data/
├── notebooks/
├── models/
├── src/
└── README.md
