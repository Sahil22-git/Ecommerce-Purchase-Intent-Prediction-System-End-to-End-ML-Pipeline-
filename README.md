# 📌 E-Commerce Purchase Intent Prediction System (End-to-End ML Pipeline)

## 📖 Project Overview

This project builds an end-to-end Machine Learning system to predict whether an e-commerce website session will result in a purchase.

The objective is to identify high-intent user sessions early so that businesses can trigger targeted marketing actions such as discount offers, chatbot assistance, or retargeting campaigns.

The model is trained on real session-level behavioral data and optimized using business-driven threshold tuning.

---

## 🎯 Business Problem

In most e-commerce platforms, only a small percentage of sessions result in a purchase.

- 15.47% sessions resulted in purchase  
- 84.53% sessions did not convert  

Predicting purchase intent enables:

- Targeted discount campaigns  
- Conversion rate optimization  
- Improved marketing ROI  
- Personalized user engagement  
- Better revenue forecasting  

---

## 📊 Dataset Information

- **Source:** Kaggle – Online Shoppers Purchasing Intention Dataset  
- **Total Records:** ~12,000 sessions  
- **Target Variable:** `Revenue` (True / False)  
- **Class Imbalance:** 15.47% positive class  

Each row represents a user session containing:

- Product page visits  
- Time spent on pages  
- Bounce rate  
- Exit rate  
- Page monetary value  
- Visitor type  
- Weekend flag  
- Traffic source  

---

## 🧠 Feature Engineering

Additional behavioral interaction features were created:

- `engagement_score = ProductRelated × ProductRelated_Duration`
- `value_intensity = PageValues × Administrative_Duration`
- `returning_user_flag` (encoded visitor type)

These engineered features improved predictive performance.

---

## ⚙️ Modeling Approach

### 1️⃣ Data Preprocessing
- Missing value verification  
- Standard scaling of numerical features  
- One-hot encoding of categorical variables  
- Stratified train-test split (80/20)  

### 2️⃣ Class Imbalance Handling
- Applied `class_weight="balanced"`  

### 3️⃣ Model Comparison
- Logistic Regression (baseline)  
- Random Forest (non-linear model)  

---

## 📈 Model Performance

| Model                 | ROC-AUC | Recall (Purchase) |
|-----------------------|---------|-------------------|
| Logistic Regression   | 0.896   | 0.75              |
| Random Forest         | 0.922   | 0.77              |

---

## 🎯 Threshold Optimization

Instead of using the default 0.5 classification threshold, tuning was applied to align with business objectives.

At threshold = **0.35**:

- Recall improved from 0.77 → **0.85**
- Increased detection of high-intent buyers  
- Slight precision tradeoff (expected behavior)  

This demonstrates business-driven ML decision making.

---

## 🔍 Feature Importance Insights

Top predictors influencing purchase probability:

1. **PageValues** (38% importance)  
2. **value_intensity** (14% importance)  
3. **ExitRates** (6% importance)  

These insights align with real-world user behavior patterns.

---

## 🏗 End-to-End System Pipeline

User Session Data  
→ Feature Engineering  
→ Preprocessing Pipeline  
→ Random Forest Model  
→ Probability Output  
→ Business Threshold Decision  

The final pipeline was serialized using `joblib` for production readiness.

---

## 🛠 Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- XGBoost  
- Matplotlib / Seaborn  
- VS Code  
- Jupyter Notebook  

---

## 📂 Project Structure

ecommerce_purchase_prediction/
│
├── data/
├── notebooks/
├── models/
│ └── purchase_model.pkl
├── src/
└── README.md


---

## 🚀 Future Improvements

- SHAP-based explainability  
- Advanced hyperparameter optimization  
- Streamlit deployment interface  
- Real-time API integration  
- A/B testing simulation  

---

## 📌 Final Outcome

An end-to-end purchase intent prediction system achieving:

- **ROC-AUC:** 0.92  
- **Optimized Recall:** 0.85  
- Production-ready ML pipeline  
- Business-aligned decision threshold  

This project demonstrates applied Machine Learning with practical business reasoning.
