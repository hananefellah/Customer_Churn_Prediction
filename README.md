# 📉 Customer Churn Prediction — Telecom
### Binary Classification | SMOTE | 5-Fold Cross-Validation | SHAP Explainability | Ensemble Methods

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![AdaBoost](https://img.shields.io/badge/Best_Model-AdaBoost-orange)
![SMOTE](https://img.shields.io/badge/Imbalance-SMOTE-purple)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-green)
![CV](https://img.shields.io/badge/Validation-5--Fold_CV-red)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## 📌 Problem Statement

Customer churn — when a customer stops using a service — is one of the most costly problems in the telecom industry. Acquiring a new customer costs 5–10x more than retaining an existing one.

This project builds a machine learning system that predicts which customers are likely to cancel their subscription, enabling the business to take proactive retention actions before churn occurs.

> **Dataset:** IBM Telco Customer Churn — 7,043 customers, 21 features  
> **Domain:** Customer Analytics / Business Intelligence  
> **Target:** Binary — Churn: Yes / No  
> **Key challenge:** Class imbalance — only 26.5% of customers churned

---

## 📊 Results

### Cross-Validation (5-Fold Stratified — on SMOTE-balanced training set)

| Model | CV Accuracy | CV F1 (Churn) | CV Std |
|-------|:---:|:---:|:---:|
| Logistic Regression | 0.7571 | 0.7728 | ±0.0077 |
| KNN | 0.7663 | 0.7875 | ±0.0049 |
| Decision Tree | 0.7902 | 0.7894 | ±0.0082 |
| SVM | 0.7923 | 0.8031 | ±0.0045 |
| Random Forest | 0.8052 | 0.8110 | ±0.0054 |
| AdaBoost | 0.8090 | 0.8181 | ±0.0059 |
| **Gradient Boosting** | **0.8354** | **0.8376** | **±0.0070** |

### Test Set Performance

| Model | Test Accuracy | F1 (Churn) | Recall (Churn) | Precision (Churn) |
|-------|:---:|:---:|:---:|:---:|
| Decision Tree | 0.725 | 0.48 | 0.48 | 0.49 |
| KNN | 0.742 | 0.50 | 0.49 | 0.52 |
| Logistic Regression | 0.763 | 0.50 | 0.44 | 0.57 |
| SVM | 0.768 | 0.53 | 0.49 | 0.58 |
| Gradient Boosting | 0.771 | 0.54 | 0.51 | 0.58 |
| Voting Classifier | 0.773 | 0.54 | 0.51 | 0.58 |
| Random Forest | 0.774 | 0.52 | 0.46 | 0.60 |
| **AdaBoost** ✅ | **0.778** | **0.56** | **0.53** | **0.59** |

> **Best test model: AdaBoost — F1 Churn 0.56, Recall 0.53**  
> **Best CV model: Gradient Boosting — CV F1 0.8376 (most stable across folds)**  
> AdaBoost catches **53% of actual churners** before they leave

---


## 🔍 SHAP Explainability — Top Churn Drivers

| Rank | Feature | Mean SHAP | Business Action |
|------|---------|:---:|---|
| 1 | **StreamingMovies** | 1.076 | Review streaming bundle pricing and satisfaction |
| 2 | **InternetService** | 0.307 | Investigate fiber optic dissatisfaction |
| 3 | **MultipleLines** | 0.276 | Review multi-line plan value proposition |
| 4 | **PaperlessBilling** | 0.244 | Target paperless customers with retention offers |
| 5 | **OnlineSecurity** | 0.187 | Promote security bundles as retention tools |
| 6 | **Contract** | 0.147 | Offer discounts to upgrade month-to-month to annual |
| 7 | **DeviceProtection** | 0.140 | Bundle device protection with other services |
| 8 | **Partner** | 0.124 | Target single customers with household plan offers |

> Note: SHAP was computed on the SMOTE-rebalanced dataset. Contract type remains the #1 driver in raw EDA — ~89% of churners were on month-to-month contracts.

---

## 🔑 Key EDA Findings

| Feature | Finding |
|---|---|
| **Contract type** | ~89% of churners on month-to-month — strongest single predictor |
| **Tenure** | Churners median tenure ~10 months vs ~38 months for loyal customers |
| **Internet Service** | 69% of all churners used fiber optic despite being most popular |
| **Monthly Charges** | Churned customers cluster at $60-100/month; loyal customers peak at ~$20 |
| **Online Security** | Customers without security bundle churn at nearly double the rate |
| **Electronic Check** | ~1,071 churners paid by electronic check — highest of all payment methods |

---

## 🧠 Key Technical Decisions

### ✅ SMOTE for Class Imbalance
With only 26.5% positive class, a naive model predicting "No churn" achieves 73.5% accuracy while being completely useless. SMOTE creates synthetic churned-customer samples until both classes are balanced in the training set — **the test set retains the real imbalance** for a realistic performance estimate.

### ✅ 5-Fold Stratified Cross-Validation
All 7 models validated with stratified CV before final test evaluation. Stratification preserves the class ratio in each fold. Low standard deviations (all < 0.01) confirm results are stable — not driven by lucky splits.

### ✅ F1 Score as Primary Metric
Accuracy is misleading on imbalanced data. F1 for the churn class balances precision and recall — critical because **missing a churner (false negative) is far more costly** than a false alarm.

### ✅ Voting Classifier Ensemble
Combines Gradient Boosting + Logistic Regression + AdaBoost using soft voting (averaging probabilities). Diverse models reduce variance and the ensemble is more robust in production than any single model.

### ✅ SHAP Explainability
Feature importance via SHAP values reveals not just which features matter globally, but how each feature value affects each individual prediction — turning the model into actionable business intelligence.

---

## 📁 Project Structure

```
Customer_Churn_Prediction/
├── Customer_Churn_Prediction_FINAL_FH.ipynb   # Main notebook (fully executed)
├── WA_Fn-UseC_-Telco-Customer-Churn.csv       # Dataset (IBM Telco)
└── plots/
    ├── 00_churn_distribution.png              # Class imbalance visualization
    ├── 01_numerical_distributions.png         # Tenure, charges distributions
    ├── 02_cv_results.png                      # Cross-validation bar chart
    ├── 03_model_comparison.png                # All 7 models benchmark
    ├── 04_roc_curves.png                      # ROC curves — all models
    ├── 05_shap_summary.png                    # SHAP beeswarm plot
    └── 06_shap_importance.png                 # Mean SHAP feature ranking
```

---

## ⚙️ How to Run

```bash
# 1. Clone
git clone https://github.com/hananefellah/Customer_Churn_Prediction
cd Customer_Churn_Prediction

# 2. Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn plotly missingno shap

# 3. Run
jupyter notebook Customer_Churn_Prediction_FINAL_FH.ipynb
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.12 | Core language |
| Pandas / NumPy | Data manipulation |
| Scikit-learn | ML models, preprocessing, CV, evaluation |
| imbalanced-learn | SMOTE oversampling |
| SHAP | Model explainability |
| Plotly | Interactive visualizations |
| Matplotlib / Seaborn | Static visualizations |
| missingno | Missing value visualization |

---

## 💼 Business Recommendations

1. **Contract upgrade incentives** — ~89% of churners on month-to-month; offer discounts to switch to annual
2. **Year-1 onboarding program** — median churner leaves after only 10 months; intensive support in year 1 is critical
3. **Investigate fiber optic quality** — 69% of churners used fiber optic despite being the most popular service
4. **Promote streaming + security bundles** — SHAP confirms these features drive churn decisions most strongly
5. **Incentivize automatic payment** — electronic check users are the highest-risk payment segment

---

## 🚀 Future Work

- [ ] Hyperparameter tuning on Gradient Boosting (best CV model, F1=0.8376)
- [ ] Deploy as FastAPI endpoint for real-time churn scoring
- [ ] Build a customer risk dashboard for the retention team
- [ ] Incorporate customer service call logs and NPS scores for richer features

---

## 📄 License
*MIT License*

## 👩‍💻 Author

**Fellah Hanane** : Data Scientist
🌐 [GitHub](https://github.com/hananefellah) 

📧 Email: hananefellah35@gmail.com · Open to Remote Roles
