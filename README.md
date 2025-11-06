# Customer Churn Prediction

## Overview
This project analyzes telecom customer data to identify patterns of churn and builds predictive models to forecast which customers are likely to leave. The goal is to help businesses implement targeted retention strategies and improve profitability.

## Objectives
- Explore customer churn patterns and distributions
- Analyze relationships between customer demographics, services, and churn
- Build and evaluate multiple machine learning models to predict churn
- Derive actionable insights for customer retention strategies

---

## Tools & Libraries Used

### Data Manipulation & Visualization
- **Pandas** – Data manipulation and analysis  
- **NumPy** – Numerical computations  
- **Missingno** – Visualize missing data  
- **Matplotlib & Seaborn** – Static data visualization  
- **Plotly** – Interactive and dynamic visualizations  

### Data Preprocessing
- **Scikit-learn** – Label encoding, train-test splitting, scaling  
- **StandardScaler, LabelEncoder** – Standardization and categorical encoding  
- **SMOTE (imbalanced-learn)** – Balancing imbalanced datasets  

---

### Machine Learning Models
- **Logistic Regression**  
- **Decision Tree Classifier**  
- **Random Forest Classifier**  
- **K-Nearest Neighbors (KNN)**  
- **Support Vector Classifier (SVC)**  
- **Gaussian Naive Bayes**  
- **MLP Classifier (Neural Networks)**  
- **AdaBoost, Gradient Boosting, Extra Trees**  
- **XGBoost, CatBoost**  

---

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score  
- Confusion Matrix  
- ROC Curve  

---

## Dataset
The dataset contains customer demographic information, service subscriptions, account details, and churn labels.  

---

## Key Steps
1. **Data Cleaning & Preprocessing:**  
   - Handled missing values (`TotalCharges`)  
   - Encoded categorical variables  
   - Scaled numerical features  
   - Balanced dataset using SMOTE  

2. **Exploratory Data Analysis (EDA):**  
   - Class distribution of churn  
   - Feature correlations with churn  
   - Patterns in services and demographics  

3. **Model Building & Evaluation:**  
   - Trained multiple classifiers  
   - Evaluated models using accuracy, precision, recall, and F1-score  
   - Selected best-performing model for predictions  

## Results
- Visualizations of churn distribution and key feature relationships  
- Classification reports and performance comparison of multiple models  
- Insights on high-risk customers and most profitable services  

---

## 📬 Contact  
Created by FELLAH HANANE

📧 Email: hananefellah35@gmail.com

🌐 GitHub: hananefellah

## 📄 License  
MIT License  
---


## How to Run
1. Clone the repository:  
```bash
git clone https://github.com/yourusername/Customer-Churn-Prediction.git
