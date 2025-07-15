# Predicting-Term-Deposit-Subscription-Using-Bank-Marketing-Data

##  Project Overview

This project focuses on predicting whether a customer will subscribe to a term deposit based on their personal and banking-related attributes. Using the **Bank Marketing dataset** from the UCI Machine Learning Repository, we develop a classification model to assist financial institutions in targeting the right customers for marketing campaigns.

The goal is to build a machine learning workflow that preprocesses the data, handles imbalances, and evaluates multiple classification algorithms, ultimately selecting the best-performing model using **GridSearchCV** for hyperparameter tuning.

---

##  Problem Statement

The objective is to **predict customer subscription (`yes`/`no`)** to a term deposit product using demographic and behavioral data. This is a binary classification problem, and the project explores various machine learning models such as Logistic Regression, Random Forest, Decision Trees, and others.

---

##  Dataset Description

**Source**: UCI Machine Learning Repository - Bank Marketing Dataset

| Detail                        | Description                                                              |
|------------------------------|--------------------------------------------------------------------------|
| **Rows**                     | 45,211                                                                   |
| **Features**                 | 17 features + 1 target variable                                          |
| **Target Variable**          | `y` — whether the client subscribed to a term deposit (`yes` or `no`)   |
| **Feature Types**            | Mix of categorical and numerical                                         |
| **Missing Values**           | No missing values                                                        |

---

##  Project Workflow

1. **Data Loading**
2. **Exploratory Data Analysis (EDA)**
3. **Preprocessing**:
   - Encoding categorical features
   - Handling class imbalance (SMOTE)
   - Skewness correction (`log1p`)
   - Outlier capping
   - Feature scaling
4. **Data Splitting** (Train/Validation/Unseen)
5. **Model Building**:
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - Gradient Boosting
6. **Hyperparameter Tuning** using **GridSearchCV**
7. **Model Evaluation**:
   - Accuracy, Precision, Recall, F1-score
   - Confusion Matrix
   - ROC-AUC
8. **Prediction on Unseen Data**

---

##  Feature Overview

The following features were considered during the analysis:

- **Demographics**: age, job, marital status, education
- **Contact Info**: contact type, month, day of week
- **Financials**: balance, duration, previous campaign contact
- **Past Campaigns**: number of contacts performed, outcome of previous campaigns

---

##  Model Evaluation

Models were evaluated using the following metrics:

- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **ROC-AUC**

The **best model** was selected based on **F1 Score and ROC-AUC**, with optimal hyperparameters determined via GridSearchCV.

---

##  How to Use

1. Clone the repository or download the `.ipynb` file.
2. Install the required libraries listed below.
3. Run the notebook step-by-step.
4. Use the final saved model to make predictions on unseen data.

---

##  Conclusion

This project demonstrated the process of building a robust classification model using real-world banking data. After comprehensive preprocessing and evaluation, the model was able to predict term deposit subscriptions with a good balance between **precision and recall**, making it valuable for targeting potential customers in marketing campaigns.

---

##  Future Enhancements

- Deploy the model using Flask or Streamlit
- Add more domain-specific features
- Integrate real-time data pipelines for production
- Implement feature selection and model stacking

---

##  Requirements

```bash
Python 3.8+
scikit-learn
pandas
numpy
matplotlib
seaborn
imblearn
xgboost
