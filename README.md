# Predicting Term Deposit Subscription Using Bank Marketing Data

##  Project Overview

This machine learning project aims to predict whether a customer will subscribe to a term deposit based on a variety of personal, contact-related, and socio-economic factors. The project is developed using the **Bank Marketing dataset** from the **UCI Machine Learning Repository**, which contains real-world data collected through phone marketing campaigns by a Portuguese banking institution.

By building a classification model with robust preprocessing and evaluation steps, this project demonstrates how data science can assist marketing strategies to improve conversion rates and reduce campaign costs.

---

##  Problem Statement

Bank marketing campaigns can be resource-intensive and often result in low success rates. Predicting the likelihood of customer subscription can significantly improve marketing efficiency. Therefore, this project seeks to answer the following:

**"Can we predict whether a customer will subscribe to a term deposit based on their profile and interaction history?"**

This is framed as a binary classification task where:

* **Positive Class (1)**: Customer subscribed to a term deposit.
* **Negative Class (0)**: Customer did not subscribe.

---

##  Dataset Description

**Dataset Source**: [UCI Machine Learning Repository - Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)

| Detail                 | Description                                                       |
| ---------------------- | ----------------------------------------------------------------- |
| **Instances (Rows)**   | 45,211                                                            |
| **Features (Columns)** | 17 + 1 target variable (`y`)                                      |
| **Target Variable**    | `y`: Term deposit subscription (`yes` or `no`)                    |
| **Data Types**         | Mix of categorical (object) and numerical (int/float)             |
| **Missing Values**     | No true nulls, but 'unknown' used as a placeholder in many fields |

### Key Feature Groups:

* **Client Info**: `age`, `job`, `marital`, `education`, `default`, `housing`, `loan`
* **Last Contact**: `contact`, `month`, `day_of_week`, `duration`
* **Campaign Info**: `campaign`, `pdays`, `previous`, `poutcome`
* **Social/Economic**: `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed`

---

##  Project Workflow

### 1. Data Loading

* Import CSV file with semicolon (`;`) delimiter.
* Display top/bottom rows, shape, data types, and unique values.

### 2. Data Preprocessing

* Encode the target variable: `y` → {"yes": 1, "no": 0}
* Detect and address class imbalance using **SMOTE** (Synthetic Minority Oversampling Technique).
* Handle skewed distributions using `np.log1p` transformation.
* Cap outliers using IQR (Interquartile Range) method.
* Scale features using `MinMaxScaler`.
* Encode categorical features with **OneHotEncoding**.

### 3. Data Splitting

* Split dataset into **Train (70%)**, **Validation (15%)**, and **Unseen Test (15%)**.

### 4. Exploratory Data Analysis (EDA)

* Analyze target distribution.
* Study correlation matrix and summary statistics.
* Visualize:

  * Categorical features vs Target using countplots
  * Numerical features vs Target using boxplots

### 5. Feature Selection

* Apply statistical tests such as **Chi-Squared** and use **SelectKBest**.

### 6. Model Building

* Evaluate the following classifiers:

  * Logistic Regression
  * Decision Tree Classifier
  * Random Forest Classifier
  * Gradient Boosting Classifier
  * K-Nearest Neighbors (KNN)
  * Support Vector Machine (SVM)

### 7. Hyperparameter Tuning

* Use **GridSearchCV** to identify the best parameter combinations for each classifier.
* Perform 5-fold cross-validation during tuning.

### 8. Model Evaluation

* Evaluate model on **Validation Set** using:

  * Confusion Matrix
  * Accuracy, Precision, Recall, F1-Score
  * ROC-AUC Curve

### 9. Final Testing on Unseen Data

* Predict outcomes on unseen test set.
* Compare performance consistency with validation set results.

---

##  Feature Overview

| Feature Group    | Sample Features                               |
| ---------------- | --------------------------------------------- |
| Demographics     | `age`, `job`, `marital`, `education`          |
| Financial Status | `balance`, `loan`, `housing`                  |
| Contact Info     | `contact`, `month`, `day_of_week`, `duration` |
| Past Campaigns   | `campaign`, `pdays`, `previous`, `poutcome`   |
| Socio-Economic   | `emp.var.rate`, `cons.conf.idx`, `euribor3m`  |

---

##  Model Evaluation Summary

| Model               | Metric Used   | Best Score (Val Set) |
| ------------------- | ------------- | -------------------- |
| Logistic Regression | ROC-AUC / F1  | \~0.78 / \~0.60      |
| Decision Tree       | Accuracy / F1 | \~0.84 / \~0.64      |
| Random Forest       | ROC-AUC / F1  | \~0.88 / \~0.70      |
| Gradient Boosting   | ROC-AUC / F1  | \~0.90 / \~0.72      |
| K-Nearest Neighbors | Accuracy / F1 | \~0.80 / \~0.65      |
| SVM Classifier      | ROC-AUC / F1  | \~0.82 / \~0.66      |

**Note**: The scores vary based on hyperparameters selected via GridSearchCV.

---

##  How to Use This Project

1. **Clone or Download** the repository.
2. Install the required libraries using pip:

   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter notebook:

   ```bash
   jupyter notebook Final\ Project.ipynb
   ```
4. Load your unseen test data in the format of the training set.
5. Use the saved model (e.g., `.pkl` file) to make predictions.

---

##  Files Included

```
├── Final Project.ipynb         # Jupyter notebook with complete ML workflow
├── unseen_data.csv             # (Optional) Test dataset for final evaluation
├── saved_model.pkl             # Trained classification model
├── README.md                   # Project documentation
```

---

##  Project Conclusion

This project highlights how data preprocessing, exploratory analysis, feature engineering, and model optimization contribute to the effectiveness of predictive modeling. Through careful tuning and model comparison, we identified models capable of delivering **balanced and reliable predictions**. This predictive capability is invaluable for directing marketing efforts more efficiently, saving time and resources.

---

##  Future Improvements

* Integrate real-time model deployment via Flask or Streamlit.
* Apply advanced model stacking or ensemble blending.
* Add explainability tools (e.g., SHAP, LIME).
* Include more robust anomaly detection.
* Connect with real-time CRM system for automated prediction and targeting.

---

##  Dependencies

* Python 3.8+
* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
* imbalanced-learn
* xgboost

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

##  Author

**Jumailath**
Aspiring Data Scientist | Mathematics Graduate | Data Analytics Enthusiast
 Malappuram, India

---
