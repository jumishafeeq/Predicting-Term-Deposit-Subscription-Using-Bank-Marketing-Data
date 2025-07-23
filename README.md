# Predicting Term Deposit Subscription Using Bank Marketing Data

##  Project Overview

This machine learning project aims to predict whether a customer will subscribe to a term deposit based on a variety of personal, contact-related, and socio-economic factors. The project is developed using the **Bank Marketing dataset** from the **UCI Machine Learning Repository**, which contains real-world data collected through phone marketing campaigns by a Portuguese banking institution.

By building a classification model with robust preprocessing and evaluation steps, this project demonstrates how data science can assist marketing strategies to improve conversion rates and reduce campaign costs.

---

## Objectives

- Understand customer behavior from marketing data.
- Build and compare classification models to predict subscription.
- Address class imbalance using SMOTE.
- Tune the best model for optimal performance.
- Deploy the final model and make predictions on unseen data.

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

* Checked and confirmed no missing values
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

#### XGBoost (Final Model)
- Tuned using `GridSearchCV` on parameters:
  - `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`
- Achieved the best overall performance in **accuracy** and **precision**
- Selected as the **final deployed model**

### 8. Model Evaluation

* Evaluate model on **Validation Set** using:

  * Confusion Matrix
  * Accuracy, Precision, Recall, F1-Score
  * ROC-AUC Curve

---

### 9. Pipeline Creation and Deployment

We built an end-to-end pipeline using `ImbPipeline`:

- Steps:
  1. Preprocessing (`ColumnTransformer`)
  2. Feature selection (`SelectKBest`)
  3. Resampling (`SMOTE`)
  4. Final model (`XGBoostClassifier`)
- Saved using `joblib.dump()`
- Can be reused on unseen datasets without re-running preprocessing

---

### 10. Prediction on Unseen Data

- Preprocessed the new dataset using the **same steps**
- Loaded saved model: `model_pipeline.pkl`
- Predicted outcomes
- Saved results to `unseen_predictions.csv`

---


##  Model Evaluation Summary

| Model                   | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|------------------------|----------|-----------|--------|----------|---------|
| Logistic Regression     | 0.7876   | 0.2997    | 0.6624 | 0.4127   | 0.7793  |
| XGBoost (Tuned)         | **0.8862** | **0.4930** | 0.3534 | **0.4117** | **0.7729** |
| Linear SVM              | 0.7804   | 0.2938    | **0.6753** | 0.4094   | 0.7800  |
| Random Forest           | 0.8701   | 0.4060    | 0.3290 | 0.3635   | 0.7554  |

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
