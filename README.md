# Customer Churn and Tenure Prediction

## Project Overview
This project analyzes customer behavior in the telecommunications domain with the goal of improving customer retention. The work focuses on two related problems:

1. Predicting whether a customer is likely to churn (classification)
2. Predicting the expected tenure of a customer in months (regression)

By using machine learning models, businesses can identify customers at higher risk of churn and better understand factors that influence long-term customer retention.

---

## Project Structure
The project is organized into three notebooks, each covering a specific stage of the workflow.

1. 01_EDA.ipynb → Data understanding and exploratory analysis
2. 02_Classification.ipynb → Churn prediction (classification)
3. 03_Regression.ipynb → Tenure prediction (regression)

---

## 1. Exploratory Data Analysis (EDA)
**Notebook:** `01_EDA.ipynb`

### Objective
To understand the dataset, clean the data, and identify patterns related to customer churn and tenure.

### Key Steps
- **Data Cleaning:**  
  Handled missing values in features such as `TotalCharges` and ensured correct data types.
- **Univariate Analysis:**  
  Analyzed churn distribution and observed class imbalance between churned and non-churned customers.
- **Bivariate Analysis:**  
  Studied relationships between churn and features like contract type, tenure, and monthly charges.
- **Correlation Analysis:**  
  Identified strong correlation between `tenure` and `TotalCharges`.
- **Feature Engineering:**  
  Created simple derived features such as `IsLongTermContract` to capture contract behavior.

### Key Observations
- Customers on month-to-month contracts tend to churn more frequently.
- Churn risk is higher during the early months of a customer’s tenure.
- Longer-tenure customers are more likely to stay with the service.

---

## 2. Churn Prediction (Classification)
**Notebook:** `02_Classification.ipynb`

### Objective
To build a binary classification model that predicts whether a customer will churn (`Yes` or `No`).

### Approach
- **Preprocessing:**  
  - One-Hot Encoding for categorical variables  
  - Standard Scaling for numerical features
- **Class Imbalance Handling:**  
  Used SMOTE to reduce bias toward the majority (non-churn) class.
- **Model Comparison:**  
  Trained and evaluated multiple models:
  - Logistic Regression (baseline)
  - Random Forest
  - Support Vector Machine (SVM)
  - XGBoost
- **Evaluation Metrics:**  
  Focused on Recall, F1-score, and ROC-AUC due to class imbalance.
- **Hyperparameter Tuning:**  
  Performed tuning using `RandomizedSearchCV`.

### Final Model Selection
**Selected Model:** XGBoost Classifier  

XGBoost was chosen because it provided better performance on the minority (churn) class and showed more stable results during cross-validation compared to simpler models.

### Output
- Trained churn prediction model saved as `best_model.pkl`

---

## 3. Tenure Prediction (Regression)
**Notebook:** `03_Regression.ipynb`

### Objective
To predict customer tenure (in months), which can be useful for estimating customer lifetime value.

### Approach
- **Feature Engineering:**  
  - Created `ServiceDensity` to represent overall service usage  
  - Removed highly correlated features such as `TotalCharges` to reduce multicollinearity
- **Model Comparison:**  
  Evaluated:
  - Linear Regression
  - ElasticNet
  - Random Forest Regressor
  - XGBoost Regressor
- **Evaluation Metrics:**  
  Used MAE, RMSE, and R² to measure prediction quality.
- **Hyperparameter Tuning:**  
  Applied `RandomizedSearchCV` for model optimization.

### Final Model Selection
**Selected Model:** XGBoost Regressor  

This model achieved lower prediction error (MAE) and better overall fit compared to baseline regression models.

### Output
- Trained tenure prediction model saved as `reg_best_model.pkl`

---

## Key Insights
- Customers with month-to-month contracts have a higher likelihood of churn.
- The first few months of customer tenure are the most critical for retention.
- Features such as `Contract`, `MonthlyCharges`, `Tenure`, and `ServiceDensity` play an important role in predicting customer behavior.

---

## Business Impact
- Helps identify customers at higher risk of churn for early intervention.
- Supports retention strategies by highlighting key churn drivers.
- Enables estimation of customer lifetime through tenure prediction.

---

## Technologies Used
- **Programming Language:** Python  
- **Data Analysis:** Pandas, NumPy  
- **Visualization:** Matplotlib, Seaborn  
- **Machine Learning:** Scikit-learn, XGBoost  
- **Imbalance Handling:** Imbalanced-learn (SMOTE)  
- **Model Saving:** Joblib  

---

## Reproducibility
1. Place the dataset `CustomerChurn.csv` in the `data/` directory.
2. Run the notebooks in the following order:
   1. `01_EDA.ipynb`
   2. `02_Classification.ipynb`
   3. `03_Regression.ipynb`
3. Load the saved `.pkl` files to generate predictions on new data.

---

## Future Improvements
- Add model explainability using feature importance or SHAP values
- Optimize classification threshold based on business cost
- Deploy models using a simple API (FastAPI or Flask)
- Monitor model performance on new data
