# Employee Absenteeism Predictor

## Problem Statement

Employee absenteeism has a direct impact on productivity and efficiency. The goal of this project is to build a machine learning model that predicts whether an employee is likely to be absent **for more than 3 hours** on a given workday based on a variety of personal, professional, and health-related features.

The problem is framed as a **binary classification task**, where:

- `1` = Employee will be absent **more than 3 hours**
- `0` = Employee will be absent **3 hours or less**

---

## Workflow Overview

The project was developed across three stages, represented by three separate notebooks and a final module:

### 1. **`absenteeism_preprocessing.ipynb`**

- Initial data exploration and cleaning
- Feature engineering:
  - Extracted date-based features like **Month** and **Weekday**
  - Simplified `Education` levels
  - Grouped `Reason for Absence` into four broad categories (`reason_type_1` to `reason_type_4`)
- Created the target variable `Excessive Absenteeism` (binary: absent > 3 hours = 1)

### 2. **`absenteeism_mL.ipynb`**

- Model training and evaluation
- Standardized select numerical features using a custom `CustomScaler` class
- Trained a **logistic regression classifier**
- Achieved an accuracy of **78%** on the test set
- Saved the trained model and scaler using `pickle`

### 3. **`absenteeism_module_integration.ipynb`**

- Built a reusable **Python module** for deployment
  - `CustomScaler`: scales specified columns only
  - `absenteeism_model`: loads model and scaler, preprocesses new data, and generates predictions
- Loaded new test data and generated predictions:
  - Class probabilities
  - Binary predictions
  - Combined prediction results with preprocessed data

---

## Features Used

| Feature | Description |
|--------|-------------|
| Reason for Absence | Categorical (grouped into 4 types) |
| Transportation Expense | Numeric |
| Distance to Work | Numeric |
| Age | Numeric |
| Daily Work Load Average | Numeric |
| Body Mass Index (BMI) | Numeric |
| Education | Binary (0 = high school, 1 = higher education) |
| Children | Count |
| Pets | Count |
| Month Values | From Date |
| Week of Day | From Date |

---

## Project Structure

```plaintext
├── absenteeism_preprocessing.ipynb
├── absenteeism_mL.ipynb
├── absenteeism_module_integration.ipynb
├── absenteeism_predictor.py
├── model (Pickled trained model)
├── scaler (Pickled custom scaler)
├── Absenteeism_data.csv (Training data)
├── Absenteeism_new_data.csv (Test data)
└── README.md
````

---

## How to Use

1. Clone the repository and install dependencies (`numpy`, `pandas`, `sklearn`)
2. Run `absenteeism_module_integration.ipynb` to load new data and get predictions
3. The module can be easily plugged into a backend API or dashboard

---

## Current Performance

* **Test Accuracy**: `78%`
* Model: **Logistic Regression**
* Scaler: CustomScaler (only on selected numeric columns)

---

## Suggestions for Improvement

1. **Model Enhancements**

   * Try more complex models: Random Forest, Gradient Boosting (XGBoost), SVM
   * Use grid search for hyperparameter tuning
   * Perform cross-validation for more robust metrics

2. **Feature Engineering**

   * Analyze interaction terms (e.g., Age × BMI)
   * Include additional features (e.g., total sick days, overtime hours, department)

3. **Model Monitoring**

   * Track model performance on new batches of test data
   * Add confidence thresholds for decision-making

4. **Deployment**

   * Convert module into a **Flask or FastAPI app**
   * Integrate with HR systems for real-time prediction

5. **Explainability**

   * Add SHAP or LIME to explain individual predictions

---

```

Let me know if you'd like this in a downloadable file or if you want help turning this into a GitHub Pages project site.
```
