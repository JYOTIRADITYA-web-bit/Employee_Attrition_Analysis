# Employee Attrition Analysis & Dashboard

This project includes a **comprehensive analysis of employee attrition** using Python, LightGBM, and Streamlit. It contains:

1. A **Jupyter notebook** for data preprocessing, EDA, feature engineering, LightGBM model training, and evaluation.
2. A **Streamlit dashboard** for interactive visualization, KPIs, at-risk employee predictions, and feature importance insights.

---

## Notebook Features

- **Data Preprocessing:** Encode categorical variables, handle irrelevant columns, balance classes using SMOTE.  
- **Exploratory Data Analysis (EDA):** Visualize attrition distribution, numeric and categorical feature distributions.  
- **Feature Engineering:** Create interaction features such as `YearsInRole_ratio`.  
- **Model Training:** Train a **LightGBM classifier** with hyperparameters and class balancing.  
- **Model Evaluation:** Accuracy, precision, recall, F1-score, confusion matrix, and feature importance visualization.  
- **Model Saving:** Save trained model for later use in Streamlit dashboard.

---

## Streamlit Dashboard Features

- **Interactive Filters:** Department, Job Role, OverTime, and numeric feature sliders (optional).  
- **Key Metrics (KPIs):**  
  - Attrition rate  
  - Average years at company  
  - Monthly income  
  - Overtime percentage  
- **Visualizations:**  
  - Attrition vs stayed by department  
  - Attrition by department & job role  
  - Numeric feature distributions with histogram + boxplot  
- **Feature Importance:** Top features influencing attrition (from LightGBM).  
- **At-Risk Employees:** Predicted attrition probabilities, top 5 at-risk employees.  
- **Insights & Recommendations:** Auto-generated based on attrition metrics, satisfaction, and overtime.  
- **Download:** Filtered employee data as CSV.  
- **Model Evaluation Metrics:** Accuracy, precision, recall, F1 score, and confusion matrix displayed.

---

## Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/JYOTIRADITYA-web-bit/Employee_Attrition_Analysis.git
cd Employee_Attrition_Analysis


## 🚀 Live Dashboard

Interact with the **Employee Attrition Dashboard** online:

[Open Dashboard](https://employeeattritionanalysis-bjfnuksb6ybybnudw4b4z5.streamlit.app/)