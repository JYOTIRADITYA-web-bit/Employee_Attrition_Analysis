# employee_attrition_dashboard_portfolio.py

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import joblib

# ------------------------------
# 1. Load Dataset & Model
# ------------------------------
df = pd.read_csv('HR_Attrition.csv')

# Map attrition to numeric
df['Attrition_num'] = df['Attrition'].map({'No': 0, 'Yes': 1})  # 0 = No, 1 = Yes

# ------------------------------
# Attrition Breakdown (Counts)
# ------------------------------
yes_count = df['Attrition_num'].sum()   # 1 = Yes
no_count = (df['Attrition_num'] == 0).sum()

st.sidebar.markdown("### 📌 Attrition Overview")
st.sidebar.write(f"Employees Left (Yes=1): **{yes_count}**")
st.sidebar.write(f"Employees Stayed (No=0): **{no_count}**")

# Load trained LightGBM model
lgb_model = joblib.load('attrition_model_lgbm.pkl')
X_train_cols = lgb_model.feature_name_

# ------------------------------
# 2. Page Layout
# ------------------------------
st.set_page_config(page_title="Employee Attrition Dashboard", layout="wide")
st.title("📊 Employee Attrition Analysis")

# ------------------------------
# 3. Sidebar Filters (default = All)
# ------------------------------
st.sidebar.header("Filters")

# Department
departments = ["All"] + df['Department'].unique().tolist()
department_filter = st.sidebar.selectbox("Department", options=departments, index=0)

# Job Role
jobroles = ["All"] + df['JobRole'].unique().tolist()
jobrole_filter = st.sidebar.selectbox("Job Role", options=jobroles, index=0)

# OverTime
overtime_filter = st.sidebar.selectbox("OverTime", options=["All", "Yes", "No"], index=0)

# ------------------------------
# 4. Apply Filters
# ------------------------------
filtered_df = df.copy()

if department_filter != "All":
    filtered_df = filtered_df[filtered_df['Department'] == department_filter]
if jobrole_filter != "All":
    filtered_df = filtered_df[filtered_df['JobRole'] == jobrole_filter]
if overtime_filter != "All":
    filtered_df = filtered_df[filtered_df['OverTime'] == overtime_filter]

# ------------------------------
# 5. KPIs
# ------------------------------
st.subheader("Key Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Attrition Rate", f"{filtered_df['Attrition_num'].mean()*100:.1f}%")
col2.metric("Avg Years at Company", f"{filtered_df['YearsAtCompany'].mean():.1f}")
col3.metric("Avg Monthly Income", f"${filtered_df['MonthlyIncome'].mean():,.0f}")
col4.metric("Overtime Employees", f"{(filtered_df['OverTime']=='Yes').mean()*100:.1f}%")

# ------------------------------
# Overall Attrition by Department (Stacked Bar)
# ------------------------------
# ------------------------------
# Overall Attrition by Department (Filtered Stacked Bar)
# ------------------------------
st.subheader("Attrition vs Stayed by Department (Filtered)")

# Group by Department and Attrition_num after applying filters
dept_attrition_stacked = (
    filtered_df.groupby(['Department', 'Attrition_num'])
    .size()
    .reset_index(name='Count')
)
dept_attrition_stacked['Attrition'] = dept_attrition_stacked['Attrition_num'].map({0:'Stayed', 1:'Left'})

fig_bar = px.bar(
    dept_attrition_stacked,
    x='Department',
    y='Count',
    color='Attrition',
    title='Attrition vs Stayed by Department (Filtered)',
    text='Count',
    color_discrete_map={'Stayed':'green','Left':'red'}
)
st.plotly_chart(fig_bar, use_container_width=True)



# ------------------------------
# 6. Visualizations
# ------------------------------
# Attrition by Dept & Job Role with toggle
st.subheader("Attrition by Department & Job Role")

view_option = st.radio("View by:", ["Attrition Count", "Attrition %"], index=1, horizontal=True)

if view_option == "Attrition Count":
    dept_job_attrition = (
        filtered_df.groupby(['Department', 'JobRole'])['Attrition_num']
        .sum()
        .reset_index()
        .rename(columns={'Attrition_num': 'Attrition_Count'})
    )
    fig1 = px.bar(dept_job_attrition, x='Department', y='Attrition_Count', color='JobRole', barmode='stack')
else:
    dept_job_attrition = (
        filtered_df.groupby(['Department', 'JobRole'])['Attrition_num']
        .mean()
        .reset_index()
        .rename(columns={'Attrition_num': 'Attrition_Rate'})
    )
    fig1 = px.bar(dept_job_attrition, x='Department', y='Attrition_Rate', color='JobRole', barmode='stack')
    fig1.update_yaxes(tickformat=".0%")

st.plotly_chart(fig1, use_container_width=True)

# Feature Distributions
st.subheader("Feature Distributions")
numeric_cols = [
    'Age', 'DistanceFromHome', 'MonthlyIncome', 'YearsAtCompany', 
    'YearsInCurrentRole', 'TotalWorkingYears', 'NumCompaniesWorked', 
    'PercentSalaryHike', 'TrainingTimesLastYear'
]
feature_to_plot = st.selectbox("Select Feature", options=numeric_cols)
fig2 = px.histogram(
    filtered_df, x=feature_to_plot, color='Attrition_num',
    marginal='box', color_discrete_map={0:'green',1:'red'},
    title=f"{feature_to_plot} Distribution by Attrition"
)
st.plotly_chart(fig2, use_container_width=True)

# ------------------------------
# 7. Feature Importance
# ------------------------------
st.subheader("Top Features Influencing Attrition")
importances = lgb_model.feature_importances_
feat_importance = pd.DataFrame({'feature': X_train_cols, 'importance': importances}).sort_values(by='importance', ascending=False)
fig3 = px.bar(feat_importance.head(10), x='feature', y='importance', title="Top 10 Features", text='importance')
st.plotly_chart(fig3, use_container_width=True)

# ------------------------------
# 8. Predict Probabilities for At-Risk Employees
# ------------------------------
X_filtered = filtered_df.copy()
for col in X_train_cols:
    if col not in X_filtered.columns:
        X_filtered[col] = 0
X_filtered = X_filtered[X_train_cols]

filtered_df['Attrition_Prob'] = lgb_model.predict_proba(X_filtered)[:,1]

st.subheader("Top 5 At-Risk Employees")
top5_risk = filtered_df.sort_values('Attrition_Prob', ascending=False).head(5)
st.dataframe(top5_risk[['EmployeeNumber','JobRole','Department','Attrition_Prob','YearsAtCompany','YearsInCurrentRole']])

# ------------------------------
# 9. Insights & Recommendations
# ------------------------------
st.subheader("Insights & Recommendations")

attrition_rate = filtered_df['Attrition_num'].mean()
avg_years_role = filtered_df['YearsInCurrentRole'].mean()
overtime_rate = (filtered_df['OverTime']=='Yes').mean()
low_satisfaction = filtered_df['JobSatisfaction'].mean()

insights = []
if attrition_rate > 0.15:
    insights.append(f"High attrition detected: {attrition_rate*100:.1f}%.")
if overtime_rate > 0.3:
    insights.append(f"{overtime_rate*100:.1f}% of employees are working overtime.")
if low_satisfaction < 3:
    insights.append(f"Job satisfaction is low (avg={low_satisfaction:.1f}).")
if avg_years_role < 3:
    insights.append(f"Employees stay short in role (avg={avg_years_role:.1f} years).")

insights.append("Recommendations:")
insights.append("- Focus retention on high-risk employees.")
insights.append("- Improve work-life balance.")
insights.append("- Conduct employee surveys to identify pain points.")

for item in insights:
    st.markdown(f"- {item}")

# ------------------------------
# 10. Download Filtered Data
# ------------------------------
csv = filtered_df.to_csv(index=False)
st.download_button("Download Filtered Data", data=csv, file_name='filtered_employee_data.csv', mime='text/csv')

# ------------------------------
# 11. Model Evaluation
# ------------------------------
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.subheader("Model Evaluation")

# Prepare X and y
X = df.copy()
for col in X_train_cols:
    if col not in X.columns:
        X[col] = 0
X = X[X_train_cols]
y = df['Attrition_num']

# Predict
y_pred = lgb_model.predict(X)

# Metrics
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{accuracy:.2f}")
col2.metric("Precision", f"{precision:.2f}")
col3.metric("Recall", f"{recall:.2f}")
col4.metric("F1 Score", f"{f1:.2f}")

# Confusion Matrix
cm = confusion_matrix(y, y_pred)
cm_df = pd.DataFrame(cm, index=['Actual No', 'Actual Yes'], columns=['Predicted No', 'Predicted Yes'])

st.write("Confusion Matrix:")
st.dataframe(cm_df)