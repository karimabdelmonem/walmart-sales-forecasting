# Walmart Store Sales Forecasting

An end-to-end Machine Learning project for forecasting Walmart weekly sales across different stores and departments.

## Project Highlights
✔ Data Cleaning & Preprocessing  
✔ Feature Engineering  
✔ Model Training & Comparison  
✔ Interactive Visualization  
✔ Streamlit Deployment  


## Problem Statement
Accurate sales forecasting helps retail businesses improve:

- Inventory management  
- Marketing strategies  
- Holiday planning  
- Revenue optimization  

This project predicts **Weekly Sales** using historical sales records, holidays, markdown events, and economic indicators.


## Dataset Features

The dataset contains:

| Feature | Description |
|---------|-------------|
| Store   | Store ID |
| Dept    | Department ID |
| Date    | Week date |
| Weekly_Sales | Target variable |
| IsHoliday | Holiday week or not |
| Temperature | Regional temperature |
| Fuel_Price | Fuel price |
| CPI     | Consumer Price Index |
| Unemployment | Unemployment rate |
| MarkDown1-5 | Promotional markdown data |


## Feature Engineering

Engineered features include:

- Year / Month / Week / Day  
- Days to Thanksgiving  
- Days to Christmas  
- Super Bowl / Labor Day / Thanksgiving / Christmas indicators  
- Black Friday indicator  
- MarkDowns Sum / Count  
- Median Sales per group  
- Lagged Sales  


## Models Trained

The following models were trained and compared:

- Linear Regression  
- Ridge Regression  
- Random Forest Regressor  
- XGBoost Regressor  

The best-performing model was selected and deployement


## 🖥 Deployment

The final model was deployed using Streamlit.

Run locally:
```bash
streamlit run 03_deployment.py