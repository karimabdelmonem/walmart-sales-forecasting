import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

DATA_PATH   = r'C:\Users\User\Desktop\project\Data sets\data_processed/'
MODELS_PATH = r'C:\Users\User\Desktop\project\models/'

st.set_page_config(page_title='Walmart Sales Forecasting', layout='wide')
st.title('Walmart Store Sales Forecasting')
st.divider()

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODELS_PATH + 'final_model.pkl')
    scaler = joblib.load(MODELS_PATH + 'scaler.pkl')
    medians = joblib.load(MODELS_PATH + 'medians.pkl')
    last_sales = joblib.load(MODELS_PATH + 'last_sales.pkl')
    feature_cols = joblib.load(MODELS_PATH + 'feature_cols.pkl')
    return model, scaler, medians, last_sales, feature_cols

@st.cache_data
def load_train():
    df = pd.read_parquet(DATA_PATH + 'train_full.parquet')
    df['Date']  = pd.to_datetime(df['Date'])
    df['Year']  = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    return df

model, scaler, medians, last_sales, FEATURE_COLS = load_artifacts()
train_df = load_train()

def engineer_features(df, medians, last_sales):
    df = df.copy()
    df['Date']  = pd.to_datetime(df['Date'])
    df['Year']  = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week']  = df['Date'].dt.isocalendar().week.astype(int)
    df['Day']   = df['Date'].dt.day

    thanksgiving_date = pd.to_datetime(df['Year'].astype(str) + '-11-24')
    christmas_date    = pd.to_datetime(df['Year'].astype(str) + '-12-24')
    df['Days_to_Thanksgiving'] = (thanksgiving_date - df['Date']).dt.days.astype(int)
    df['Days_to_Christmas']    = (christmas_date    - df['Date']).dt.days.astype(int)

    df['SuperBowl']    = (df['Week'] == 6).astype(int)
    df['LaborDay']     = (df['Week'] == 36).astype(int)
    df['Thanksgiving'] = (df['Week'] == 47).astype(int)
    df['Christmas']    = (df['Week'] == 52).astype(int)
    df['BlackFriday']  = ((df['Month'] == 11) & (df['Week'].isin([47, 48]))).astype(int)

    md_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
    for col in md_cols:
        if col not in df.columns:
            df[col] = 0
        df[f'{col}_present'] = (df[col] > 0).astype(int)
    df['MarkdownsSum']   = df[md_cols].sum(axis=1)
    df['MarkdownsCount'] = (df[md_cols] > 0).sum(axis=1)

    df['IsHoliday']   = df['IsHoliday'].astype(int)
    df['TypeEncoded'] = df['Type'].map({'A': 1, 'B': 2, 'C': 3}).fillna(1)

    median_key    = ['Type', 'Store', 'Dept', 'Month', 'IsHoliday']
    available_key = [k for k in median_key if k in df.columns and k in medians.columns]
    df = df.merge(medians, on=available_key, how='left')
    df['MedianSales'] = df['MedianSales'].fillna(medians['MedianSales'].median())

    df = df.merge(last_sales[['Store', 'Dept', 'LaggedSales']], on=['Store', 'Dept'], how='left')
    df['LaggedSales'] = df['LaggedSales'].fillna(df['MedianSales'])

    df = df.fillna(0)
    return df


st.subheader('Monthly Average Weekly Sales Evolution')

monthly = train_df.groupby(['Year', 'Month'])['Weekly_Sales'].mean().reset_index()
fig = px.line(monthly, x='Month', y='Weekly_Sales', animation_frame='Year',
              title='Monthly Average Weekly Sales Evolution',
              labels={'Weekly_Sales': 'Mean Weekly Sales ($)'})
fig.update_layout(xaxis=dict(tickmode='linear', dtick=1))
st.plotly_chart(fig, use_container_width=True)

st.divider()

# SECTION 2 — Prediction
st.subheader('Predict Weekly Sales')

col1, col2, col3 = st.columns(3)

with col1:
    store       = st.number_input('Store',         min_value=1,    max_value=45,     value=1)
    dept        = st.number_input('Department',    min_value=1,    max_value=99,     value=1)
    store_type  = st.selectbox('Store Type',       ['A', 'B', 'C'])
    size        = st.number_input('Store Size',    min_value=1000, max_value=250000, value=150000, step=5000)
    is_holiday  = st.checkbox('Holiday Week?')

with col2:
    date        = st.date_input('Date')
    temperature = st.number_input('Temperature (°F)', value=60.0)
    fuel_price  = st.number_input('Fuel Price ($)',   value=3.5)
    cpi         = st.number_input('CPI',              value=180.0)
    unemployment= st.number_input('Unemployment (%)', value=8.0)

with col3:
    md1 = st.number_input('MarkDown1', value=0.0)
    md2 = st.number_input('MarkDown2', value=0.0)
    md3 = st.number_input('MarkDown3', value=0.0)
    md4 = st.number_input('MarkDown4', value=0.0)
    md5 = st.number_input('MarkDown5', value=0.0)

if st.button('Predict', use_container_width=True):
    row = pd.DataFrame([{
        'Store': store, 'Dept': dept, 'Date': str(date),
        'Type': store_type, 'Size': size, 'IsHoliday': int(is_holiday),
        'Temperature': temperature, 'Fuel_Price': fuel_price,
        'CPI': cpi, 'Unemployment': unemployment,
        'MarkDown1': md1, 'MarkDown2': md2, 'MarkDown3': md3,
        'MarkDown4': md4, 'MarkDown5': md5,
    }])

    row_fe = engineer_features(row, medians, last_sales)
    for c in FEATURE_COLS:
        if c not in row_fe.columns:
            row_fe[c] = 0

    X_scaled = scaler.transform(row_fe[FEATURE_COLS])
    pred     = model.predict(X_scaled)[0]

    st.success(f'Predicted Weekly Sales:  **${pred:,.2f}**')
