from os import name
import streamlit as st
import numpy as np
import pandas as pd
from pandas_datareader import data
import datetime
import  joblib
from sklearn.metrics import explained_variance_score
import matplotlib.pyplot as plt
@st.cache
def get_X_y(df):
     xy_df = pd.DataFrame(df)
     xy_df['Year'] = df['Date'].dt.year
     xy_df['Month'] = df['Date'].dt.month
     xy_df['Day'] = df['Date'].dt.day
     X = xy_df[['Day', 'Month', 'Year', 'High', 'Low', 'Open']]
     y = xy_df[['Close']]
     return X, y

def get_df(stock, start_date, end_date):
     return data.DataReader(name=stock, data_source='yahoo', start=start_date, end=end_date)


st.title('Stock Market Prediction')
col1, col2, col3 = st.beta_columns(3)
start_date = str(col1.date_input(label='Start Date', value=datetime.date(2019, 1, 1)))
end_date = str(col2.date_input(label='End Date', value=datetime.date(2020, 1, 1)))
stock = col3.text_input(label='Enter Stock Ticker', value='NFLX')
df = get_df(stock, start_date, end_date)
df.reset_index(inplace=True)
# st.write(df.head())
# corr = df.corr(method='pearson')
# import seaborn as sns
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
# heatmap = sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, cmap='RdBu_r', ax=ax, annot=True, linewidths=0.5)
# plt.rcParams['figure.figsize'] = (2,2)
# st.write(fig)
@st.cache
def get_model():
     return joblib.load('trained_linear_regression_model')

model = get_model()
X, y= get_X_y(df)
y_predict = model.predict(X)
res_df = pd.DataFrame({'actual': y.iloc[:, 0], 'predicted': y_predict})
res_df['Date'] = df['Date']
res_df.set_index('Date', inplace=True)
# fig = plt.figure(figsize=(16,8))
# plt.style.use('fivethirtyeight')
# plt.plot(df['Date'], y, label='Actual')
# plt.plot(df['Date'], y_predict, label='Predicted')
# plt.tight_layout()
score = explained_variance_score(y, y_predict) * 100

st.line_chart(res_df)
st.write('Explained Variance Score: ', score)
st.write(res_df)
