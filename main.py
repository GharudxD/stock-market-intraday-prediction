#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: vinitshah

"""

# importing
import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout , TimeDistributed , Conv1D , MaxPooling1D , Flatten , Bidirectional
from keras.callbacks import ModelCheckpoint
import pickle5 as pickle
import matplotlib.pyplot as plt
import json


# nifty = pd.read_csv('ind_niftynext50list.csv').Symbol.to_list()
# with open('stock_list.txt','r') as f:
#     x = f.readlines()
# nifty = [s.strip() for s in x]
with open('stocks.json','r') as f:
    data = json.load(f)
nifty = data.keys()
st.write(
    """
    # Intraday Stock Price Prediction
    """
)

st.write(
"""
#
"""
    )
option = st.selectbox("Choose Your Stock for Prediction",nifty)
# col1,col2 = st.columns([3,1])
# with col1:
#     option = st.write("Which Stock's price you wanna predict",nifty)

# with col2:
#     frame = st.write("Chose DataFram",['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'])

# making columns for start date and end date
st_ = datetime.date.today() - relativedelta(years=5)
# st_ = datetime.date.today() - relativedelta(months=3)
print(st_)
# start,end = st.columns(2)

# with start:
#     # st.header("Start Date")
#     start_date = st.date_input(
#     "Start Date",
#     datetime.date(2019, 7, 6))

# with end:
#     # st.header("End Date")
#     end_date = st.date_input(
#     "End Date",
#     datetime.date.today())
# print
stock = yf.Ticker(data[option]+".NS")
df = stock.history(period='1d',
    start = st_,end = datetime.date.today())

def charts(option):
    price,volumes = st.columns(2)
    # st.write(f"""
    #     ## 5 Year Price history of {option}
    #     """)
    with price:
        st.header("Price")
        st.line_chart(df.Close)

    # st.write(f"""
    #     ## 5 Year Volume history of {option}
    #     """)
    with volumes:
        st.header("Volume")
        st.line_chart(df.Volume)

chat_ = st.checkbox(f"5 year Chart of {option}")

if chat_:
    charts(option)

def makeFeature(data,feature=24):
    x = list()
    y = list()
    for i in range(len(data)-feature-1):
        x.append(data[i:(i + feature),0])
        y.append(data[i + feature,0])
    x = np.array(x)
    y = np.array(y)
    return x,y


def prediction(df):
    model = pickle.load(open('model_val_best.pkl', 'rb'))
    min_max = pickle.load(open('minmax.pkl','rb'))
    df_minmax = df.iloc[:,3:4]
    # min_max = MinMaxScaler(feature_range=(0,1))
    df_minmax = min_max.transform(df_minmax)
    x_input = df_minmax[-24:].reshape(1,-1)
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    x_input = np.reshape(x_input,(x_input.shape[0],1,24,1))
    yhat = model.predict(x_input, verbose=0)
    return min_max.inverse_transform(yhat) > df['Close'][-1]

# st.write(f"""
#     # Next day Predicition for {option}
#     """)
one_,real_,two_ = st.columns([1,1,1])
with real_:
    result = st.button(f"Intraday Predicition")

if result:
    # st.write(f"""
    # # Next day Predicition for {option}
    # """)
    one,real,two = st.columns([1,1,1])
    with real:
        pred = prediction(df)
        if pred[0][0]:
            # st.write('Bullish',color='g')
            new_title = '<p style="font-family:sans-serif; color:Green; font-size: 42px;"><centre>Bullish🐂</centre></p>'
            st.markdown(new_title, unsafe_allow_html=True)
        else:
            # st.write('Barrish',color='r')
            new_title = '<p style="font-family:sans-serif; color:Red; font-size: 42px;"><centre>Barrish🐻</centre></p>'
            st.markdown(new_title, unsafe_allow_html=True)
            
