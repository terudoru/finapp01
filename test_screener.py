import yfinance as yf
import datetime
import pandas as pd
import quantstats as qs
import numpy as np

# from streamlit_app import ... is hard due to st setup.
# let's write minimal reproduction.
import streamlit as st
import sys
import os

from scripts.quant_engine import add_time_series_features, get_xgb_model
from streamlit_app import fetch_data, fetch_vix, fetch_live_price, WATCHLIST
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from xgboost import XGBClassifier

today = datetime.date.today()
start_date = today - datetime.timedelta(days=365*20)
end_date = today
timeframe = '1d'

results = []
all_tickers = []
for cat, items in WATCHLIST.items():
    if cat == "✍️ 自分で入力 (カスタム)": continue
    for name, tic in items.items():
        all_tickers.append({"name": name, "ticker": tic, "category": cat})

print(f"Testing {len(all_tickers)} tickers...")

for item in all_tickers[:5]: # testing 5
    tic = item["ticker"]
    is_cry = ("仮想通貨" in item["category"])
    print(f"checking {tic}...")
    try:
        df = fetch_data(tic, start_date, end_date, timeframe, is_cry)
        if df.empty or len(df) < 50:
            print(" -> Empty or < 50")
            continue
            
        df, features = add_time_series_features(df, True, True, True, True)
        
        ml_df = df.dropna().copy()
        if len(ml_df) < 30:
            print(" -> dropna < 30")
            continue
            
        X = ml_df[features]
        y = ml_df['Target']
        
        model = get_xgb_model(X, y, False)
        
        latest_features = df.iloc[-1:][features]
        if latest_features.isna().any().any():
            latest_features = ml_df.iloc[-1:][features]
            
        prob_up = model.predict_proba(latest_features)[0][1]
        print(f" -> prob_up: {prob_up}")
    except Exception as e:
        print(f" -> ERROR: {e}")

