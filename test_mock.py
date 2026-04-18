import yfinance as yf
from streamlit_app import fetch_data, WATCHLIST, get_ticker_info
from scripts.quant_engine import add_time_series_features, get_xgb_model
import datetime

import warnings
warnings.filterwarnings('ignore')

tic = 'AAPL'
start = datetime.date.today() - datetime.timedelta(days=365*20)
end = datetime.date.today()
print("Fetching...")
df = fetch_data(tic, start, end, '1d', False)
print(f"Data rows: {len(df)}")
df, feats = add_time_series_features(df, True, True, True, True)

ml_df = df.dropna()
X = ml_df[feats]
y = ml_df['Target']

print(f"Training on {len(X)} rows...")
# force auto_tune=False
model = get_xgb_model(X, y, False)
latest_features = ml_df.iloc[-1:][feats]
prob_up = model.predict_proba(latest_features)[0][1]

print(f"Prob up: {prob_up}")
print("Testing heuristics...")
latest_rsi = df.iloc[-1].get('RSI', 50)
print(latest_rsi)

info = get_ticker_info(tic)
print("Target Mean:", info.get('targetMeanPrice'))
