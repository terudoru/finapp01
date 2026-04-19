import yfinance as yf
import pandas as pd
import datetime

start = datetime.date.today() - datetime.timedelta(days=100)
end = datetime.date.today()

# Stock
df = yf.download("7203.T", start=start, end=end + datetime.timedelta(days=1), interval='1d', progress=False)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# VIX
vix_df = yf.download("^VIX", start=start, end=end + datetime.timedelta(days=1), interval='1d', progress=False)
if isinstance(vix_df.columns, pd.MultiIndex):
    vix_df.columns = vix_df.columns.get_level_values(0)
vix_df = vix_df[['Close']].rename(columns={'Close': 'VIX_Close'})

# Apply logic
if df.index.tz is not None:
    df.index = df.index.tz_localize(None)
if vix_df.index.tz is not None:
    vix_df.index = vix_df.index.tz_localize(None)

df = df.sort_index()
vix_df = vix_df.sort_index()
# Drop duplicated dates
df = df[~df.index.duplicated(keep='last')]
vix_df = vix_df[~vix_df.index.duplicated(keep='last')]

merged = pd.merge_asof(df, vix_df[['VIX_Close']], left_index=True, right_index=True, direction='backward')
merged['VIX_Close'] = merged['VIX_Close'].ffill()

print("Original DF length:", len(df))
print("VIX DF length:", len(vix_df))
print("Merged VIX NaNs:", merged['VIX_Close'].isna().sum())
print("Merged head:")
print(merged[['Close', 'VIX_Close']].head())
