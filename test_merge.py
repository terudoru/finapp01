import yfinance as yf
import pandas as pd
import datetime
from streamlit_app import fetch_vix, fetch_data

start = datetime.date.today() - datetime.timedelta(days=100)
end = datetime.date.today()

try:
    df_us = fetch_data("AAPL", start, end, "1d", False)
    df_jp = fetch_data("7203.T", start, end, "1d", False)
    vix = fetch_vix(start, end)
    
    print("US index tz:", df_us.index.tz)
    print("JP index tz:", df_jp.index.tz)
    print("VIX index tz:", vix.index.tz)
    
    # Try merge_asof for US
    df_us = df_us.sort_index()
    vix = vix.sort_index()
    print("Merging US...")
    pd.merge_asof(df_us, vix[['VIX_Close']], left_index=True, right_index=True, direction='backward')
    print("US merged.")
    
    # Try merge_asof for JP
    df_jp = df_jp.sort_index()
    print("Merging JP...")
    pd.merge_asof(df_jp, vix[['VIX_Close']], left_index=True, right_index=True, direction='backward')
    print("JP merged.")
except Exception as e:
    print("ERROR:", e)

