import pandas as pd
import datetime
from streamlit_app import fetch_data, add_time_series_features, fetch_vix, train_and_cache_model

start_date = datetime.date.today() - datetime.timedelta(days=100)
end_date = datetime.date.today()
timeframe = '1d'
tic = "7203.T"
is_cry = False
use_sma = True; use_rsi = True; use_macd = True; use_bb = True

try:
    df = fetch_data(tic, start_date, end_date, timeframe, is_cry)
    if df.empty or len(df) < 5:
        print("DF empty or < 5")
        exit()
        
    df, features = add_time_series_features(df, use_sma, use_rsi, use_macd, use_bb)
    print("Features extracted:", features)
    if True:
        vix_df = fetch_vix(start_date, end_date)
        if not vix_df.empty:
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            if vix_df.index.tz is not None:
                vix_df.index = vix_df.index.tz_localize(None)
            df = df.sort_index()
            vix_df = vix_df.sort_index()
            df = pd.merge_asof(df, vix_df[['VIX_Close']], left_index=True, right_index=True, direction='backward')
            df['VIX_Close'] = df['VIX_Close'].ffill()
            features.append('VIX_Close')
            print("VIX merged")
        else:
            print("VIX empty")
            
    ml_df = df.dropna().copy()
    print("ML DF len:", len(ml_df))
    if len(ml_df) < 3:
        print("ML DF too short")
        exit()
        
    X = ml_df[features]
    y = ml_df['Target']
    
    # 高速化のため強制的にAutoTune無効化
    model = train_and_cache_model(tic, timeframe, start_date, end_date, False, X, y)
    print("Model trained!")
except Exception as e:
    import traceback
    traceback.print_exc()
