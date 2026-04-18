import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator

def add_time_series_features(df, use_sma=True, use_rsi=True, use_macd=True, use_bb=True):
    """
    株価データフレームに対して、プロ仕様のテクニカル指標とターゲット変数を追加する。
    """
    features = []
    
    # 基本リターン
    df['Daily_Return'] = df['Close'].pct_change()
    features.append('Daily_Return')
    
    # 時系列ラグ特徴量（過去3足分のリターン動向）
    for i in range(1, 4):
        col_name = f'Return_lag_{i}'
        df[col_name] = df['Daily_Return'].shift(i)
        features.append(col_name)
    
    # 出来高の変化率（データが存在する場合のみ）
    if 'Volume' in df.columns and (df['Volume'] > 0).any():
        df['Volume_Change'] = df['Volume'].pct_change()
        features.append('Volume_Change')
        # OBV 追加
        df['OBV'] = OnBalanceVolumeIndicator(close=df["Close"], volume=df["Volume"]).on_balance_volume()
        features.append('OBV')
        
    if use_sma:
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        # 乖離率 (モメンタム)
        df['Close_to_SMA20'] = (df['Close'] / df['SMA_20']) - 1.0
        features.extend(['SMA_20', 'SMA_50', 'Close_to_SMA20'])
        
    if use_rsi:
        indicator_rsi = RSIIndicator(close=df["Close"], window=14)
        df['RSI'] = indicator_rsi.rsi()
        features.append('RSI')
        
        # Stochastic Oscillator も追加
        indicator_stoch = StochasticOscillator(high=df["High"], low=df["Low"], close=df["Close"], window=14, smooth_window=3)
        df['Stoch_k'] = indicator_stoch.stoch()
        features.append('Stoch_k')
        
    if use_macd:
        indicator_macd = MACD(close=df["Close"])
        df['MACD'] = indicator_macd.macd()
        df['MACD_Signal'] = indicator_macd.macd_signal()
        features.extend(['MACD', 'MACD_Signal'])
        
    if use_bb:
        indicator_bb = BollingerBands(close=df["Close"], window=20, window_dev=2)
        df['BB_Width'] = indicator_bb.bollinger_wband()
        features.append('BB_Width')
        
    # ATR を特徴量としても追加
    df['ATR_Feature'] = AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"], window=14).average_true_range()
    features.append('ATR_Feature')
    
    # カレンダー特徴量
    df['DayOfWeek'] = df.index.dayofweek
    df['Is_Month_End'] = df.index.is_month_end.astype(int)
    features.extend(['DayOfWeek', 'Is_Month_End'])
        
    # ターゲット変数の作成（次の足で上がる(1)か下がる(0)か。ノイズ除去のため0.15%の閾値を設ける）
    df['Next_Close'] = df['Close'].shift(-1)
    df['Target'] = np.where(df['Next_Close'] > df['Close'] * 1.0015, 1, 0)
    
    return df, features


def get_xgb_model(X_train, y_train, auto_tune=True):
    """
    XGBoostモデルを構築・学習する。自動チューニングがオンの場合はGridSearchCVを使用。
    """
    if auto_tune and len(X_train) > 50:
        param_dist = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 4, 5, 6],
            'subsample': [0.8, 1.0]
        }
        xgb_base = XGBClassifier(random_state=42, eval_metric="logloss")
        tscv = TimeSeriesSplit(n_splits=5)
        rs = GridSearchCV(xgb_base, param_grid=param_dist, cv=tscv, scoring='accuracy', n_jobs=-1)
        rs.fit(X_train, y_train)
        return rs.best_estimator_
    else:
        model = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42, eval_metric="logloss")
        model.fit(X_train, y_train)
        return model
