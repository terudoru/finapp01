import warnings
warnings.filterwarnings('ignore')

import yfinance as yf
import pandas as pd
import numpy as np
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import datetime

from scripts.quant_engine import add_time_series_features, get_xgb_model
from scripts.sentiment_engine import analyze_sentiment

def fetch_data(ticker, start_date, end_date):
    """Yahoo Financeから株価データを取得します"""
    print(f"[{ticker}] のデータを取得中...")
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    print(f"データ取得完了: {len(df)}行")
    return df

def fetch_vix(start_date, end_date):
    """VIX指数（恐怖指数）を取得します"""
    try:
        vix = yf.download("^VIX", start=start_date, end=end_date, progress=False)
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.get_level_values(0)
        return vix[['Close']].rename(columns={'Close': 'VIX_Close'})
    except:
        return pd.DataFrame()

def main():
    print("🌟 株価予測プロジェクト (XGBoost版 CLI) 🌟\n")
    ticker = 'AAPL'
    
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=365*4)
    
    # 1. データ取得
    df = fetch_data(ticker, start_date, end_date)
    vix_df = fetch_vix(start_date, end_date)
    if not vix_df.empty:
        df = df.join(vix_df, how='left')
        df['VIX_Close'] = df['VIX_Close'].ffill()
    
    # 2. 特徴量生成 & ターゲット生成 (quant_engineを使用)
    print("テクニカル指標 (SMA, RSI, MACD, BB, ATR, Stoch, OBV) を計算中...")
    df, features = add_time_series_features(df, use_sma=True, use_rsi=True, use_macd=True, use_bb=True)
    
    if 'VIX_Close' in df.columns:
        features.append('VIX_Close')
        
    df = df.dropna()
    if len(df) < 50:
        print("データが少なすぎるため終了します。")
        return
        
    X = df[features]
    y = df['Target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    print(f"学習データ: {len(X_train)} 日分")
    print(f"テストデータ: {len(X_test)} 日分")
    
    # 5. XGBoostモデルの構築とハイパーパラメータの自動最適化
    print("\n🌲 ハイパーパラメータの網羅的自動最適化を開始... (負荷高め/精度重視)")
    model = get_xgb_model(X_train, y_train, auto_tune=True)
    
    # 6. 予測と精度出力
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print("\n" + "="*30)
    print("   📊 バックテスト結果 📊")
    print("="*30)
    print(f"🎯 正答率 (Accuracy): {accuracy * 100:.2f} %")
    print("-" * 30)
    print("【分類レポート】")
    print(classification_report(y_test, predictions, target_names=["下落(0)", "上昇(1)"]))
    
    # 最新のニュース分析 (FinBERT連携)
    print("\n" + "="*30)
    print(" 📰 最新ニュース感情分析 (AI)")
    print("="*30)
    try:
        ticker_obj = yf.Ticker(ticker)
        news_data = ticker_obj.news
        if news_data and len(news_data) > 0:
            titles = [article.get('title', '') for article in news_data[:5] if article.get('title', '')]
            adv_score, adv_mood = analyze_sentiment(titles, use_finbert=True)
            print(f"総合判定: [{adv_mood}] (AIスコア: {adv_score:+.2f})")
            print("【分析対象のヘッドライン】")
            for t in titles[:3]:
                print(f" - {t}")
        else:
            print("ニュースが見つかりませんでした。")
    except Exception as e:
        print(f"ニュース情報の取得に失敗しました: {e}")
    
    # 最新のリスク管理提案
    latest_close = df.iloc[-1]['Close']
    latest_atr = df.iloc[-1]['ATR_Feature'] if 'ATR_Feature' in df.columns else df.iloc[-1]['ATR']
    stop_loss = latest_close - (1.5 * latest_atr)
    take_profit = latest_close + (2.5 * latest_atr)
    
    # SHAP分析 (説明可能性)
    print("\n" + "="*30)
    print(" 🧠 AI予測の根拠 (SHAP分析)")
    print("="*30)
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    last_shap = shap_values[-1]
    top_features = sorted(zip(features, last_shap.values), key=lambda x: abs(x[1]), reverse=True)[:3]
    print("最新の予測に最も影響を与えた指標:")
    for f, val in top_features:
        impact = "上昇要因" if val > 0 else "下落要因"
        print(f" - {f}: {impact} ({val:.4f})")
    
    print("\n" + "="*30)
    print(" 💡 トレード・リスク管理 (参考)")
    print("="*30)
    print(f"本日の終値: ${latest_close:.2f}")
    if not np.isnan(latest_atr):
        print(f"推奨 利確ライン (Take Profit): ${take_profit:.2f}")
        print(f"推奨 損切りライン (Stop Loss) : ${stop_loss:.2f}")
    
    print("\n✅ コマンドラインテストが完了しました！GUIは streamlit run streamlit_app.py で確認してください。")

if __name__ == "__main__":
    main()
