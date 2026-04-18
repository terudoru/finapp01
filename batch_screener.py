import warnings
warnings.filterwarnings('ignore')

import datetime
import pandas as pd
import requests

from scripts.database import get_bookmarks, get_setting
from scripts.quant_engine import add_time_series_features, get_xgb_model
from predict import fetch_data, fetch_vix

# 通知関数の再定義（スタンドアロン動作用）
def send_notification(message):
    """Discord/Telegramに通知を送信する"""
    print("通知を送信中です...")
    
    # Discord
    discord_url = get_setting("discord_webhook")
    if discord_url:
        try:
            res = requests.post(discord_url, json={"content": message}, timeout=5)
            print(f"Discord通知結果: {res.status_code}")
        except Exception as e:
            print(f"Discord通知エラー: {e}")
            
    # Telegram
    token = get_setting("telegram_token")
    chat_id = get_setting("telegram_chat_id")
    if token and chat_id:
        try:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            res = requests.post(url, json={"chat_id": chat_id, "text": message}, timeout=5)
            print(f"Telegram通知結果: {res.status_code}")
        except Exception as e:
            print(f"Telegram通知エラー: {e}")

def main():
    print("================================")
    print("  🚀 AI Batch Screener Started")
    print(f"  {datetime.datetime.now()}")
    print("================================\n")
    
    bookmarks = get_bookmarks()
    if not bookmarks:
        print("ブックマークが登録されていません。ダッシュボードから銘柄を登録してください。")
        return
        
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=365*4)
    
    # VIXは一括取得
    vix_df = fetch_vix(start_date, end_date)
    
    results = []
    
    for name, ticker in bookmarks.items():
        is_crypto = "USDT" in ticker or "USD" in ticker
        print(f"[{ticker}] ({name}) を解析中...")
        
        try:
            df = fetch_data(ticker, start_date, end_date)
            if df.empty or len(df) < 50:
                print(f" -> データ不足スキップ")
                continue
                
            if not vix_df.empty and not is_crypto:
                df = df.join(vix_df, how='left')
                df['VIX_Close'] = df['VIX_Close'].ffill()
                
            df, features = add_time_series_features(df, use_sma=True, use_rsi=True, use_macd=True, use_bb=True)
            if 'VIX_Close' in df.columns: features.append('VIX_Close')
                
            ml_df = df.dropna().copy()
            if len(ml_df) < 30:
                continue
                
            X = ml_df[features]
            y = ml_df['Target']
            
            # 高速化のため自動チューニングはオフでベースモデルを使用
            model = get_xgb_model(X, y, auto_tune=False)
            
            latest_features = ml_df.iloc[-1:][features]
            prob_up = model.predict_proba(latest_features)[0][1] * 100
            
            curr_price = df.iloc[-1]['Close']
            
            results.append({
                "name": name,
                "ticker": ticker,
                "price": curr_price,
                "prob": prob_up
            })
            
        except Exception as e:
            print(f" -> エラー: {e}")

    # 結果をまとめる
    if not results:
        print("有効な結果が得られませんでした。")
        return
        
    # 上昇確率が高い順にソート (上位3つ、かつ確率60%以上をピックアップ)
    results.sort(key=lambda x: x["prob"], reverse=True)
    high_conviction = [r for r in results if r["prob"] >= 60.0]
    
    if high_conviction:
        msg = "🚀 **本日のAI強気シグナル検知！**\n\n"
        for r in high_conviction[:3]:
            msg += f"・**{r['name']} ({r['ticker']})**: 現在価格 ${r['price']:.2f} ➡ 上昇確率 **{r['prob']:.1f}%**\n"
            
        msg += "\n*※ 詳細はダッシュボードのバックテスト等で最終確認してください。*"
        print("\n" + msg)
        
        # 設定があれば送信
        send_notification(msg)
    else:
        print("\n本日は強いサイン (60%以上) の銘柄はありませんでした。")

if __name__ == "__main__":
    main()
