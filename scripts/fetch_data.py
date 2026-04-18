import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def fetch_stock_data(ticker='AAPL', years=5):
    # 過去5年の日付範囲を設定
    end_date = datetime.today()
    start_date = end_date - timedelta(days=years*365)

    # 株価データを取得
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            raise ValueError(f'No data found for ticker: {ticker}')

        # データを保存 (CSV形式)
        data.to_csv('data/stock_data.csv')

        # SQLiteにも保存する場合（オプション）
        # data.to_sql('stock_data', pd.sqlite3.connect('data/stock.db'), if_exists='replace')

        return data
    except Exception as e:
        print(f'Error fetching data: {str(e)}')
        return None

if __name__ == '__main__':
    # テスト実行
    data = fetch_stock_data()
    if data is not None:
        print('Data fetched successfully!')
