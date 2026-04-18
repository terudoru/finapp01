import sqlite3
import pandas as pd
import os
import json

DB_PATH = "finance_app.db"

def get_connection():
    """Streamlitのマルチスレッド環境に対応した安全なDBコネクションを取得"""
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    """データベースの初期化とテーブル作成"""
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # ポートフォリオテーブル
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS portfolio (
            id TEXT PRIMARY KEY,
            ticker TEXT,
            is_crypto INTEGER,
            purchase_date TEXT,
            purchase_time TEXT,
            purchase_price REAL,
            quantity REAL,
            notes TEXT
        )
        ''')
        
        # ブックマークテーブル
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS bookmarks (
            name TEXT PRIMARY KEY,
            ticker TEXT
        )
        ''')
        
        # 予測履歴テーブル (精度追跡用)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            ticker TEXT,
            predicted_trend INTEGER,
            probability REAL,
            actual_trend INTEGER NULL,
            is_correct INTEGER NULL
        )
        ''')

        # 設定テーブル (Webhook URL用)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        ''')
        conn.commit()

def migrate_from_json():
    """既存のJSONファイルからSQLiteへ移行"""
    init_db()
    
    # Bookmarks
    if os.path.exists("bookmarks.json"):
        with open("bookmarks.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            with get_connection() as conn:
                for name, ticker in data.items():
                    conn.execute("INSERT OR REPLACE INTO bookmarks (name, ticker) VALUES (?, ?)", (name, ticker))
                conn.commit()
            
    # Portfolio
    if os.path.exists("portfolio.json"):
        with open("portfolio.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            with get_connection() as conn:
                for item in data:
                    conn.execute('''
                    INSERT OR REPLACE INTO portfolio (id, ticker, is_crypto, purchase_date, purchase_time, purchase_price, quantity, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (item['id'], item['ticker'], 1 if item.get('is_crypto') else 0, item['purchase_date'], item['purchase_time'], item['purchase_price'], item['quantity'], item.get('notes', '')))
                conn.commit()

def get_bookmarks():
    with get_connection() as conn:
        df = pd.read_sql_query("SELECT * FROM bookmarks", conn)
    return dict(zip(df['name'], df['ticker']))

def add_bookmark(name, ticker):
    with get_connection() as conn:
        conn.execute("INSERT OR REPLACE INTO bookmarks (name, ticker) VALUES (?, ?)", (name, ticker))
        conn.commit()

def delete_bookmark(name):
    with get_connection() as conn:
        conn.execute("DELETE FROM bookmarks WHERE name = ?", (name,))
        conn.commit()

def get_portfolio():
    with get_connection() as conn:
        df = pd.read_sql_query("SELECT * FROM portfolio", conn)
    
    # Python側のロジックと互換性を保つため、boolに戻す等の変換
    records = df.to_dict('records')
    for r in records:
        r['is_crypto'] = bool(r.get('is_crypto', 0))
    return records

def add_portfolio_item(item):
    with get_connection() as conn:
        conn.execute('''
        INSERT OR REPLACE INTO portfolio (id, ticker, is_crypto, purchase_date, purchase_time, purchase_price, quantity, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (item['id'], item['ticker'], 1 if item.get('is_crypto') else 0, item['purchase_date'], item['purchase_time'], item['purchase_price'], item['quantity'], item.get('notes', '')))
        conn.commit()

def delete_portfolio_item(item_id):
    with get_connection() as conn:
        conn.execute("DELETE FROM portfolio WHERE id = ?", (item_id,))
        conn.commit()

def update_setting(key, value):
    with get_connection() as conn:
        conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, value))
        conn.commit()

def get_setting(key, default=None):
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM settings WHERE key = ?", (key,))
        row = cursor.fetchone()
    return row[0] if row else default

if __name__ == "__main__":
    migrate_from_json()
    print("Database initialized and migrated.")
