import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import datetime
import ccxt
import json
import os
from scripts.database import init_db, migrate_from_json, get_bookmarks, add_bookmark, delete_bookmark, get_portfolio, add_portfolio_item, delete_portfolio_item, get_setting, update_setting
from scripts.sentiment_engine import analyze_sentiment, summarize_news
from scripts.quant_engine import add_time_series_features, get_xgb_model
import shap
import matplotlib.pyplot as plt
# --- ページ設定 ---
st.set_page_config(page_title="株価分析＆AI予測ダッシュボード", page_icon="📈", layout="wide")

# --- 🛰️ カスタムスタイル (CSS) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&family=Montserrat:wght@700&display=swap');
    
    html, body, [class*="ViewContainer"], .stApp, p, span, div, label {
        font-family: 'Inter', sans-serif;
        color: #ffffff !important;
    }
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgba(15, 23, 42, 1) 0%, rgba(2, 6, 23, 1) 90.2%);
    }
    .stMetric {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 15px;
        border-radius: 12px;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    .stMetric:hover {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(56, 189, 248, 0.3);
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.8; }
        100% { opacity: 1; }
    }
    .stMetric div[data-testid="stMetricValue"] {
        animation: pulse 4s infinite ease-in-out;
    }
    .stSidebar {
        background-color: rgba(15, 23, 42, 0.8) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    h1, h2, h3 {
        font-family: 'Montserrat', sans-serif;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .css-1r6p8d1 {
        border-radius: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    /* Buttons */
    .stButton>button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(56, 189, 248, 0.4);
    }

    /* === モバイル対応 (Mobile Responsive CSS) === */
    @media (max-width: 768px) {
        .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 2rem !important;
            padding-left: 0.8rem !important;
            padding-right: 0.8rem !important;
        }
        h1 {
            font-size: 1.5rem !important;
        }
        h2 {
            font-size: 1.3rem !important;
        }
        h3 {
            font-size: 1.1rem !important;
        }
        .stMetric {
            padding: 10px !important;
        }
        /* Ensure data structures are fully scrollable */
        [data-testid="stDataFrame"], [data-testid="stTable"] {
            overflow-x: auto !important;
        }
        /* Larger tap areas for buttons on mobile */
        .stButton > button {
            width: 100% !important;
            min-height: 3rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ 株価予測プラットフォーム - AI & Quant Intelligence")
st.markdown("""
<div style="background: rgba(56, 189, 248, 0.1); border-left: 5px solid #38bdf8; padding: 15px; border-radius: 5px; margin-bottom: 25px;">
    <strong>QUANT VIEW:</strong> 機械学習（XGBoost）と世界の経済指標、マクロ感応度を統合したツールです。
</div>
""", unsafe_allow_html=True)

# (Market Pulse section moved down to where functions are accessible)

# --- 初心者向け用語集 ---
with st.expander("📚 投資・AI用語集（初心者の方はまずこちらをお読みください）"):
    st.markdown("""
    - **SMA (単純移動平均線)**: 過去の株価の平均を結んだ線。トレンド（上昇中か下落中か）を見るのに使います。
    - **RSI (相対力指数)**: 「買われすぎ（70以上）」「売られすぎ（30以下）」を示す指標です。逆張りの目安になります。
    - **MACD (マックディー)**: トレンドの方向性や転換点を捉えるための指標です。プロのトレーダーもよく使います。
    - **ボリンジャーバンド**: 株価がどのくらいの範囲に収まりやすいかを統計的に示した帯（バンド）です。
    - **XGBoost (エックスジーブースト)**: データ分析コンペなどで優れた実績を持つ、強力な機械学習アルゴリズムです。過去のパターンから高精度な予測を行います。
    - **ATR (アベレージ・トゥルー・レンジ)**: 1日にだいたいどれくらい株価が動くか（ボラティリティ＝値幅）を算出した指標です。プロはこれを基準に利確や損切りの位置を決めます。
    - **バックテスト**: その投資手法（今回はAIの予測）に従って過去からずっと売買していたら、資産がどのくらい増減していたかをシミュレーションすることです。
    - **ニュース感情分析 (VADER)**: SNSやニュースなどのテキストから、世間のムードが「楽観的（ポジティブ）」か「悲観的（ネガティブ）」かをスコア化するAI技術です。
    """)

# --- サイドバー (入力情報のUI) ---
# --- サイドバー (入力情報のUI) ---
app_mode = st.sidebar.radio("🔍 アプリモード", ["📊 個別銘柄の詳細分析", "🧪 バックテスト・検証機能", "🏆 AI 一斉スクリーナー (買い/売り)", "💼 ポートフォリオ管理", "⚙️ 設定"])
st.sidebar.markdown("---")

st.sidebar.header("📊 パラメータ設定")

# データベース初期化
init_db()
if not os.path.exists("finance_app.db_migrated"):
    migrate_from_json()
    with open("finance_app.db_migrated", "w") as f: f.write("done")

# ウォッチリストの定義
custom_bookmarks = get_bookmarks()

WATCHLIST = {
    "⭐ マイ・ブックマーク": custom_bookmarks,
    "🇺🇸 米国株": {
        "Apple (AAPL)": "AAPL",
        "Microsoft (MSFT)": "MSFT",
        "NVIDIA (NVDA)": "NVDA",
        "Tesla (TSLA)": "TSLA",
        "Amazon (AMZN)": "AMZN",
        "S&P 500 ETF (SPY)": "SPY"
    },
    "🇯🇵 日本株": {
        "トヨタ自動車 (7203)": "7203.T",
        "三菱UFJ (8306)": "8306.T",
        "ソニー (6758)": "6758.T",
        "ソフトバンクG (9984)": "9984.T"
    },
    "🪙 仮想通貨": {
        "Bitcoin (BTC/USDT)": "BTC/USDT",
        "Ethereum (ETH/USDT)": "ETH/USDT",
        "Solana (SOL/USDT)": "SOL/USDT",
        "Ripple (XRP/USDT)": "XRP/USDT"
    },
    "🛢️ 先物・コモディティ": {
        "金・ゴールド (GC=F)": "GC=F",
        "WTI原油 (CL=F)": "CL=F",
        "天然ガス (NG=F)": "NG=F",
        "S&P 500 先物 (ES=F)": "ES=F"
    },
    "✍️ 自分で入力 (カスタム)": {}
}

selected_category = st.sidebar.selectbox("📈 アセット（カテゴリ）を選択", list(WATCHLIST.keys()))

if selected_category == "✍️ 自分で入力 (カスタム)":
    ticker = st.sidebar.text_input("銘柄ティッカーを手動で入力", value="AAPL")
    
    # ブックマーク保存UI
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🌟 頻繁に見る銘柄を保存**")
    bm_name = st.sidebar.text_input("表示名 (例: Google株)")
    if st.sidebar.button("ブックマークに追加 (⭐)"):
        if bm_name and ticker:
            add_bookmark(bm_name, ticker)
            st.sidebar.success(f"「{bm_name}」を保存しました！")
            st.rerun()
        else:
            st.sidebar.warning("表示名とティッカーの両方を入力してください。")
elif selected_category == "⭐ マイ・ブックマーク" and not custom_bookmarks:
    st.sidebar.warning("まだブックマークがありません。「自分で入力」から追加してください。")
    ticker = "AAPL" # fallback
else:
    options = WATCHLIST[selected_category]
    selected_name = st.sidebar.selectbox("🏢 ウォッチリストから銘柄を選択", list(options.keys()))
    ticker = options[selected_name]

# 期間の設定
today = datetime.date.today()
default_start = today - datetime.timedelta(days=365*4) # 4年前
start_date = st.sidebar.date_input("開始日 (日足用)", value=default_start)
end_date = st.sidebar.date_input("終了日 (日足用)", value=today)

st.sidebar.markdown("---")
st.sidebar.header("⏱️ タイムフレーム設定")
timeframe_opts = {"1日 (日足)": "1d", "1時間 (スイング)": "1h", "5分 (デイトレ)": "5m"}
selected_tf_label = st.sidebar.selectbox("データ間隔", list(timeframe_opts.keys()))
timeframe = timeframe_opts[selected_tf_label]
is_crypto = (selected_category == "🪙 仮想通貨")

st.sidebar.markdown("---")
st.sidebar.header("⚙️ 機械学習の設定")

st.sidebar.markdown("**予測に使用する特徴量（指標）**")
use_sma = st.sidebar.checkbox("移動平均線 (SMA 20, SMA 50)", value=True)
use_rsi = st.sidebar.checkbox("RSI (買われすぎ/売られすぎ指標)", value=True)
use_macd = st.sidebar.checkbox("MACD (トレンド転換指標)", value=True)
use_bb = st.sidebar.checkbox("ボリンジャーバンド幅", value=True)

st.sidebar.markdown('---')
use_news = st.sidebar.checkbox('📰 ニュース感情分析を予測に加味する', value=True)

st.sidebar.markdown('---')
st.sidebar.markdown('**🧠 学習設定**')
auto_tune = st.sidebar.checkbox('🤖 ハイパーパラメータ自動最適化 (Auto-Tune)', value=True, help='処理時間が数秒伸びますが精度が向上します。一斉スクリーナーの際は時間がかかるためオフを推奨します。')

if app_mode == "📊 個別銘柄の詳細分析":
    run_button = st.sidebar.button("AIの予測とシグナル表示を実行", type="primary")
elif app_mode == "🧪 バックテスト・検証機能":
    run_button = st.sidebar.button("🔬 詳細なバックテストを実行", type="primary")
elif app_mode == "🏆 AI 一斉スクリーナー (買い/売り)":
    run_button = st.sidebar.button("🚀 今すぐ一斉スキャンを開始", type="primary")
elif app_mode == "⚙️ 設定":
    run_button = False
else:
    run_button = False  # ポートフォリオモードはボタン不要

# --- メイン処理 ---

@st.cache_data(ttl="15m")
def get_ticker_info(ticker):
    """
    銘柄の基本情報を取得（キャッシュ対応）
    """
    try:
        t_obj = yf.Ticker(ticker)
        return t_obj.info
    except:
        return {}

@st.cache_data(ttl="15m")
def fetch_data(t, start, end, tf, is_crypto):
    if is_crypto:
        exchange = ccxt.binance()
        # BinanceからCCXTを使ってOHLCVを取得
        ohlcv = exchange.fetch_ohlcv(t, timeframe=tf, limit=1000)
        df = pd.DataFrame(ohlcv, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['Date'] = pd.to_datetime(df['Date'], unit='ms')
        df.set_index('Date', inplace=True)
        return df
    else:
        # yfinanceの制限対応
        target_start = start
        if tf == '1m':
            target_start = max(start, today - datetime.timedelta(days=6))
        elif tf == '5m':
            target_start = max(start, today - datetime.timedelta(days=59))
        elif tf == '1h':
            target_start = max(start, today - datetime.timedelta(days=729))
            
        df = yf.download(t, start=target_start, end=end + datetime.timedelta(days=1), interval=tf, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df

@st.cache_data(ttl="1h")
def fetch_macro_data(start, end):
    macros = {'米ドル指数 (DXY)': 'DX-Y.NYB', '米10年債利回り': '^TNX', 'S&P 500': '^GSPC', '金先物': 'GC=F'}
    df_m = pd.DataFrame()
    for name, t_m in macros.items():
        try:
            d = yf.download(t_m, start=start, end=end + datetime.timedelta(days=1), progress=False)
            if not d.empty:
                if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
                df_m[name] = d['Close']
        except: pass
    return df_m

@st.cache_data(ttl="30m")
def fetch_vix(start, end):
    try:
        vix = yf.download("^VIX", start=start, end=end + datetime.timedelta(days=1), interval='1d', progress=False)
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.get_level_values(0)
        return vix[['Close']].rename(columns={'Close': 'VIX_Close'})
    except:
        return pd.DataFrame()

@st.cache_data(ttl="5m")
def fetch_live_price(t, is_crypto):
    try:
        if is_crypto:
            exchange = ccxt.binance()
            ticker_info = exchange.fetch_ticker(t)
            return ticker_info.get('last'), ticker_info.get('percentage')
        else:
            fi = yf.Ticker(t).fast_info
            last_price = fi.get('lastPrice', 0.0)
            prev_close = fi.get('previousClose', last_price)
            pct_change = ((last_price / prev_close) - 1.0) * 100 if prev_close else 0.0
            return last_price, pct_change
    except:
        return None, None

def send_notification(message):
    """Discord/Telegramに通知を送信する"""
    import requests
    # Discord
    discord_url = get_setting("discord_webhook")
    if discord_url:
        try:
            requests.post(discord_url, json={"content": message}, timeout=5)
        except: pass
    
    # Telegram
    token = get_setting("telegram_token")
    chat_id = get_setting("telegram_chat_id")
    if token and chat_id:
        try:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            requests.post(url, json={"chat_id": chat_id, "text": message}, timeout=5)
        except: pass

# ============================================================
# ポートフォリオ管理モード
# ============================================================
PORTFOLIO_FILE = "portfolio.json"

def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return []
    return []

@st.cache_data
def calculate_macro_sensitivity(ticker, days=180):
    """
    対象銘柄と主要マクロ指標の相関（感応度）を計算する
    """
    macro_symbols = {
        "S&P 500 (市場全体)": "^GSPC",
        "WTI 原油 (エネルギー)": "CL=F",
        "Gold (安全資産)": "GC=F",
        "VIX (恐怖指数)": "^VIX"
    }
    
    end_date_macro = datetime.date.today()
    start_date_macro = end_date_macro - datetime.timedelta(days=days)
    
    # 対象銘柄のデータを取得
    try:
        target_df = fetch_data(ticker, start_date_macro, end_date_macro, '1d', ("/USDT" in ticker))
        if target_df.empty: return None
        target_returns = target_df['Close'].pct_change().dropna()
    except: return None
    
    results = {}
    for name, sym in macro_symbols.items():
        try:
            m_df = fetch_data(sym, start_date_macro, end_date_macro, '1d', False)
            if not m_df.empty:
                m_returns = m_df['Close'].pct_change().dropna()
                # 直近の共通期間で結合
                combined = pd.concat([target_returns, m_returns], axis=1).dropna()
                if not combined.empty:
                    corr = combined.iloc[:, 0].corr(combined.iloc[:, 1])
                    results[name] = corr
        except: continue
        
    return results

@st.cache_data(ttl="15m")
def get_portfolio_events(ticker_list):
    """
    ポートフォリオ全体銘柄のイベント（決算・配当等）を取得・集計する
    """
    events = []
    for t in ticker_list:
        if "/" in t: continue # 仮想通貨はスキップ
        cal = get_corporate_events(t)
        if cal:
            d_date = cal.get('Dividend Date')
            if d_date:
                events.append({"銘柄": t, "イベント": "配当落ち(予定)", "日付": d_date})
            e_dates = cal.get('Earnings Date')
            if e_dates and isinstance(e_dates, list):
                for ed in e_dates:
                    events.append({"銘柄": t, "イベント": "決算発表(予定)", "日付": ed})
    
    if not events: return None
    # 日付順にソートし、未来のものだけ抽出（過去のものは直近のものだけに限定）
    edf = pd.DataFrame(events)
    today_dt = datetime.date.today()
    # 今日以降の予定
    upcoming = edf[edf['日付'] >= today_dt].sort_values('日付')
    return upcoming

def save_portfolio(portfolio_data):
    with open(PORTFOLIO_FILE, "w", encoding="utf-8") as f:
        json.dump(portfolio_data, f, ensure_ascii=False, indent=4)

def lookup_price_at_time(ticker_sym, purchase_dt_naive, is_crypto_flag):
    """東京時間の購入日時から、その時点の約定価格を逆算する"""
    tokyo_tz = pytz.timezone('Asia/Tokyo')
    purchase_dt = tokyo_tz.localize(purchase_dt_naive)

    if is_crypto_flag:
        try:
            exchange = ccxt.binance()
            since_ms = int(purchase_dt.timestamp() * 1000)
            ohlcv = exchange.fetch_ohlcv(ticker_sym, timeframe='1m', since=since_ms, limit=5)
            if ohlcv and len(ohlcv) > 0:
                return float(ohlcv[0][4])
        except:
            pass
        return None
    else:
        now_tokyo = datetime.datetime.now(tokyo_tz)
        days_ago = (now_tokyo - purchase_dt).days
        d = purchase_dt.date()
        start_d = d - datetime.timedelta(days=1)
        end_d = d + datetime.timedelta(days=2)

        if days_ago <= 6:
            iv = '1m'
        elif days_ago <= 59:
            iv = '5m'
        elif days_ago <= 729:
            iv = '1h'
        else:
            iv = '1d'

        try:
            df_p = yf.download(ticker_sym, start=start_d, end=end_d, interval=iv, progress=False)
            if isinstance(df_p.columns, pd.MultiIndex):
                df_p.columns = df_p.columns.get_level_values(0)
            if df_p.empty:
                df_p = yf.download(ticker_sym, start=d, end=d + datetime.timedelta(days=1), progress=False)
                if isinstance(df_p.columns, pd.MultiIndex):
                    df_p.columns = df_p.columns.get_level_values(0)
            if df_p.empty:
                return None
            if iv == '1d':
                return float(df_p.iloc[0]['Close'])
            if df_p.index.tz is not None:
                df_p.index = df_p.index.tz_convert(tokyo_tz)
            target_ts = purchase_dt.timestamp()
            diffs = [abs(idx.timestamp() - target_ts) for idx in df_p.index]
            closest_idx = diffs.index(min(diffs))
            return float(df_p.iloc[closest_idx]['Close'])
        except:
            return None

@st.cache_data(ttl="30m")
def get_corporate_events(t):
    try:
        t_obj = yf.Ticker(t)
        cal = t_obj.calendar
        return cal
    except: return None

@st.cache_data(ttl="30m")
def calculate_correlations(ticker_list, days=90):
    if not ticker_list: return None
    # yfinance用にティッカーを変換 (BTC/USDT -> BTC-USD)
    yf_tickers = [t.replace('/', '-').replace('USDT', 'USD') for t in ticker_list]
    start = datetime.date.today() - datetime.timedelta(days=days)
    try:
        data = yf.download(yf_tickers, start=start, progress=False)['Close']
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
        # ラベルを元のティッカーに戻す
        col_map = dict(zip(yf_tickers, ticker_list))
        data = data.rename(columns=col_map)
        return data.pct_change().corr()
    except: return None

def compute_risk_metrics(t_history, benchmark_history):
    """Beta, Volatility, VaRを計算"""
    r_t = t_history.pct_change().dropna()
    r_b = benchmark_history.pct_change().dropna()
    
    # 共通の期間を抽出 (Indexが重複している部分のみ)
    common = r_t.index.intersection(r_b.index)
    if len(common) < 20: return 0.0, 0.0, 0.0
    
    # Beta (コバリアンス / ベンチマークのバリアンス)
    var_b = r_b.loc[common].var()
    cov = r_t.loc[common].cov(r_b.loc[common])
    beta = cov / var_b if var_b != 0 else 1.0
    
    # Vol (Annualized)
    vol = r_t.std() * np.sqrt(252)
    
    # 95% VaR (Historical)
    var_95 = np.percentile(r_t, 5) if len(r_t) > 0 else 0.0
    
    return float(beta), float(vol), float(var_95)

# --- 🌍 市場全体のムード (Market Pulse) ---
st.markdown("### 🌍 市場全体のトレンド (Market Pulse)")
macro_start = today - datetime.timedelta(days=30)
df_pulse = fetch_macro_data(macro_start, today)
if not df_pulse.empty:
    cols = st.columns(len(df_pulse.columns))
    for i, col_name in enumerate(df_pulse.columns):
        current_v = df_pulse[col_name].iloc[-1]
        prev_v = df_pulse[col_name].iloc[-2]
        delta_v = ((current_v / prev_v) - 1.0) * 100
        cols[i].metric(col_name, f"{current_v:,.2f}", f"{delta_v:+.2f}%")
else:
    st.info("マクロデータの取得中、または市場が閉まっています。")

# ============================================================
# 設定モード
# ============================================================
if app_mode == "⚙️ 設定":
    st.header("⚙️ システム設定")
    
    st.subheader("🔔 通知設定 (Webhook)")
    st.markdown("シグナル発生時に自動で通知を送るための設定です。")
    
    discord_url = st.text_input("Discord Webhook URL", value=get_setting("discord_webhook", ""))
    if st.button("Discord設定を保存"):
        update_setting("discord_webhook", discord_url)
        st.success("保存しました。")
        
    telegram_token = st.text_input("Telegram Bot Token", value=get_setting("telegram_token", ""))
    telegram_chat_id = st.text_input("Telegram Chat ID", value=get_setting("telegram_chat_id", ""))
    if st.button("Telegram設定を保存"):
        update_setting("telegram_token", telegram_token)
        update_setting("telegram_chat_id", telegram_chat_id)
        st.success("保存しました。")

    st.markdown("---")
    st.subheader("🤖 AI分析設定")
    use_finbert_pref = st.checkbox("AI感情分析 (FinBERT) を使用する", value=(get_setting("use_finbert", "1") == "1"), help="オフにすると軽量なVADERを使用します。")
    if st.button("AI設定を保存"):
        update_setting("use_finbert", "1" if use_finbert_pref else "0")
        st.success("保存しました。")

    st.markdown("---")
    st.subheader("💾 データベース管理")
    if st.button("⚠️ 全データをバックアップ (JSON)"):
        # 実装略、DBの内容をJSONで掃き出す処理
        st.info("この機能は現在準備中です。")

    st.stop()

if app_mode == "💼 ポートフォリオ管理":
    st.header("💼 マイ・ポートフォリオ")
    st.markdown("購入履歴を登録し、現在の損益と **AIによる売却タイミングのアドバイス** を確認できます。")

    portfolio = load_portfolio()

    # --- 🏦 ポートフォリオ分散分析 (New Feature) ---
    if portfolio:
        st.markdown("---")
        st.subheader("📊 ポートフォリオ分散分析 (セクター別)")
        
        with st.spinner("セクター情報を取得・集計中..."):
            sector_data = []
            for entry in portfolio:
                tic = entry["ticker"]
                is_cry = entry.get("is_crypto", False)
                qty = entry["quantity"]
                
                # 現在価格を取得
                curr_p = 0.0
                try:
                    if is_cry:
                        curr_p, _ = fetch_live_price(tic, True)
                    else:
                        ticker_df = fetch_data(tic, today - datetime.timedelta(days=5), today, '1d', False)
                        curr_p = ticker_df.iloc[-1]['Close'] if not ticker_df.empty else entry["purchase_price"]
                except:
                    curr_p = entry["purchase_price"]
                
                eval_val = curr_p * qty
                
                # セクター情報を取得
                if is_cry:
                    sector = "仮想通貨"
                else:
                    info = get_ticker_info(tic)
                    sector = info.get('sectorDisp', info.get('sector', 'その他/不明'))
                
                sector_data.append({"Sector": sector, "Value": eval_val})
            
            if sector_data:
                sdf = pd.DataFrame(sector_data)
                sdf = sdf.groupby("Sector")["Value"].sum().reset_index()
                
                fig_pie = px.pie(sdf, values='Value', names='Sector', 
                                 title="セクター別・評価額構成比",
                                 hole=0.4,
                                 color_discrete_sequence=px.colors.sequential.Skyblue_r)
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                fig_pie.update_layout(margin=dict(l=20, r=20, t=50, b=20), height=450)
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("集計可能なデータがありません。")

        # --- 📈 ポートフォリオ相関ヒートマップ (NEW) ---
        st.markdown("---")
        st.subheader("📈 保有銘柄の相関分析")
        st.markdown("保有銘柄同士がどれだけ「同じ方向へ動くか」を分析します。相関が高い（赤色に近い）銘柄ペアは、同時に下落するリスクがあります。")
        portfolio_tickers = list(set([e['ticker'] for e in portfolio]))
        if len(portfolio_tickers) > 1:
            with st.spinner("相関データを計算中..."):
                corr_pf = calculate_correlations(portfolio_tickers)
                if corr_pf is not None:
                    fig_corr_pf = go.Figure(data=go.Heatmap(
                        z=corr_pf.values, x=corr_pf.columns, y=corr_pf.columns,
                        colorscale='RdBu_r', zmin=-1, zmax=1
                    ))
                    fig_corr_pf.update_layout(height=450, margin=dict(l=20, r=20, t=20, b=20))
                    st.plotly_chart(fig_corr_pf, use_container_width=True)
                else:
                    st.warning("相関データの取得に失敗しました。")
        else:
            st.info("相関を分析するには2つ以上の異なる銘柄が必要です。")

        # --- 📅 ポートフォリオ・イベントカレンダー (NEW) ---
        st.markdown("---")
        st.subheader("📅 保有銘柄の予定スケジュール (決算・配当)")
        with st.spinner("イベントスケジュールを取得中..."):
            stock_tickers = [e['ticker'] for e in portfolio if not e.get('is_crypto', False)]
            if stock_tickers:
                upcoming_events = get_portfolio_events(stock_tickers)
                if upcoming_events is not None and not upcoming_events.empty:
                    st.dataframe(upcoming_events, use_container_width=True, hide_index=True)
                    csv_events = upcoming_events.to_csv(index=False).encode('utf-8-sig')
                    st.download_button("📥 予定表データをCSVで保存", data=csv_events, file_name="portfolio_events.csv", mime="text/csv")
                else:
                    st.info("近日の決算・配当予定は見つかりませんでした。")
            else:
                st.info("仮想通貨以外の銘柄が登録されていません。")


    # --- 新規購入の登録フォーム ---
    with st.expander("📝 新規購入を記録する", expanded=(len(portfolio) == 0)):
        st.markdown("購入した時刻（東京時間）を入力すると、その時点の株価を自動で取得し、購入単価として記録します。")
        col_t, col_d, col_time = st.columns(3)
        with col_t:
            new_ticker = st.text_input("銘柄ティッカー", value=ticker, key="pf_ticker")
        with col_d:
            new_date = st.date_input("購入日 (東京時間)", value=today, key="pf_date")
        with col_time:
            new_time = st.time_input("購入時刻 (東京時間)", value=datetime.time(10, 0), step=60, key="pf_time")

        col_q, col_n = st.columns(2)
        with col_q:
            new_quantity = st.number_input("購入数量 (株/枚)", min_value=0.0001, value=1.0, step=0.01, key="pf_qty")
        with col_n:
            new_notes = st.text_input("メモ (任意)", value="", key="pf_notes")

        is_new_crypto = ("/" in new_ticker and "USDT" in new_ticker.upper())

        col_price_input, col_btn = st.columns(2)
        with col_price_input:
            manual_price = st.number_input("購入単価を手動入力 (0=自動取得)", min_value=0.0, value=0.0, step=0.01, key="pf_manual_price")
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            add_btn = st.button("✅ この購入を記録する", type="primary", key="pf_add_btn")

        if add_btn:
            purchase_dt_naive = datetime.datetime.combine(new_date, new_time)
            if manual_price > 0:
                fetched_price = manual_price
            else:
                with st.spinner("購入時刻の株価を取得中..."):
                    fetched_price = lookup_price_at_time(new_ticker, purchase_dt_naive, is_new_crypto)

            if fetched_price is not None and fetched_price > 0:
                new_entry = {
                    "id": str(len(portfolio) + 1) + "_" + new_ticker + "_" + str(new_date),
                    "ticker": new_ticker,
                    "is_crypto": is_new_crypto,
                    "purchase_date": str(new_date),
                    "purchase_time": str(new_time),
                    "purchase_price": round(fetched_price, 6),
                    "quantity": new_quantity,
                    "notes": new_notes
                }
                portfolio.append(new_entry)
                save_portfolio(portfolio)
                st.success(f"✅ 記録完了！ **{new_ticker}** を **${fetched_price:,.4f}** × {new_quantity} で登録しました。")
                st.rerun()
            else:
                st.error("❌ その日時の株価データを取得できませんでした。購入単価を手動で入力してください。")

    st.markdown("---")

    # --- ポートフォリオ一覧表示 ---
    if not portfolio:
        st.info("まだ購入記録がありません。上のフォームから記録を追加してください。")
        st.stop()

    st.header("📊 保有銘柄一覧 & 損益")

    rows = []
    total_invested = 0.0
    total_current = 0.0
    price_cache = {}  # 同一ティッカーの重複取得を防ぐ

    # ベンチマーク(S&P 500)のデータ取得 (リスク計算用)
    bench_df = yf.download("^GSPC", start=today - datetime.timedelta(days=365), end=today, progress=False)['Close']
    if isinstance(bench_df.columns, pd.MultiIndex): bench_df.columns = bench_df.columns.get_level_values(0)

    with st.spinner("現在の価格とリスク分析を取得中..."):
        for entry in portfolio:
            t_pf = entry["ticker"]
            is_cry_pf = entry.get("is_crypto", False)
            if t_pf not in price_cache:
                live_p, _ = fetch_live_price(t_pf, is_cry_pf)
                price_cache[t_pf] = live_p
                # リスクメトリクスの計算
                try:
                    t_hist = yf.download(t_pf, start=today - datetime.timedelta(days=365), end=today, progress=False)['Close']
                    if isinstance(t_hist.columns, pd.MultiIndex): t_hist.columns = t_hist.columns.get_level_values(0)
                    beta, vol, var95 = compute_risk_metrics(t_hist, bench_df)
                except: beta, vol, var95 = 0, 0, 0
                price_cache[f"{t_pf}_risk"] = (beta, vol, var95)
            else:
                live_p = price_cache[t_pf]
                beta, vol, var95 = price_cache[f"{t_pf}_risk"]

            buy_price = entry["purchase_price"]
            qty = entry["quantity"]
            invested = buy_price * qty
            if live_p and live_p > 0:
                current_val = live_p * qty
                pnl = current_val - invested
                pnl_pct = ((live_p / buy_price) - 1.0) * 100
            else:
                live_p = 0
                current_val = 0
                pnl = -invested
                pnl_pct = -100.0

            total_invested += invested
            total_current += current_val

            rows.append({
                "銘柄": t_pf,
                "購入日時 (東京)": f"{entry['purchase_date']} {entry['purchase_time']}",
                "購入単価": f"${buy_price:,.4f}",
                "数量": qty,
                "投資額": f"${invested:,.2f}",
                "現在値": f"${live_p:,.4f}" if live_p else "取得不可",
                "評価額": f"${current_val:,.2f}",
                "損益": f"${pnl:+,.2f}",
                "損益率": f"{pnl_pct:+.2f}%",
                "Beta": f"{beta:.2f}",
                "ボラティリティ(年率)": f"{vol*100:.1f}%",
                "VaR(95%)": f"{var95*100:.1f}%",
                "メモ": entry.get("notes", "")
            })

    total_pnl = total_current - total_invested
    total_pnl_pct = ((total_current / total_invested) - 1.0) * 100 if total_invested > 0 else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("💰 総投資額", f"${total_invested:,.2f}")
    m2.metric("📈 総評価額", f"${total_current:,.2f}")
    m3.metric("💵 総損益", f"${total_pnl:+,.2f}", delta=f"{total_pnl_pct:+.2f}%")
    m4.metric("📦 銘柄数", f"{len(portfolio)} 件")

    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # --- 削除機能 ---
    with st.expander("🗑️ 購入記録を削除する"):
        delete_options = [f"{e['ticker']} ({e['purchase_date']} {e['purchase_time']}) - ${e['purchase_price']:,.4f} × {e['quantity']}" for e in portfolio]
        selected_delete = st.selectbox("削除する記録を選択", delete_options)
        if st.button("🗑️ この記録を削除", type="secondary"):
            idx = delete_options.index(selected_delete)
            portfolio.pop(idx)
            save_portfolio(portfolio)
            st.success("削除しました。")
            st.rerun()

    # --- AI売却アドバイスセクション ---
    st.markdown("---")
    st.header("🤖 AI売却タイミングアドバイス")
    st.markdown("各保有銘柄について、AIが「次に上がるか・下がるか」を分析し、購入価格を基準にした **利確・損切りライン** と合わせて売却すべきかアドバイスします。")

    if st.button("🚀 AIによる売却アドバイスを取得", type="primary", key="pf_ai_btn"):
        for entry in portfolio:
            t_pf = entry["ticker"]
            is_cry_pf = entry.get("is_crypto", False)
            buy_price = entry["purchase_price"]
            qty = entry["quantity"]

            st.markdown(f"### {t_pf}  ─  購入: {entry['purchase_date']} {entry['purchase_time']}")

            try:
                with st.spinner(f"{t_pf} を分析中..."):
                    df_ai = fetch_data(t_pf, start_date, end_date, '1d', is_cry_pf)
                    if df_ai.empty or len(df_ai) < 50:
                        st.warning(f"{t_pf}: データが不足しているためスキップします。")
                        continue

                    df_ai, feats = add_time_series_features(df_ai, True, True, True, True)

                    if not is_cry_pf:
                        vix_df_ai = fetch_vix(start_date, end_date)
                        if not vix_df_ai.empty:
                            df_ai = df_ai.join(vix_df_ai, how='left')
                            df_ai['VIX_Close'] = df_ai['VIX_Close'].ffill()
                            feats.append('VIX_Close')

                    df_ai['ATR'] = AverageTrueRange(high=df_ai["High"], low=df_ai["Low"], close=df_ai["Close"], window=14).average_true_range()

                    ml_df_ai = df_ai.dropna().copy()
                    if len(ml_df_ai) < 50:
                        st.warning(f"{t_pf}: データ不足のためスキップ。")
                        continue

                    X_ai = ml_df_ai[feats]
                    y_ai = ml_df_ai['Target']
                    model_ai = get_xgb_model(X_ai, y_ai, auto_tune)

                    latest_ai = df_ai.iloc[-1:]
                    latest_feats_ai = latest_ai[feats]
                    if latest_feats_ai.isna().any().any():
                        latest_feats_ai = ml_df_ai.iloc[-1:][feats]

                    prob_ai = model_ai.predict_proba(latest_feats_ai)[0]
                    prob_up_ai = prob_ai[1]

                    current_price_ai = float(df_ai.iloc[-1]['Close'])
                    current_atr_ai = float(df_ai.iloc[-1]['ATR'])
                    pnl_val = (current_price_ai - buy_price) * qty
                    pnl_pct_ai = ((current_price_ai / buy_price) - 1.0) * 100

                    # 購入価格基準のATR利確・損切りライン
                    sl_line = buy_price - (1.5 * current_atr_ai)
                    tp_line = buy_price + (2.5 * current_atr_ai)

                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric("現在の損益", f"{pnl_pct_ai:+.2f}%", delta=f"${pnl_val:+,.2f}")
                    with c2:
                        st.metric("AI 上昇確率 (次の日足)", f"{prob_up_ai*100:.1f}%")
                    with c3:
                        if current_price_ai <= sl_line:
                            st.error("🚨 **即座に損切りを推奨**")
                        elif current_price_ai >= tp_line:
                            st.success("🎯 **利確（売却）を推奨**")
                        elif prob_up_ai < 0.4:
                            st.warning("📉 **売却を検討**")
                        elif prob_up_ai > 0.6:
                            st.info("📈 **保有継続を推奨**")
                        else:
                            st.info("➖ **様子見（明確なシグナルなし）**")

                    st.markdown(f"""
                    | 項目 | 値 |
                    |---|---|
                    | 購入単価 | **${buy_price:,.4f}** |
                    | 現在値 | **${current_price_ai:,.4f}** |
                    | 🎯 利確目標 (購入価格 + ATR×2.5) | `${tp_line:,.2f}` |
                    | 🛡️ 損切りライン (購入価格 - ATR×1.5) | `${sl_line:,.2f}` |
                    | ATR (1日の平均値幅) | ${current_atr_ai:,.2f} |
                    """)

                    # ミニチャート：購入日以降の価格推移＋購入ラインを表示
                    purchase_date_dt = datetime.date.fromisoformat(entry['purchase_date'])
                    chart_df = df_ai[df_ai.index >= pd.Timestamp(purchase_date_dt)]
                    if not chart_df.empty:
                        mini_fig = go.Figure()
                        mini_fig.add_trace(go.Candlestick(
                            x=chart_df.index, open=chart_df['Open'], high=chart_df['High'],
                            low=chart_df['Low'], close=chart_df['Close'], name='価格',
                            increasing_line_color='green', decreasing_line_color='red'
                        ))
                        mini_fig.add_hline(y=buy_price, line_dash="dash", line_color="blue",
                                           annotation_text=f"購入価格 ${buy_price:,.2f}", annotation_position="top left")
                        mini_fig.add_hline(y=tp_line, line_dash="dot", line_color="green",
                                           annotation_text=f"利確 ${tp_line:,.2f}", annotation_position="top left")
                        mini_fig.add_hline(y=sl_line, line_dash="dot", line_color="red",
                                           annotation_text=f"損切り ${sl_line:,.2f}", annotation_position="bottom left")
                        mini_fig.update_layout(
                            title=f"{t_pf} 購入後の価格推移", height=400,
                            yaxis_title="Price", template="plotly_white",
                            xaxis_rangeslider_visible=False
                        )
                        st.plotly_chart(mini_fig, use_container_width=True)

            except Exception as e:
                st.error(f"{t_pf}: 分析中にエラーが発生しました。({str(e)[:100]})")

        st.markdown("---")
        st.caption("⚠️ AIの予測は参考情報であり、投資の最終判断はご自身で行ってください。")

    st.stop()

if app_mode == "🏆 AI 一斉スクリーナー (買い/売り)":
    st.header(f"🏆 AI 一斉スクリーナー ({timeframe})")
    
    col_type, col_filter = st.columns(2)
    with col_type:
        scan_type = st.radio("ランキングの種類", ["🔼 買いシグナル (上昇確率順)", "🔽 売りシグナル (下落確率順)"])
    with col_filter:
        only_bookmarks = st.checkbox("⭐ マイ・ブックマークの銘柄のみスキャン", value=False)
        
    st.markdown("AIが一斉に計算し、「次に上がる(下がる)確率が高い」と判断した順にランキング表示します。")
    if run_button:
        with st.spinner("🚀 全銘柄を一斉スキャン中..."):
            all_tickers = []
            
            if only_bookmarks:
                for name, tic in WATCHLIST["⭐ マイ・ブックマーク"].items():
                    # 仮想通貨の判定用に無理やりカテゴリ名をつける
                    cat = "🪙 仮想通貨" if "USDT" in tic or "USD" in tic else "ブックマーク"
                    all_tickers.append({"name": name, "ticker": tic, "category": cat})
            else:
                for cat, items in WATCHLIST.items():
                    if cat == "✍️ 自分で入力 (カスタム)": continue
                    for name, tic in items.items():
                        all_tickers.append({"name": name, "ticker": tic, "category": cat})
            
            if not all_tickers:
                st.warning("スキャン対象の銘柄がありません。ブックマークを確認してください。")
                st.stop()
            
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, item in enumerate(all_tickers):
                tic = item["ticker"]
                is_cry = ("仮想通貨" in item["category"])
                status_text.text(f"スキャン中 ({i+1}/{len(all_tickers)}): {item['name']}")
                
                try:
                    df = fetch_data(tic, start_date, end_date, timeframe, is_cry)
                    if df.empty or len(df) < 50:
                        continue
                        
                    df, features = add_time_series_features(df, use_sma, use_rsi, use_macd, use_bb)
                    
                    if not is_cry and timeframe == '1d':
                        vix_df = fetch_vix(start_date, end_date)
                        if not vix_df.empty:
                            df = df.join(vix_df, how='left')
                            df['VIX_Close'] = df['VIX_Close'].ffill()
                            features.append('VIX_Close')
                        
                    ml_df = df.dropna().copy()
                    if len(ml_df) < 30:
                        continue
                        
                    X = ml_df[features]
                    y = ml_df['Target']
                    model = get_xgb_model(X, y, auto_tune)
                    
                    latest_features = df.iloc[-1:][features]
                    if latest_features.isna().any().any():
                        latest_features = ml_df.iloc[-1:][features]
                        
                    prob_up = model.predict_proba(latest_features)[0][1]
                    prob_down = 1.0 - prob_up
                    live_p, _ = fetch_live_price(tic, is_cry)
                    curr_price = live_p if live_p else df.iloc[-1]['Close']
                    
                    results.append({
                        "銘柄名": item["name"],
                        "ティッカー": tic,
                        "カテゴリ": item["category"].split(" ", 1)[-1] if " " in item["category"] else item["category"],
                        "上昇確率 (%)": round(prob_up * 100, 1),
                        "下落確率 (%)": round(prob_down * 100, 1),
                        "現在価格": f"${curr_price:,.2f}" if "日本株" not in item["category"] else f"¥{curr_price:,.1f}"
                    })
                except Exception as e:
                    pass
                    
                progress_bar.progress((i + 1) / len(all_tickers))
            
            status_text.empty()
            
            if results:
                res_df = pd.DataFrame(results)
                
                # スキャンタイプに応じてソートとカラム調整を行う
                if "買い" in scan_type:
                    res_df = res_df.sort_values(by="上昇確率 (%)", ascending=False).reset_index(drop=True)
                    res_df = res_df.drop(columns=["下落確率 (%)"])
                    st.success("✨ スキャン完了！ **「上昇する確率が高い順」** に表示しています。")
                else:
                    res_df = res_df.sort_values(by="下落確率 (%)", ascending=False).reset_index(drop=True)
                    res_df = res_df.drop(columns=["上昇確率 (%)"])
                    st.success("📉 スキャン完了！ **「下落する（空売りすべき）確率が高い順」** に表示しています。")
                    
                st.dataframe(res_df, use_container_width=True)
                
                # スキャン結果のCSVダウンロード機能
                csv_results = res_df.to_csv(index=False).encode('utf-8-sig')
                st.download_button("📥 スキャン結果をCSV形式で保存", data=csv_results, file_name="ai_screener_results.csv", mime="text/csv")
            else:
                st.warning("スキャン結果が得られませんでした。")
        st.stop()

if run_button:
    # リアルタイム価格バッジ
    live_price, live_change = fetch_live_price(ticker, is_crypto)
    if live_price is not None:
        st.metric(label=f"🔴 {ticker} リアルタイム現在値", value=f"${live_price:,.4f}", delta=f"{live_change:+.2f}% (24h/前日比)")
        st.markdown("---")

    with st.spinner("データを取得・計算中..."):
        df = fetch_data(ticker, start_date, end_date, timeframe, is_crypto)
        
    if df.empty:
        st.error("データの取得に失敗しました。開始日が古すぎる（5分足の60日制限など）か、銘柄名を確認してください。")
    else:
        st.success(f"取得完了: {len(df)} 期間分のデータ ({timeframe})")
        
        # --- 特徴量エンジニアリング (時系列ラグ ＋ ta指標) ---
        df, features = add_time_series_features(df, use_sma, use_rsi, use_macd, use_bb)
        
        # VIX特化処理 (日足・非暗号資産のみ)
        if not is_crypto and timeframe == '1d':
            vix_df = fetch_vix(start_date, end_date)
            if not vix_df.empty:
                df = df.join(vix_df, how='left')
                df['VIX_Close'] = df['VIX_Close'].ffill()
                features.append('VIX_Close')
        
        # ボラティリティ評価用(利確・損切りライン計算用)のリスク指標は個別分析専用
        df['ATR'] = AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"], window=14).average_true_range()
        
        # NaNを含む行を除外（指標計算の初期期間＋最後の行）
        ml_df = df.dropna().copy()
        
        latest_closing_price = df.iloc[-1]['Close']
        latest_atr = df.iloc[-1]['ATR']
        
        if len(ml_df) < 100:
            st.warning("データが少なすぎるため、機械学習の精度が出ません。開始日を昔に設定してください。")
        else:
            X = ml_df[features]
            y = ml_df['Target']
            
            # 時系列分割
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            
            
            # XGBoostモデルの設定と学習 (Auto-Tune付き)
            model = get_xgb_model(X_train, y_train, auto_tune)
            
            # テストデータでの評価
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            
            # --- テスト期間のAIのシグナルを抽出 ---
            test_indices = y_test.index
            ml_df.loc[test_indices, 'AI_Signal'] = predictions

            if app_mode == "📊 個別銘柄の詳細分析":
                st.header(f"🤖 AI予測 ＆ リスク管理（利確・損切）の目安")
                
                # --- 🏢 企業基本情報 ＆ 主要財務指標 (New Feature) ---
                with st.expander("🏢 企業基本情報 ＆ 主要財務指標", expanded=True):
                    try:
                        ticker_info_obj = yf.Ticker(ticker)
                        info = ticker_info_obj.info
                        
                        f_col1, f_col2, f_col3 = st.columns(3)
                        with f_col1:
                            st.write("**基本データ**")
                            st.write(f"- セクター: {info.get('sectorDisp', info.get('sector', 'N/A'))}")
                            st.write(f"- 業界: {info.get('industryDisp', info.get('industry', 'N/A'))}")
                            m_cap = info.get('marketCap')
                            if m_cap:
                                st.write(f"- 時価総額: ${m_cap:,.0f}" if not is_crypto else f"- 時価総額: ${m_cap:,.0f}")
                            else:
                                st.write("- 時価総額: N/A")
                                
                        with f_col2:
                            st.write("**投資指標**")
                            pe = info.get('trailingPE')
                            st.write(f"- PER (実績): {f'{pe:.2f}' if pe else 'N/A'}")
                            pb = info.get('priceToBook')
                            st.write(f"- PBR: {f'{pb:.2f}' if pb else 'N/A'}")
                            div = info.get('dividendYield')
                            st.write(f"- 配当利回り: {f'{div*100:.2f}%' if div else 'N/A'}")
                            
                        with f_col3:
                            st.write("**財務健全性**")
                            roe = info.get('returnOnEquity')
                            st.write(f"- ROE: {f'{roe*100:.2f}%' if roe else 'N/A'}")
                            de = info.get('debtToEquity')
                            st.write(f"- 自己資本比率(D/E): {f'{de:.2f}' if de else 'N/A'}")
                            beta = info.get('beta')
                            st.write(f"- ベータ値 (感応度): {f'{beta:.2f}' if beta else 'N/A'}")
                    except Exception as e:
                        st.write("銘柄情報の取得に失敗しました。詳細データがない可能性があります。")

                
                # --- 💡 インサイト ---
                insight_col1, insight_col2 = st.columns(2)
                with insight_col1:
                    st.subheader("📅 企業イベント (予定)")
                    events = get_corporate_events(ticker)
                    if events:
                        st.write(pd.DataFrame([{"項目": k, "値": str(v)} for k, v in events.items()]))
                    else:
                        st.write("イベント情報がありません（仮想通貨/ETF等）。")
                
                with insight_col2:
                    st.subheader("🔗 ウォッチリスト相関係数 (90日)")
                    all_watchlist = list(WATCHLIST[selected_category].values())
                    if len(all_watchlist) > 1:
                        corr_df = calculate_correlations(all_watchlist)
                        if corr_df is not None:
                            fig_corr = go.Figure(data=go.Heatmap(
                                z=corr_df.values, x=corr_df.columns, y=corr_df.columns,
                                colorscale='Viridis', zmin=-1, zmax=1
                            ))
                            fig_corr.update_layout(height=400, margin=dict(l=20, r=20, t=20, b=20))
                            st.plotly_chart(fig_corr, use_container_width=True)
                        else: st.write("相関データの取得に失敗しました。")
                    else: st.write("相関を計算するには複数の銘柄が必要です。")
                
                # 本日のデータで次の期間の予測を行う
                latest_features = df.iloc[-1:][features]
                if latest_features.isna().any().any():
                    latest_features = ml_df.iloc[-1:][features]
                    
                tomorrow_prob = model.predict_proba(latest_features)[0]
                prob_down, prob_up = tomorrow_prob[0], tomorrow_prob[1]
                
                # --- 🧠 AIの判断根拠 (SHAP分析) ---
                with st.expander("🧠 AIはどうしてそう判断したの？ (説明可能性分析)", expanded=False):
                    st.write("SHAPと呼ばれる技術を使い、AIがどの指標を重視して予測を行ったかを可視化します。")
                    try:
                        explainer = shap.Explainer(model)
                        shap_values = explainer(latest_features)
                        
                        fig_shap, ax_shap = plt.subplots(figsize=(10, 5))
                        shap.bar_plot(shap_values[0], feature_names=features, show=False)
                        st.pyplot(fig_shap)
                        st.caption("※ 右に伸びている指標は価格を押し上げる要因、左に伸びている指標は押し下げる要因として働いています。")
                    except Exception as e:
                        st.warning(f"SHAP分析の生成に失敗しました: {e}")

                # --- 📉 マクロ指標感応度分析 (NEW) ---
                with st.expander("📉 マクロ指標感応度分析 (市場・原油・金・VIXとの連動性)", expanded=False):
                    st.write("対象銘柄が主要な外部要因にどの程度影響を受けやすいかを、過去180日のデータから分析します。")
                    with st.spinner("マクロ感応度を計算中..."):
                        macro_res = calculate_macro_sensitivity(ticker)
                        if macro_res:
                            m_df = pd.DataFrame(list(macro_res.items()), columns=['指標', '相関係数'])
                            fig_macro = px.bar(m_df, x='相関係数', y='指標', orientation='h',
                                              color='相関係数', color_continuous_scale='RdBu_r', range_color=[-1, 1],
                                              title=f"{ticker} のマクロ要因・相関プロファイル")
                            fig_macro.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
                            st.plotly_chart(fig_macro, use_container_width=True)
                            st.caption("※ 1に近いほど正の連動、-1に近いほど逆の動きをします。0付近は無相関です。")
                        else:
                            st.info("マクロデータの取得または計算に失敗しました。")

                # --- ニュース・感情分析 (Opt-out可能) ---
                news_adjustment = 0.0
                sentiment_summary = []
                if use_news:
                    st.markdown("---")
                    st.header("📰 最新ニュース・感情分析")
                    try:
                        ticker_obj = yf.Ticker(ticker)
                        news_data = ticker_obj.news
                        if news_data and len(news_data) > 0:
                            # FinBERTを使用した分析
                            use_finbert = (get_setting("use_finbert", "1") == "1")
                            titles = [a.get('title', '') for a in news_data[:5] if a.get('title', '')]
                            adv_score, adv_mood = analyze_sentiment(titles, use_finbert=use_finbert)
                            
                            st.metric("AI判定によるセンチメント", f"{adv_mood}", delta=f"{adv_score:+.2f}")
                            
                            with st.expander("詳細なニュース内容"):
                                for article in news_data[:5]:
                                    title = article.get('title', '')
                                    if title:
                                        st.write(f"- {title}")
                            
                            # スコアに応じて確率を調整 (例: 最大+-10%の補正)
                            news_adjustment = adv_score * 0.10
                            prob_up = max(0.0, min(1.0, prob_up + news_adjustment))
                            prob_down = 1.0 - prob_up
                            
                            if news_adjustment > 0:
                                st.info(f"💡 ニュースがポジティブなため、AIの「上昇確率」を +{news_adjustment*100:.1f} % 上方修正しました。")
                            elif news_adjustment < 0:
                                st.warning(f"⚠️ ニュースがネガティブなため、AIの「上昇確率」を {news_adjustment*100:.1f} % 下方修正しました。")
                        else:
                            st.write("最新のニュースが見つかりませんでした。")
                    except Exception as e:
                        st.error(f"ニュース取得中にエラーが発生しました: {e}")
                
                st.markdown("---")
                tomorrow_pred = 1 if prob_up > 0.5 else 0
                
                pred_col1, pred_col2 = st.columns(2)
                
                with pred_col1:
                    st.subheader(f"次の期間({timeframe})の予測")
                    if tomorrow_pred == 1:
                        st.success(f"🔼 **上昇する確率が高い**（確率: {prob_up*100:.1f} %）\n\n**現在価格**: ${latest_closing_price:,.4f}")
                    else:
                        st.error(f"🔽 **下落する確率が高い**（確率: {prob_down*100:.1f} %）\n\nリスクが高いため購入を見送るか、空売りを検討してください。")
                        
                with pred_col2:
                    st.subheader("💡 買う場合のリスク管理（損切りと利確）")
                    if pd.isna(latest_atr):
                        st.write("ボラティリティデータが不足しています。")
                    else:
                        stop_loss_price = latest_closing_price - (1.5 * latest_atr)
                        take_profit_price = latest_closing_price + (2.5 * latest_atr)
                        
                        st.markdown(f"""
                        現在のボラティリティ（1日の値動き: 約${latest_atr:.2f}）を考慮した、安全な取引の目安です：
                        
                        - 🎯 **利確目標 (Take Profit)**: **`${take_profit_price:.2f}`** （購入価格よりかなり上がったら欲張らず一部売却）
                        - 🛡️ **損切りライン (Stop Loss)**: **`${stop_loss_price:.2f}`** （予想に反してこの価格まで下がったら、**必ず損切りして逃げる**）
                        """)

                # --- Plotlyによるローソク足 ＋ 過去のシグナル表示 ---
                st.markdown("---")
                st.header("📉 テクニカルチャート & 売買シグナル")
                st.markdown("ローソク足に加え、RSIやMACDなどの指標をマルチペインで確認できます。")
                
                col_chart_opts1, col_chart_opts2 = st.columns(2)
                with col_chart_opts1:
                    show_rsi = st.checkbox("RSI (買われすぎ/売られすぎ) を表示", value=True)
                with col_chart_opts2:
                    show_macd = st.checkbox("MACD (トレンド転換) を表示", value=True)
                
                rows = 1
                row_heights = [0.6]
                if show_rsi:
                    rows += 1
                    row_heights.append(0.2)
                if show_macd:
                    rows += 1
                    row_heights.append(0.2)
                
                total = sum(row_heights)
                row_heights = [r/total for r in row_heights]
                
                fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                                    vertical_spacing=0.03, row_heights=row_heights)

                # ローソク足
                fig.add_trace(go.Candlestick(x=df.index,
                                open=df['Open'], high=df['High'],
                                low=df['Low'], close=df['Close'],
                                name='Candlestick',
                                increasing_line_color='green', decreasing_line_color='red'), row=1, col=1)

                if use_sma:
                    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], line={'color':'orange', 'width':1.5}, name='SMA 20'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], line={'color':'purple', 'width':1.5}, name='SMA 50'), row=1, col=1)

                if use_bb:
                    fig.add_trace(go.Scatter(x=df.index, y=df['Close'] * (1 + df['BB_Width']/2), line={'color':'lightblue', 'width':1, 'dash':'dash'}, name='BB Upper', showlegend=False), row=1, col=1)
                    fig.add_trace(go.Scatter(x=df.index, y=df['Close'] * (1 - df['BB_Width']/2), line={'color':'lightblue', 'width':1, 'dash':'dash'}, name='BB Lower', fill='tonexty', fillcolor='rgba(173,216,230,0.1)', showlegend=False), row=1, col=1)

                buy_signals = ml_df[ml_df['AI_Signal'] == 1]
                if not buy_signals.empty:
                    fig.add_trace(go.Scatter(
                        x=buy_signals.index,
                        y=buy_signals['Low'] * 0.98,
                        mode='markers',
                        marker={'symbol':'triangle-up', 'size':12, 'color':'blue', 'line':{'width':1, 'color':'darkblue'}},
                        name='AI 買いシグナル(🔼)'
                    ), row=1, col=1)
                    
                curr_row = 2
                if show_rsi:
                    fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line={'color':'#38bdf8', 'width':1.5}, name='RSI'), row=curr_row, col=1)
                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=curr_row, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=curr_row, col=1)
                    fig.update_yaxes(title_text="RSI", row=curr_row, col=1)
                    curr_row += 1
                    
                if show_macd:
                    fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], line={'color':'blue', 'width':1.5}, name='MACD'), row=curr_row, col=1)
                    fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], line={'color':'orange', 'width':1.5}, name='Signal'), row=curr_row, col=1)
                    macd_hist = df['MACD'] - df['MACD_Signal']
                    fig.add_trace(go.Bar(x=df.index, y=macd_hist, name='Histogram', marker_color=np.where(macd_hist<0, 'red', 'green')), row=curr_row, col=1)
                    fig.update_yaxes(title_text="MACD", row=curr_row, col=1)

                fig.update_layout(title=f'{ticker} テクニカルチャート',
                                  height=800,
                                  template="plotly_white",
                                  xaxis_rangeslider_visible=False,
                                  showlegend=True,
                                  legend={'yanchor':"top", 'y':0.99, 'xanchor':"left", 'x':0.01})
                st.plotly_chart(fig, use_container_width=True)
                
                if st.button("📢 今日の予測結果をSNS/チャットに通知飛ばす", type="secondary"):
                    msg = f"📈 【AI予測通知】 {ticker}\n判定: {'上昇(1)' if tomorrow_pred == 1 else '下落(0)'}\n確率: {prob_up*100:.1f}%\n現在値: ${latest_closing_price:,.2f}\n精度: {accuracy*100:.1f}%"
                    send_notification(msg)
                    st.toast("通知を送信しました！")

            elif app_mode == "🧪 バックテスト・検証機能":
                # --- バーチャル・ポートフォリオ（バックテスト） ---
                st.markdown("---")
                st.header(f"💰 【{ticker}】バックテスト（仮想の資産推移シミュレーション）")
                st.markdown(f"過去のテスト期間（{y_test.index[0].date()} 〜 {y_test.index[-1].date()}）におけるAIシグナルとパフォーマンスを検証します。")
                
                initial_capital = st.number_input("初期投資額 (USD)", value=10000)
                enable_short = st.checkbox("🔄 「空売り（ショート）」戦略をシミュレーションに含める", value=False, help="AIが下落すると予測した日（AIシグナル=0）に空売りを仕掛け、相場の下落も利益に変えるヘッジ戦略を検証します。")
                
                test_df = ml_df.loc[test_indices].copy()
                
                # 翌日のリターン (Next Close / Close - 1)
                test_df['Tomorrow_Return'] = test_df['Close'].shift(-1) / test_df['Close'] - 1
                # 最後の行は翌日がないため0埋め
                test_df['Tomorrow_Return'] = test_df['Tomorrow_Return'].fillna(0)
                
                # 保有戦略（Buy & Hold）の単利/複利
                test_df['Buy_Hold_Eq'] = (1.0 + test_df['Tomorrow_Return']).cumprod() * initial_capital
                
                # AI戦略
                if enable_short:
                    test_df['Strategy_Return'] = np.where(test_df['AI_Signal'] == 1, test_df['Tomorrow_Return'], -test_df['Tomorrow_Return'])
                else:
                    test_df['Strategy_Return'] = np.where(test_df['AI_Signal'] == 1, test_df['Tomorrow_Return'], 0)
                    
                test_df['Strategy_Eq'] = (1.0 + test_df['Strategy_Return']).cumprod() * initial_capital
                
                # グラフ描画
                eq_fig = go.Figure()
                eq_fig.add_trace(go.Scatter(x=test_df.index, y=test_df['Buy_Hold_Eq'], mode='lines', name='何もしない（一括投資・Buy&Hold）', line={'color':'gray', 'dash':'dot'}))
                
                strat_name = 'AIシグナル戦略 (ロング＆ショート両建て)' if enable_short else 'AIシグナル戦略 (上がる日だけ保有)'
                eq_fig.add_trace(go.Scatter(x=test_df.index, y=test_df['Strategy_Eq'], mode='lines', name=strat_name, line={'color':'#38bdf8', 'width':2.5}))
                
                final_buy_hold = test_df['Buy_Hold_Eq'].iloc[-1]
                final_ai = test_df['Strategy_Eq'].iloc[-1]
                
                eq_fig.update_layout(title='資産の推移 (Equity Curve)', yaxis_title='Account Balance', height=500, template="plotly_white")
                st.plotly_chart(eq_fig, use_container_width=True)
                
                # 統計指標の計算
                strat_returns = test_df['Strategy_Return']
                if not strat_returns.empty:
                    sharpe = (strat_returns.mean() / strat_returns.std()) * np.sqrt(252) if strat_returns.std() != 0 else 0
                    rolling_max = test_df['Strategy_Eq'].cummax()
                    drawdown = (test_df['Strategy_Eq'] - rolling_max) / rolling_max
                    max_dd = drawdown.min()
                    # ゼロリターンを除外して勝率を出す
                    active_returns = strat_returns[strat_returns != 0]
                    win_rate = (active_returns > 0).mean() * 100 if len(active_returns) > 0 else 0
                else:
                    sharpe, max_dd, win_rate = 0, 0, 0

                st.write(f"📈 **AI戦略の最終資産**: **${final_ai:,.2f}** (単純保有: ${final_buy_hold:,.2f})")
                
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("🎯 モデル正答率", f"{accuracy * 100:.1f} %")
                c2.metric("📊 シャープレシオ", f"{sharpe:.2f}")
                c3.metric("📉 最大ドローダウン", f"{max_dd * 100:.1f} %")
                c4.metric("🏆 トレード勝率", f"{win_rate:.1f} %")
                
                # 追加：トレード回数プロファイル
                total_days = len(test_df)
                long_days = len(test_df[test_df['AI_Signal'] == 1])
                short_days = len(test_df[test_df['AI_Signal'] == 0])
                if enable_short:
                    st.info(f"💡 **運用プロファイル**: 全{total_days}営業日のうち、ロング(買い)ポジションを持った日数が **{long_days}** 日、ショート(空売り)が **{short_days}** 日です。相場下落時にも強力に利益を狙う攻撃的な設計です。")
                else:
                    st.info(f"💡 **運用プロファイル**: 全{total_days}営業日のうち、AIが上昇を予想してリスクを取り「買い」を持った日数は **{long_days}** 日 ({long_days/total_days*100:.1f}%) です。下落リスクを回避することでドローダウンを抑える守備的な設計です。")
