import sys

with open("streamlit_app.py", "r") as f:
    text = f.read()

old_block = """    with st.spinner("データを取得・計算中..."):
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
            model = train_and_cache_model(ticker, timeframe, start_date, end_date, auto_tune, X_train, y_train)"""

new_block = """    with st.status("🤖 AI分析パイプラインを実行中...", expanded=True) as status:
        st.write("📈 株価データをダウンロード中...")
        df = fetch_data(ticker, start_date, end_date, timeframe, is_crypto)
        
        if df.empty:
            status.update(label="❌ データの取得に失敗しました", state="error", expanded=True)
            st.error("データの取得に失敗しました。開始日が古すぎるか銘柄名を確認してください。")
            st.stop()
            
        st.write(f"✅ 取得完了: {len(df)} 期間分のデータ ({timeframe})")
        
        st.write("⚙️ 特徴量エンジニアリング（テクニカル指標計算）中...")
        df, features = add_time_series_features(df, use_sma, use_rsi, use_macd, use_bb)
        
        if not is_crypto and timeframe == '1d':
            vix_df = fetch_vix(start_date, end_date)
            if not vix_df.empty:
                df = df.join(vix_df, how='left')
                df['VIX_Close'] = df['VIX_Close'].ffill()
                features.append('VIX_Close')
        
        df['ATR'] = AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"], window=14).average_true_range()
        ml_df = df.dropna().copy()
        
        latest_closing_price = df.iloc[-1]['Close']
        latest_atr = df.iloc[-1]['ATR']
        
        if len(ml_df) < 100:
            status.update(label="⚠️ データ不足", state="error", expanded=True)
            st.warning("データが少なすぎるため機械学習の精度が出ません。開始日を昔に設定してください。")
            st.stop()
            
        X = ml_df[features]
        y = ml_df['Target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        st.write("🧠 XGBoostモデルの学習・予測中...")
        model = train_and_cache_model(ticker, timeframe, start_date, end_date, auto_tune, X_train, y_train)
        
        status.update(label="✨ 分析が完了しました！", state="complete", expanded=False)"""

if old_block in text:
    text = text.replace(old_block, new_block)
    with open("streamlit_app.py", "w") as f:
        f.write(text)
    print("Success")
else:
    print("Failed to replace block")
