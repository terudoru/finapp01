import sys
import re

with open("streamlit_app.py", "r") as f:
    text = f.read()

# Locate the boundaries
start_marker = '            if app_mode == "📊 個別銘柄の詳細分析":'
end_marker = '            elif app_mode == "🧪 バックテスト・検証機能":'

if start_marker not in text or end_marker not in text:
    print("Markers not found!")
    sys.exit(1)

start_idx = text.find(start_marker)
end_idx = text.find(end_marker)

# Code to be inserted (the 4 tabs layout with all 5 features)
new_code = """            if app_mode == "📊 個別銘柄の詳細分析":
                st.header(f"📊 {ticker} のプロフェッショナル分析")
                
                # --- NEW: UI organization by Tabs ---
                tab1, tab2, tab3, tab4 = st.tabs(["📈 チャート&シグナル", "🏛️ ファンダメンタルズ", "📊 リスク・シミュレーション", "🤖 AI予測(XGBoost)"])
                
                with tab1:
                    st.subheader("📉 テクニカルチャート & 売買シグナル")
                    st.markdown("ローソク足に加え、RSIやMACD、**自動検知されたチャートパターン**、さらに**マクロ指標(米10年債利回り)**をオーバーレイ表示します。")
                    
                    col_chart_opts1, col_chart_opts2, col_chart_opts3 = st.columns(3)
                    with col_chart_opts1:
                        show_rsi = st.checkbox("RSI (買われすぎ/売られすぎ) ", value=True)
                    with col_chart_opts2:
                        show_macd = st.checkbox("MACD (トレンド転換) ", value=True)
                    with col_chart_opts3:
                        show_macro = st.checkbox("マクロ指標 (米10年国債利回り)", value=False, help="米国債利回りと株価の逆相関などの関係を確認できます。")
                    
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
                    
                    # Create subplots for extra indicators. If macro overlay is true, create secondary_y for row 1.
                    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                                        vertical_spacing=0.03, row_heights=row_heights,
                                        specs=[[{"secondary_y": True}]] + [[{}]] * (rows-1))

                    # ローソク足
                    fig.add_trace(go.Candlestick(x=df.index,
                                    open=df['Open'], high=df['High'],
                                    low=df['Low'], close=df['Close'],
                                    name='Candlestick',
                                    increasing_line_color='green', decreasing_line_color='red'), row=1, col=1, secondary_y=False)

                    if use_sma:
                        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], line={'color':'orange', 'width':1.5}, name='SMA 20'), row=1, col=1, secondary_y=False)
                        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], line={'color':'purple', 'width':1.5}, name='SMA 50'), row=1, col=1, secondary_y=False)
                        
                        # --- Feature: Automated Chart Patterns (Golden/Dead Cross) ---
                        # SMA20 crossing above SMA50 (Golden)
                        gc_points = df[(df['SMA_20'] > df['SMA_50']) & (df['SMA_20'].shift(1) <= df['SMA_50'].shift(1))]
                        dc_points = df[(df['SMA_20'] < df['SMA_50']) & (df['SMA_20'].shift(1) >= df['SMA_50'].shift(1))]
                        if not gc_points.empty:
                            fig.add_trace(go.Scatter(x=gc_points.index, y=gc_points['SMA_20'] * 0.98, mode='markers', 
                                          marker=dict(symbol='star', size=12, color='gold', line=dict(width=1, color='orange')), name='🌟 G-Cross'), row=1, col=1, secondary_y=False)
                        if not dc_points.empty:
                            fig.add_trace(go.Scatter(x=dc_points.index, y=dc_points['SMA_20'] * 1.02, mode='markers', 
                                          marker=dict(symbol='x', size=10, color='black', line=dict(width=1, color='red')), name='⚔️ D-Cross'), row=1, col=1, secondary_y=False)

                    if use_bb:
                        fig.add_trace(go.Scatter(x=df.index, y=df['Close'] * (1 + df['BB_Width']/2), line={'color':'lightblue', 'width':1, 'dash':'dash'}, name='BB Upper', showlegend=False), row=1, col=1, secondary_y=False)
                        fig.add_trace(go.Scatter(x=df.index, y=df['Close'] * (1 - df['BB_Width']/2), line={'color':'lightblue', 'width':1, 'dash':'dash'}, name='BB Lower', fill='tonexty', fillcolor='rgba(173,216,230,0.1)', showlegend=False), row=1, col=1, secondary_y=False)

                    curr_row = 2
                    if show_rsi:
                        fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], line={'color':'#38bdf8', 'width':1.5}, name='RSI'), row=curr_row, col=1)
                        fig.add_hline(y=70, line_dash="dash", line_color="red", row=curr_row, col=1)
                        fig.add_hline(y=30, line_dash="dash", line_color="green", row=curr_row, col=1)
                        
                        # --- Feature: Chart Patterns (RSI Oversold/Overbought markers) ---
                        os_points = df[(df['RSI'] < 30) & (df['RSI'].shift(1) >= 30)]
                        ob_points = df[(df['RSI'] > 70) & (df['RSI'].shift(1) <= 70)]
                        if not os_points.empty:
                            fig.add_trace(go.Scatter(x=os_points.index, y=os_points['RSI'], mode='markers', marker=dict(symbol='triangle-up', size=8, color='green'), name='RSI Oversold'), row=curr_row, col=1)
                        if not ob_points.empty:
                            fig.add_trace(go.Scatter(x=ob_points.index, y=ob_points['RSI'], mode='markers', marker=dict(symbol='triangle-down', size=8, color='red'), name='RSI Overbought'), row=curr_row, col=1)

                        fig.update_yaxes(title_text="RSI", row=curr_row, col=1)
                        curr_row += 1
                        
                    if show_macd:
                        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], line={'color':'blue', 'width':1.5}, name='MACD'), row=curr_row, col=1)
                        fig.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], line={'color':'orange', 'width':1.5}, name='Signal'), row=curr_row, col=1)
                        macd_hist = df['MACD'] - df['MACD_Signal']
                        fig.add_trace(go.Bar(x=df.index, y=macd_hist, name='Histogram', marker_color=np.where(macd_hist<0, 'red', 'green')), row=curr_row, col=1)
                        fig.update_yaxes(title_text="MACD", row=curr_row, col=1)

                    # --- Feature: Macro Overlay ---
                    if show_macro:
                        with st.spinner("FREDからマクロ指標を取得中..."):
                            try:
                                import pandas_datareader as pdr
                                # DGS10 = 10-Year Treasury Constant Maturity Rate
                                macro_series = pdr.DataReader('DGS10', 'fred', df.index[0], df.index[-1])
                                # Reindex to match trading days, forward fill
                                macro_series = macro_series.reindex(df.index).ffill()
                                fig.add_trace(go.Scatter(x=macro_series.index, y=macro_series['DGS10'], line={'color':'purple', 'width':1, 'dash':'dot'}, name='US 10Yr Yield (R)'), row=1, col=1, secondary_y=True)
                                fig.update_yaxes(title_text="US 10Yr Yield (%)", secondary_y=True, row=1, col=1)
                            except Exception as e:
                                st.error(f"マクロデータの取得に失敗しました: {e}")

                    fig.update_layout(title=f'{ticker} テクニカルチャート',
                                      height=800 if rows > 1 else 500,
                                      template="plotly_white",
                                      xaxis_rangeslider_visible=False,
                                      showlegend=True,
                                      legend={'yanchor':"top", 'y':0.99, 'xanchor':"left", 'x':0.01})
                    st.plotly_chart(fig, use_container_width=True)


                with tab2:
                    st.header("🏢 ファンダメンタルズ ＆ 企業情報")
                    
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
                            
                        st.markdown("---")
                        # --- Feature: Historical Fundamentals Chart ---
                        st.subheader("📊 業績推移 (売上高・純利益)")
                        if not is_crypto:
                            with st.spinner("業績データを取得中..."):
                                inc_stmt = ticker_info_obj.income_stmt
                                if inc_stmt is not None and not inc_stmt.empty:
                                    try:
                                        # YFinance return columns as dates
                                        # Get Revenue and Net Income
                                        rev = inc_stmt.loc['Total Revenue'] if 'Total Revenue' in inc_stmt.index else None
                                        net = inc_stmt.loc['Net Income'] if 'Net Income' in inc_stmt.index else None
                                        
                                        if rev is not None and net is not None:
                                            # Create DataFrame
                                            funds_df = pd.DataFrame({'Total Revenue': rev, 'Net Income': net}).sort_index()
                                            # Format index just to Year
                                            funds_df.index = [str(d.date()) for d in funds_df.index]
                                            
                                            fig_funds = go.Figure()
                                            fig_funds.add_trace(go.Bar(x=funds_df.index, y=funds_df['Total Revenue'], name='Total Revenue', marker_color='lightblue'))
                                            fig_funds.add_trace(go.Bar(x=funds_df.index, y=funds_df['Net Income'], name='Net Income', marker_color='darkblue'))
                                            fig_funds.update_layout(barmode='group', title=f"{ticker} 業績推移 (直近数年)", yaxis_title="Amount ($)", height=400)
                                            st.plotly_chart(fig_funds, use_container_width=True)
                                        else:
                                            st.info("データが一部欠落しています。")
                                    except Exception as e:
                                        st.warning("業績グラフの生成に失敗しました。")
                                else:
                                    st.info("過去の財務データが見つかりませんでした。")
                        else:
                            st.info("仮想通貨のため財務データはありません。")
                            
                    except Exception as e:
                        st.write("銘柄情報の取得に失敗しました。詳細データがない可能性があります。")
                        
                    # 💡 インサイト
                    st.markdown("---")
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


                with tab3:
                    st.header("📊 プロフェッショナル リスク分析 & シミュレーション")
                    
                    # --- Feature: Quantstats Tear Sheet ---
                    st.subheader("📈 ハイレベル成果分析 (Tear Sheet)")
                    st.write("過去の価格変動から、プロレベルのリスク・リターン指標（シャープレシオや最大下落幅など）を算出します。")
                    
                    try:
                        import quantstats as qs
                        # Calculate daily returns
                        daily_returns = df['Close'].pct_change().dropna()
                        daily_returns.index = pd.to_datetime(daily_returns.index)
                        
                        # Use quantstats to calculate core metrics
                        qs_col1, qs_col2, qs_col3, qs_col4 = st.columns(4)
                        sharpe = qs.stats.sharpe(daily_returns)
                        sortino = qs.stats.sortino(daily_returns)
                        max_dd = qs.stats.max_drawdown(daily_returns)
                        win_rate = qs.stats.win_rate(daily_returns)
                        
                        qs_col1.metric("Sharpe Ratio", f"{sharpe:.2f}", help="リスク調整後のリターン。1.0以上なら優秀です。")
                        qs_col2.metric("Sortino Ratio", f"{sortino:.2f}", help="下落リスクに対するリターン（シャープレシオの改良版）。")
                        qs_col3.metric("Max Drawdown", f"{max_dd*100:.2f}%", help="過去最大のピークからの下落幅。")
                        qs_col4.metric("Win Rate", f"{win_rate*100:.1f}%", help="日次リターンがプラスになった確率。")
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Quantstatsの計算に失敗しました: {e}")
                    
                    # --- Feature: Monte Carlo Simulation ---
                    st.markdown("---")
                    st.subheader("🎲 モンテカルロ・シミュレーション (1年後の株価予測)")
                    st.write("過去のボラティリティと平均リターンを元に、1,000パターンのランダムな将来推移を計算します。")
                    
                    with st.spinner("1,000パターンの確率シミュレーションを実行中..."):
                        days_to_simulate = 252 # 1 Trading Year
                        simulations = 1000
                        mu = daily_returns.mean()
                        sigma = daily_returns.std()
                        
                        last_price = df['Close'].iloc[-1]
                        simulation_df = np.zeros((days_to_simulate, simulations))
                        simulation_df[0] = last_price
                        
                        # Generate random paths
                        for x in range(1, days_to_simulate):
                            shock = np.random.normal(loc=mu, scale=sigma, size=simulations)
                            simulation_df[x] = simulation_df[x - 1] * (1 + shock)
                            
                        # Plot visually appealing summary (Avoid plotting 1000 lines, plot percentiles)
                        sim_percentiles = np.percentile(simulation_df, [5, 50, 95], axis=1)
                        
                        fig_mc = go.Figure()
                        future_dates = [df.index[-1] + datetime.timedelta(days=i) for i in range(days_to_simulate)]
                        
                        # 95th Percentile
                        fig_mc.add_trace(go.Scatter(x=future_dates, y=sim_percentiles[2], line=dict(color='green', dash='dot'), name='Top 5% (Best Case)'))
                        # Median
                        fig_mc.add_trace(go.Scatter(x=future_dates, y=sim_percentiles[1], line=dict(color='blue', width=2), name='Median (予想中央値)'))
                        # 5th Percentile
                        fig_mc.add_trace(go.Scatter(x=future_dates, y=sim_percentiles[0], line=dict(color='red', dash='dot'), name='Bottom 5% (Worst Case)'))
                        
                        fig_mc.update_layout(title="Monte Carlo Price Projection - Next 1 Year", yaxis_title="Price ($)", height=500, template='plotly_white')
                        st.plotly_chart(fig_mc, use_container_width=True)
                        
                        mc_col1, mc_col2, mc_col3 = st.columns(3)
                        mc_col1.metric("ワーストケース (下位5%)", f"${sim_percentiles[0][-1]:.2f}")
                        mc_col2.metric("予想中央値 (Median)", f"${sim_percentiles[1][-1]:.2f}")
                        mc_col3.metric("ベストケース (上位5%)", f"${sim_percentiles[2][-1]:.2f}")


                with tab4:
                    st.header(f"🤖 AI予測 ＆ リスク管理（利確・損切）の目安")
                    
                    tomorrow_pred = 1 if prob_up > 0.5 else 0
                    
                    pred_col1, pred_col2 = st.columns(2)
                    
                    with pred_col1:
                        st.subheader(f"次の期間({timeframe})のXGBoost予測")
                        if tomorrow_pred == 1:
                            st.success(f"🔼 **上昇する確率が高い**（確率: {prob_up*100:.1f} %）\\n\\n**現在価格**: ${latest_closing_price:,.4f}")
                        else:
                            st.error(f"🔽 **下落する確率が高い**（確率: {prob_down*100:.1f} %）\\n\\nリスクが高いため購入を見送るか、空売りを検討してください。")
                            
                    with pred_col2:
                        st.subheader("💡 買う場合のリスク管理（損切りと利確）")
                        if pd.isna(latest_atr):
                            st.write("ボラティリティデータが不足しています。")
                        else:
                            stop_loss_price = latest_closing_price - (1.5 * latest_atr)
                            take_profit_price = latest_closing_price + (2.5 * latest_atr)
                            
                            st.markdown(f\"\"\"
                            現在のボラティリティ（1日の値動き: 約${latest_atr:.2f}）を考慮した、安全な取引の目安です：
                            
                            - 🎯 **利確目標 (Take Profit)**: **`${take_profit_price:.2f}`**
                            - 🛡️ **損切りライン (Stop Loss)**: **`${stop_loss_price:.2f}`** （必ず損切りして逃げる）
                            \"\"\")

                    # --- 🧠 AIの判断根拠 (SHAP分析) ---
                    st.markdown("---")
                    st.subheader("🧠 AI予測の根拠 (SHAP分析)")
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

                    # --- ニュース・感情分析 (Opt-out可能) ---
                    if use_news:
                        st.markdown("---")
                        st.header("📰 最新ニュース・感情分析")
                        with st.spinner("AIによるニュース要約と感情スコアを計算中..."):
                            try:
                                ticker_obj = yf.Ticker(ticker)
                                news_data = ticker_obj.news
                                
                                if news_data:
                                    titles = [item.get('title', '') for item in news_data[:5]]
                                    adv_score, adv_mood = analyze_sentiment(titles, use_finbert=use_finbert)
                                    
                                    st.metric("市場感情(AI)", adv_mood, f"スコア: {adv_score:+.2f}")
                                    
                                    for item in news_data[:3]:
                                        title = item.get('title', '')
                                        link = item.get('link', '#')
                                        st.markdown(f"- **[{title}]({link})**")
                                else:
                                    st.write("最近のニュースは見つかりませんでした。")
                            except Exception as e:
                                st.error(f"ニュース取得中にエラーが発生しました: {e}")
                                
                if st.button("📢 今日の予測結果をSNS/チャットに通知飛ばす", type="secondary"):
                    msg = f"📈 【AI予測通知】 {ticker}\\n判定: {'上昇(1)' if tomorrow_pred == 1 else '下落(0)'}\\n確率: {prob_up*100:.1f}%\\n現在値: ${latest_closing_price:,.2f}\\n精度: {accuracy*100:.1f}%"
                    send_notification(msg)
                    st.toast("通知を送信しました！")

"""

new_text = text[:start_idx] + new_code + "\n" + text[end_idx:]

with open("streamlit_app.py", "w") as f:
    f.write(new_text)

print("Insertion complete.")
# Also replace \\n in the inserted string with actual newlines if pandas barfed
