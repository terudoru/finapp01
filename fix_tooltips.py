import sys

with open("streamlit_app.py", "r") as f:
    text = f.read()

text = text.replace('use_sma = st.sidebar.checkbox("移動平均線 (SMA 20, SMA 50)", value=True)', 'use_sma = st.sidebar.checkbox("移動平均線 (SMA 20, SMA 50)", value=True, help="過去20日・50日の平均価格に基づくトレンド指標です。")')
text = text.replace('use_rsi = st.sidebar.checkbox("RSI (買われすぎ/売られすぎ指標)", value=True)', 'use_rsi = st.sidebar.checkbox("RSI (買われすぎ/売られすぎ指標)", value=True, help="相場の過熱感を示すオシレーター指標です(70以上で買われすぎ、30以下で売られすぎ)。")')
text = text.replace('use_macd = st.sidebar.checkbox("MACD (トレンド転換指標)", value=True)', 'use_macd = st.sidebar.checkbox("MACD (トレンド転換指標)", value=True, help="短期と長期の移動平均の差からトレンドの転換点を見極めます。")')
text = text.replace('use_bb = st.sidebar.checkbox("ボリンジャーバンド幅", value=True)', 'use_bb = st.sidebar.checkbox("ボリンジャーバンド幅", value=True, help="価格のボラティリティ（変動率）をバンド幅で表す指標です。")')

with open("streamlit_app.py", "w") as f:
    f.write(text)
print("Finished fixing tooltips.")
