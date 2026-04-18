import sys
import re

with open("streamlit_app.py", "r") as f:
    text = f.read()

# 1. Define train_and_cache_model
wrapper = """
@st.cache_resource(show_spinner=False)
def train_and_cache_model(ticker_id, timeframe, start, end, auto_tune, _X, _y):
    return get_xgb_model(_X, _y, auto_tune)
"""
text = text.replace("# ===========================================================\n# ポートフォリオ管理モード", wrapper + "\n# ===========================================================\n# ポートフォリオ管理モード", 1)

# 2. Use it instead of get_xgb_model
text = text.replace("model_ai = get_xgb_model(X_ai, y_ai, auto_tune)", "model_ai = train_and_cache_model(t_pf, timeframe, start_date, end_date, auto_tune, X_ai, y_ai)")
text = text.replace("model = get_xgb_model(X, y, auto_tune)", "model = train_and_cache_model(tic, timeframe, start_date, end_date, auto_tune, X, y)")
text = text.replace("model = get_xgb_model(X_train, y_train, auto_tune)", "model = train_and_cache_model(ticker, timeframe, start_date, end_date, auto_tune, X_train, y_train)")

# 3. Handle Portfolio functions
text = re.sub(r'PORTFOLIO_FILE = "portfolio.json"\n\ndef load_portfolio\(\):\n(?: {4}.*\n)*? {4}return \[\]\n', 'def load_portfolio():\n    return get_portfolio()\n', text)
text = re.sub(r'def save_portfolio\(portfolio_data\):\n(?: {4}.*\n)*', '', text)

# 4. Handle append and save
text = text.replace('portfolio.append(new_entry)\n                save_portfolio(portfolio)\n                st.success(f"✅ 記録完了', 'add_portfolio_item(new_entry)\n                st.toast(f"✅ 記録完了')
text = text.replace('portfolio.pop(idx)\n            save_portfolio(portfolio)\n            st.success("削除しました。")', 'item_id = delete_options[idx].split(" ")[0] # This fails because item_id is needed, let me fix it manually.')

with open("streamlit_app.py", "w") as f:
    f.write(text)
