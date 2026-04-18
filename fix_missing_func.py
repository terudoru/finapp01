lines = open("streamlit_app.py", "r").readlines()
out = []
inserted_pytz = False
inserted_func = False

for line in lines:
    out.append(line)
    if "import pandas as pd" in line and not inserted_pytz:
        out.append("import pytz\n")
        inserted_pytz = True
    
    if "st.set_page_config" in line and not inserted_func:
        out.append("\n@st.cache_resource(ttl=3600, show_spinner=False)\n")
        out.append("def train_and_cache_model(ticker, timeframe, start_date, end_date, auto_tune, _X_train, _y_train):\n")
        out.append("    from scripts.quant_engine import get_xgb_model\n")
        out.append("    return get_xgb_model(_X_train, _y_train, auto_tune)\n\n")
        inserted_func = True

with open("streamlit_app.py", "w") as f:
    f.writelines(out)

print("Fixed!")
