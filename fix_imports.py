import sys

with open("streamlit_app.py", "r") as f:
    text = f.read()

imports = """
import quantstats as qs
import pandas_datareader as pdr
"""

if "quantstats" not in text:
    text = text.replace("import streamlit as st\n", "import streamlit as st\n" + imports)
    with open("streamlit_app.py", "w") as f:
        f.write(text)
    print("Imports added")
else:
    print("Imports exist")
