import yfinance as yf
ticker = yf.Ticker("AAPL")

print("--- Institutional Holders ---")
try: print(ticker.institutional_holders)
except: print("N/A")

print("\n--- Insider Transactions ---")
try: print(ticker.insider_transactions.head())
except: print("N/A")

print("\n--- Recommendations ---")
try: print(ticker.recommendations.head())
except: print("N/A")

print("\n--- Earnings Dates ---")
try: print(ticker.earnings_dates.head())
except: print("N/A")
