import yfinance as yf
t = yf.Ticker("AAPL")

print("--- Earnings ---")
try: print(t.earnings_dates.head(2))
except: print("No earnings")

print("\n--- Options ---")
try: 
    dates = t.options
    if dates:
        chain = t.option_chain(dates[0])
        print("Calls shape:", chain.calls.shape)
except: print("No options")

print("\n--- Cashflow ---")
try: print(t.cash_flow.head(3))
except: print("No cashflow")
