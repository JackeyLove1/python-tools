'''
pip install yfinance
pip install -U pdblp
'''
import yfinance as yf

def get_stock_data(symbol):
    stock = yf.Ticker(symbol)
    # get historical market data
    hist = stock.history(period="5d")  # get 5 days' data
    return hist

print(get_stock_data("GOOG"))