from backend.stock import Stock
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_country_detection():
    """Test the country detection logic for various tickers"""
    test_tickers = [
        # US stocks
        'AAPL',    # Apple
        'MSFT',    # Microsoft
        'GOOGL',   # Alphabet
        'AMZN',    # Amazon
        'TSLA',    # Tesla
        
        # UK stocks
        'BP.L',    # BP (London)
        'HSBA.L',  # HSBC (London)
        
        # German stocks
        'BMW.DE',  # BMW (Germany)
        'SAP.DE',  # SAP (Germany)
        
        # Other European stocks
        'MC.PA',   # LVMH (Paris)
        'PHIA.AS', # Philips (Amsterdam)
        
        # Asian stocks
        '7203.T',  # Toyota (Tokyo)
        '0700.HK', # Tencent (Hong Kong)
        '005930.KS', # Samsung (Korea)
        
        # Indices
        '^GSPC',   # S&P 500
        '^FTSE',   # FTSE 100
        '^GDAXI',  # DAX
        '^N225',   # Nikkei 225
    ]
    
    print("Testing country detection logic:")
    print("--------------------------------")
    
    for ticker in test_tickers:
        try:
            stock = Stock(ticker)
            country = stock.get_country()
            print(f"{ticker:10} -> {country}")
        except Exception as e:
            print(f"{ticker:10} -> Error: {e}")
    
    print("--------------------------------")
    print("Test completed")

if __name__ == "__main__":
    test_country_detection() 