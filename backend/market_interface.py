import datetime
import yfinance as yf

class MarketInterface:

    def __init__(self):
        pass

    def convert_currency(self, amount, from_currency, to_currency, exchange_date: datetime.date = datetime.date.today()):
        if type(exchange_date) is not datetime.date:
            raise TypeError(f"exchange_date must be a date object, but it is of type {type(exchange_date)}")
        return amount * self.get_rate(from_currency, to_currency, exchange_date)


    def get_rate(self, from_currency, to_currency, date):
        """Get daily closing exchange rate between currencies using Yahoo Finance.
        
        Args:
            from_currency (str): 3-letter base currency code
            to_currency (str): 3-letter target currency code
            date (datetime.date): Historical date for conversion
        
        Returns:
            float: Exchange rate or None if unavailable
        """
        if from_currency == to_currency:
            return 1.0
        
        ticker = f"{from_currency}{to_currency}=X"
        end = date.strftime('%Y-%m-%d')
        start = (date - datetime.timedelta(days=3)).strftime('%Y-%m-%d')
        
        t = yf.Ticker(ticker)

        try:
            data = t.history(
                start=start,
                end=end,
            )
            
            return data.iloc[-1]['Close'] if data.shape[0] > 0 else None
        except Exception as e:
            print(f"Error retrieving rate: {e}")
            return None
