import datetime
from constants import DEFAULT_CURRENCY
from market_interface import MarketInterface

class MoneyAmount:

    def __init__(self, amount: float, currency: str = DEFAULT_CURRENCY, date: datetime.date = datetime.date.today()):
        self._amount = amount
        self._currency = currency
        self._date = date
        self._market_interface = MarketInterface()

    def get_money_amount(self, currency: str = DEFAULT_CURRENCY) -> float:
        return self._market_interface.convert_currency(self._amount, self._currency, currency, self._date)

    @property
    def euro_value(self):
        return self.get_money_amount("EUR")

if __name__ == "__main__":
    money = MoneyAmount(100, "USD", datetime.date(2025, 2, 21))
    print(money.euro_value)  # 90.0