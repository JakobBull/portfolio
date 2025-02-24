
class Position:

    def __init__(self, transaction):
        if transaction.type != "buy":
            raise ValueError(f"Transaction type {transaction.type} is not supported. New positions can only be opened with a 'buy'.")
        self.amount = transaction.amount
        self.stock = transaction.stock
        self.purchase_price_net = transaction.price
        self.purchase_cost = transaction.cost
        self.purchase_price_gross = transaction.price + transaction.cost

    def update(self, transaction):
        if transaction.type == "buy":
            self.amount += transaction.amount
        elif transaction.type == "sell":
            if transaction.amount <= self.amount:
                self.amount -= transaction.amount
            else:
                raise ValueError(f"Trying to sell {transaction.amount} shares of {self.ticker}, but only {self.amount} shares are owned.")
        else:
            raise ValueError(f"Transaction type {transaction.type} is not supported. Supported types are 'buy' and 'sell'.")

    def get_value(self, currency, exchange_date):
        return self.stock.get_value(currency, exchange_date) * self.amount
