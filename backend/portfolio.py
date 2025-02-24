from backend.transaction import Transaction

class Portfolio:

    def __init__(self):
        self.positions = {}
        self.transactions = []

    def get_value(self, currency):
        pass

    @property
    def euro_value(self)
        retiurn self.get_value('EUR')

    def get_gross_purchase_price(self, currency):
        pass

    @property
    def euro_gross_purchase_price(self):
        return self.get_gross_purchase_price('EUR')

    def get_net_purchase_price(self, currency):
        pass

    @property
    def euro_net_purchase_price(self):
        return self.get_net_purchase_price('EUR')

    def transaction(self, transaction_type, stock, amount, price, currency, date):
        transaction = Transaction(transaction_type, stock, amount, price, currency date)

        self.update_position(transaction)
        self.transactions.append(transaction)
        return True

    def update_position(self, transaction):
        if transaction.type == "buy":
            if transaction.stock.ticker not in self.positions.keys():
                self.positions[transaction.stock.ticker] = Position(transaction)
            else:
                self.positions[transaction.stock.ticker].update_position(transaction)
        elif transaction.type == "sell":
            if transaction.stock.ticker not in self.positions.keys():
                return False
            else:
                self.positions[transaction.stock.ticker].update(transaction)
                if self.positions[transaction.stock.ticker].amount == 0:
                    del self.positions[transaction.stock.ticker]
        return True

