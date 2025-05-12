from backend.controller import Controller

controller = Controller()

print("Portfolio Positions:")
for ticker, position in controller.portfolio.positions.items():
    print(f"Position: {ticker}")
    print(f"  Purchase Date: {position.purchase_date}")
    print(f"  Amount: {position.amount}")
    print(f"  Purchase Price: {position.purchase_price_net.amount} {position.purchase_price_net.currency}")

print("\nTransaction History:")
transactions = controller.portfolio.get_transaction_history()
for i, t in enumerate(transactions):
    print(f"Transaction {i+1}:")
    print(f"  Type: {t.type}")
    print(f"  Ticker: {t.stock.ticker}")
    print(f"  Amount: {t.amount}")
    print(f"  Date: {t.date}") 