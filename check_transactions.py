from backend.database import db_manager

def check_transactions():
    # Get all transactions
    transactions = db_manager.get_all_transactions()
    
    # Print the 5 most recent transactions
    print('Recent transactions:')
    for t in transactions[-5:]:
        print(f"Ticker: {t['ticker']}, Type: {t['type']}, Amount: {t['amount']}, Price: {t['price']}, Cost: {t['cost']}, Date: {t['transaction_date']}")
    
    # Check positions
    positions = db_manager.get_all_positions()
    print('\nPositions:')
    for pos in positions:
        print(f"Ticker: {pos['ticker']}, Name: {pos['name']}, Amount: {pos['amount']}, Cost Basis: {pos['cost_basis']}")

if __name__ == "__main__":
    check_transactions() 