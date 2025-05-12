from backend.database import db_manager

def check_database():
    # Check portfolio positions
    positions = db_manager.get_all_positions()
    print('Portfolio positions:')
    for pos in positions:
        print(f"Ticker: {pos['ticker']}, Name: {pos['name']}, Sector: {pos['sector']}")
    
    # Check watchlist items
    watchlist = db_manager.get_all_watchlist_items()
    print('\nWatchlist items:')
    for item in watchlist:
        print(f"Ticker: {item['ticker']}, Name: {item['name']}, Sector: {item['sector']}")

if __name__ == "__main__":
    check_database() 