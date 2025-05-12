import csv
from datetime import datetime
import sys
import yfinance as yf
from utilities import calculate_fees

def main():
    trade_time = str(datetime.now())

    # Set a default value if no command-line argument is provided
    default_value = {
        "Stock": "NAME",
        "Ticker": "TICKER",
        "Number": 0,
    }

    # Specify the path to the existing CSV file
    current_csv_file_path = 'current_portfolio.csv'
    trade_csv_file_path = "trade_history.csv"

    argument = {}

    # Check if a command-line argument is provided
    if len(sys.argv) == 4:
        # Extract the command-line argument
        argument["Stock"] = sys.argv[1]
        argument["Ticker"] = sys.argv[2]
        argument["Number"] = sys.argv[3]
        print("Values read from command line.")
    else:
        if len(sys.argv) > 1 and len(sys.argv) < 4:
            print("Missing arguments.")
        # No command-line argument provided, use default value
        argument = default_value
        print("Values provided in script.")



    stock_data = yf.Ticker(argument["Ticker"])

    try:
        price = stock_data.history(period='1d')['Close'].iloc[-1]
    except IndexError:
        print(f"No data found for stock {argument['Ticker']}.")

    trade_cost = calculate_fees(price, int(argument['Number']))

    trade_data = [argument["Stock"], argument["Ticker"], price, argument['Number'], 
                  trade_time, "SELL", trade_cost]

    with open(current_csv_file_path, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    edited = True
    remove_row = None

    # Check if stock_ticker matches any values from the CSV file's 'Ticker' column
    for row in rows:
        if row['Ticker'] == argument["Ticker"]:
            if int(argument["Number"]) == int(row["Number"]):
                remove_row = row
            elif int(argument["Number"]) < int(row["Number"]):
                row["Number"] = str(int(row['Number']) - int(argument['Number']))
            else:
                edited = False
                print(f"You do not have this many {argument['Ticker']} stocks.")

    if edited:
        if remove_row:
            rows = [row for row in rows if row != remove_row]

        # Write the updated contents back to the CSV file
        with open(current_csv_file_path, mode='w', newline='') as file:
            fieldnames = ['Stock', 'Ticker', 'Number', 'Sell Target']
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            # Write the header
            writer.writeheader()

            # Write the updated rows
            writer.writerows(rows)

        with open(trade_csv_file_path, mode='a', newline='') as file:
            # Create a CSV writer object
            writer = csv.writer(file)

            # Write the new row to the CSV file
            writer.writerow(trade_data)

if __name__ == "__main__":
    main()
