import os

# Activate the Conda environment
os.system('/Users/jakobbull/anaconda3/envs/portfolio/bin')
#Navigate to correct directory
os.chdir('/Users/jakobbull/Documents/Projects/portfolio')

import pandas as pd
import yfinance as yf
from datetime import datetime


last_execution_file = 'last_execution.txt'

# Read the watchlist CSV file
watchlist_df = pd.read_csv('watchlist.csv')
selllist_df = pd.read_csv('selllist.csv')
reviewlist_df = pd.read_csv('reviewlist.csv')

# Get the current date and time
current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
current_day = datetime.now().strftime("%Y-%m-%d")
current_month = datetime.now().strftime("%Y-%m")

# Define the folder path for alarms
buy_alarm_folder = f'alarms/buy/{current_month}'
sell_alarm_folder = f'alarms/sell/{current_month}'
review_alarm_folder = f'alarms/review/{current_month}'

# Create the alarms folder if it does not exist
if not os.path.exists(buy_alarm_folder):
    os.makedirs(buy_alarm_folder)

if not os.path.exists(sell_alarm_folder):
    os.makedirs(sell_alarm_folder)

if not os.path.exists(review_alarm_folder):
    os.makedirs(review_alarm_folder)

#-----BUY CHECK-----#

# Create an empty list to store stocks below target
below_target_stocks = []

# Check each stock in the watchlist
for index, row in watchlist_df.iterrows():
    name = row["Stock"]
    ticker = row['Ticker']
    target_price = row['Price Target']
    
    # Fetch stock data using Yahoo Finance API
    try:
        stock_data = yf.Ticker(ticker)
        current_price = stock_data.history(period='1d')['Close'].iloc[-1]
        # Check if current price is below target price
        if current_price < target_price:
            below_target_stocks.append({'Stock': name, 'Ticker': ticker, 'Price Target': target_price, 'Current Price': current_price})
    except IndexError:
        print(f"No data available for {name} with ticker: {ticker}.")
    finally:
        pass


# If there are stocks below target, create an alarm CSV file
if below_target_stocks:
    buy_alarm_df = pd.DataFrame(below_target_stocks)
    buy_alarm_file_name = os.path.join(buy_alarm_folder, f'buy_alarm_{current_day}.csv')
    buy_alarm_df.to_csv(buy_alarm_file_name, index=False)
    print(f'Buy alarm file "{buy_alarm_file_name}" created successfully!')
else:
    print(f'No stocks found below target prices on {current_day}.')

#-----SELL CHECK-----#

above_target_stocks = []

# Check each stock in the watchlist
for index, row in selllist_df.iterrows():
    name = row["Stock"]
    ticker = row['Ticker']
    target_price = row['Price Target']
    
    # Fetch stock data using Yahoo Finance API
    try:
        stock_data = yf.Ticker(ticker)
        current_price = stock_data.history(period='1d')['Close'].iloc[-1]
    except IndexError:
        print(f"No data available for {name} with ticker: {ticker}.")
    finally:
        pass

    # Check if current price is below target price
    if current_price > target_price:
        above_target_stocks.append({'Stock': name, 'Ticker': ticker, 'Price Target': target_price, 'Current Price': current_price})

# If there are stocks below target, create an alarm CSV file
if above_target_stocks:
    sell_alarm_df = pd.DataFrame(above_target_stocks)
    sell_alarm_file_name = os.path.join(sell_alarm_folder, f'sell_alarm_{current_day}.csv')
    sell_alarm_df.to_csv(sell_alarm_file_name, index=False)
    print(f'Sell alarm file "{sell_alarm_file_name}" created successfully!')
else:
    print(f'No stocks found above target prices on {current_day}.')

#-----TO BE REVIEWED CHECK-----#


# Create an empty list to store stocks below target
to_be_reviewed_stocks = []

# Check each stock in the watchlist
for index, row in reviewlist_df.iterrows():
    name = row["Stock"]
    ticker = row['Ticker']
    review_date = row['Review Date']

    # Check if current price is below target price
    if datetime.strptime(review_date, '%Y-%m-%d').date() <=  datetime.now().date():
        to_be_reviewed_stocks.append({'Stock': name, 'Ticker': ticker, 'Review Date': review_date})

# If there are stocks below target, create an alarm CSV file
if to_be_reviewed_stocks:
    review_alarm_df = pd.DataFrame(to_be_reviewed_stocks)
    review_alarm_file_name = os.path.join(review_alarm_folder, f'review_alarm_{current_day}.csv')
    review_alarm_df.to_csv(review_alarm_file_name, index=False)
    print(f'Review alarm file "{review_alarm_file_name}" created successfully!')
else:
    print(f'No stocks to be reviewed on {current_day}.')

# Update the last execution time
with open(last_execution_file, 'w') as f:
    f.write(str(datetime.now().timestamp()))