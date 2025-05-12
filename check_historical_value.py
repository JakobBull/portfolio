from backend.controller import Controller
import datetime
import pandas as pd

controller = Controller()

# Get the earliest transaction date
transactions = controller.portfolio.get_transaction_history()
earliest_date = min(t.date for t in transactions) if transactions else datetime.date.today()
print(f"Earliest transaction date: {earliest_date}")

# Get historical value for a range of dates
start_date = earliest_date - datetime.timedelta(days=10)  # Start 10 days before first transaction
end_date = datetime.date.today()
print(f"Calculating historical value from {start_date} to {end_date}")

# Get historical value data
df = controller.get_historical_value(start_date, end_date, 'EUR', 'daily')

# Print summary statistics
print(f"Data shape: {df.shape}")
print("First few rows:")
print(df.head())
print("Last few rows:")
print(df.tail())

# Check for zero values
zero_values = df[df['value'] == 0]
print(f"Number of dates with zero values: {len(zero_values)}")

# Check for non-zero values
non_zero_values = df[df['value'] > 0]
print(f"Number of dates with non-zero values: {len(non_zero_values)}")

# Print the first date with a non-zero value
if not non_zero_values.empty:
    first_non_zero_date = non_zero_values['date'].min()
    print(f"First date with non-zero value: {first_non_zero_date}")
    
    # Print the value on that date
    first_non_zero_row = non_zero_values[non_zero_values['date'] == first_non_zero_date]
    print(f"Value on {first_non_zero_date}: {first_non_zero_row['value'].iloc[0]}")
else:
    print("No dates with non-zero values found")

# Check if there are any dates after the first transaction with zero values
if not zero_values.empty and not transactions:
    zero_after_first = zero_values[zero_values['date'] >= earliest_date]
    print(f"Number of dates with zero values after first transaction: {len(zero_after_first)}")
    if not zero_after_first.empty:
        print("First few dates with zero values after first transaction:")
        print(zero_after_first.head())

# Now check the portfolio performance data
print("\nChecking portfolio performance data:")
performance_df = controller.get_portfolio_performance(start_date, end_date, 'EUR')
print(f"Performance data shape: {performance_df.shape}")
print("First few rows:")
print(performance_df.head())
print("Last few rows:")
print(performance_df.tail())

# Check normalized values
print(f"Normalized value range: min={performance_df['normalized_value'].min()}, max={performance_df['normalized_value'].max()}")

# Check if all normalized values are the same
if performance_df['normalized_value'].nunique() == 1:
    print("WARNING: All normalized values are the same!")
else:
    print(f"Number of unique normalized values: {performance_df['normalized_value'].nunique()}")
    
    # Print some sample normalized values
    print("Sample normalized values:")
    sample_indices = sorted(performance_df.index[::len(performance_df)//10])[:10]  # Get ~10 evenly spaced indices
    for idx in sample_indices:
        if idx < len(performance_df):
            row = performance_df.iloc[idx]
            print(f"Date: {row['date']}, Value: {row['value']}, Normalized: {row['normalized_value']}") 