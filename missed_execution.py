
import subprocess
from datetime import datetime, timedelta

def check_timestamp(timestamp):
    # Calculate the current date and time
    current_time = datetime.now()
    print(f"Checking missed execution at time {current_time}.")

    # Convert the timestamp to a datetime object
    timestamp_datetime = datetime.fromtimestamp(timestamp)

    # Check if the time difference is more than 1 day
    if timestamp_datetime.date() != current_time.date() and current_time.hour >= 8:
        # Check if the timestamp falls within Tuesday to Saturday (1 to 5, where Monday is 0 and Sunday is 6)
        if 1 <= timestamp_datetime.weekday() <= 5:
            return True
    return False

# Path to the file storing the last execution time
last_execution_file = 'last_execution.txt'
# Load the last execution time from the file
try:
    with open(last_execution_file, 'r') as f:
        last_execution_timestamp = float(f.read().strip())
except FileNotFoundError:
    # Set an initial last execution time if the file doesn't exist
    last_execution_timestamp = 0

# Check if the difference exceeds the threshold
if check_timestamp(last_execution_timestamp):
    # Perform necessary actions for missed executions
    print("Missed execution detected. Performing catch-up actions...")
    subprocess.run(['python', 'script.py'])
    # Update the last execution time
    with open(last_execution_file, 'w') as f:
        f.write(str(datetime.now().timestamp()))
else:
    print("No missed execution.")
