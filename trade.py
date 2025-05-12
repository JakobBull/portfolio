import sys
import subprocess

def main():
    # Check if the correct number of command-line arguments are provided
    if len(sys.argv) < 2:
        print("Usage: python trade.py BUY/SELL [additional arguments]")
        sys.exit(1)

    # Get the action (BUY or SELL) from the command line
    action = sys.argv[1]

    # Validate the action
    if action not in ['BUY', 'SELL']:
        print("Action must be BUY or SELL")
        sys.exit(1)

    # Set the path to the appropriate script based on the action
    script = 'add_to_portfolio.py' if action == 'BUY' else 'remove_from_portfolio.py'

    # Set the command to run the script using the appropriate Python interpreter
    command = f'/Users/jakobbull/anaconda3/envs/portfolio/bin/python {script} {" ".join(sys.argv[2:])}'

    # Run the command using subprocess
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running the script: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
