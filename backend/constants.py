DEFAULT_CURRENCY = "EUR"

# Supported currencies
SUPPORTED_CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD"]

# Market interface settings
YFINANCE_MAX_REQUESTS = 1000  # Maximum requests per hour (more conservative estimate)
YFINANCE_TIME_WINDOW = 3600   # Time window in seconds (1 hour)
YFINANCE_BURST_LIMIT = 30     # Maximum requests per minute (more conservative)
YFINANCE_BURST_TIME = 60      # Burst time window in seconds
YFINANCE_MAX_RETRIES = 2      # Maximum number of retries for failed requests (reduced)
YFINANCE_RETRY_DELAY = 10     # Delay between retries in seconds (increased)
YFINANCE_COOLDOWN_PERIOD = 300  # Cooldown period in seconds after detecting rate limit (5 minutes)
YFINANCE_BATCH_SIZE = 5       # Number of tickers to batch in a single request
YFINANCE_REQUEST_TIMEOUT = 15  # Timeout for API requests in seconds

# Cache settings
CACHE_EXPIRY_EXCHANGE_RATES = 14  # Days to keep exchange rates in cache (increased)
CACHE_EXPIRY_STOCK_PRICES = 3     # Days to keep stock prices in cache (increased)
CACHE_EXPIRY_HISTORICAL = 14      # Days to keep historical data in cache (increased)
CACHE_NEARBY_DATE_RANGE = 7       # Days to look for nearby dates in cache when exact date not found

# API status tracking
API_STATUS_CHECK_INTERVAL = 60  # Seconds between API status checks
API_FAILURE_THRESHOLD = 5       # Number of consecutive failures before assuming API is down
API_SUCCESS_THRESHOLD = 3       # Number of consecutive successes before assuming API is back up

# Default exchange rates (used as fallback when API fails)
DEFAULT_EXCHANGE_RATES = {
    'USD_EUR': 0.85,
    'EUR_USD': 1.18,
    'GBP_EUR': 1.17,
    'EUR_GBP': 0.85,
    'USD_GBP': 0.73,
    'GBP_USD': 1.37,
    'JPY_USD': 0.0091,
    'USD_JPY': 110.0,
    'EUR_JPY': 129.5,
    'JPY_EUR': 0.0077,
    'CHF_USD': 1.09,
    'USD_CHF': 0.92,
    'EUR_CHF': 1.08,
    'CHF_EUR': 0.93,
    'CAD_USD': 0.80,
    'USD_CAD': 1.25,
    'EUR_CAD': 1.47,
    'CAD_EUR': 0.68,
    'AUD_USD': 0.75,
    'USD_AUD': 1.33,
    'EUR_AUD': 1.57,
    'AUD_EUR': 0.64,
}