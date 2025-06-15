# Portfolio Manager

A comprehensive portfolio management system with tax calculations, dividend tracking, and benchmark comparison.

## Features

- **Portfolio Management**: Track positions, transactions, and performance
- **German Tax Calculations**: Calculate taxes according to German tax rules
- **Dividend Tracking**: Monitor dividend income and yield
- **Benchmark Comparison**: Compare portfolio performance against major indices
- **Interactive Dashboard**: Visualize portfolio data and performance
- **Enhanced Market Interface**: Robust API handling with rate limiting and persistent caching

## Backend Architecture

The backend is built using Python with SQLAlchemy for database management. Here's an overview of the key components:

### Database Models

- **Stock**: Stores basic stock information (ticker, name, currency)
- **StockPrice**: Historical price data for stocks
- **Position**: Portfolio positions with purchase details
- **Transaction**: Buy/sell/dividend transactions
- **Watchlist**: Stocks to watch with target prices

### Core Components

1. **Database Manager**
   - Handles all database operations
   - Uses SQLAlchemy ORM for data access
   - Provides CRUD operations for all models
   - Manages database sessions and transactions

2. **Portfolio Manager**
   - Tracks positions and transactions
   - Calculates portfolio value and performance
   - Handles position updates and cost basis tracking

3. **Market Interface**
   - Fetches stock prices and market data
   - Implements rate limiting for API calls
   - Caches data for offline access
   - Handles currency conversions

4. **Tax Calculator**
   - Implements German tax rules
   - Calculates capital gains tax
   - Handles dividend taxation
   - Supports partial exemption for funds

5. **Benchmark Comparison**
   - Compares portfolio performance against indices
   - Supports custom benchmarks
   - Calculates relative performance metrics

### Key Features

1. **Data Persistence**
   - SQLite database for local storage
   - Automatic schema management
   - Transaction support for data integrity

2. **Error Handling**
   - Graceful error recovery
   - Detailed error logging
   - Transaction rollback on failure

3. **Performance Optimization**
   - Efficient database queries
   - Connection pooling
   - Caching of frequently accessed data

4. **Extensibility**
   - Modular design for easy extension
   - Clear separation of concerns
   - Well-defined interfaces

## Getting Started

### Prerequisites

- Python 3.8+
- SQLite 3
- Required Python packages (see requirements.txt)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/portfolio-manager.git
   cd portfolio-manager
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Initialize the database:
   ```
   python backend/migrate_db.py
   ```

### Usage

1. Start the backend server:
   ```
   python backend/server.py
   ```

2. Access the dashboard at http://localhost:8050

## Testing

The application includes a comprehensive test suite using pytest. Run the tests with:

```
pytest tests/
```

Key test areas:
- Database operations
- Portfolio management
- Market data handling
- Tax calculations
- Benchmark comparisons

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.