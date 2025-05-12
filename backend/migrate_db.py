import sqlite3
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database file path
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'portfolio.db')

def migrate_database():
    """Add missing columns to the database tables"""
    try:
        # Connect to the database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check if the columns already exist in the portfolio table
        cursor.execute("PRAGMA table_info(portfolio)")
        columns = [column[1] for column in cursor.fetchall()]
        
        # Add name and sector columns to portfolio table if they don't exist
        if 'name' not in columns:
            logger.info("Adding 'name' column to portfolio table")
            cursor.execute("ALTER TABLE portfolio ADD COLUMN name VARCHAR")
        
        if 'sector' not in columns:
            logger.info("Adding 'sector' column to portfolio table")
            cursor.execute("ALTER TABLE portfolio ADD COLUMN sector VARCHAR")
        
        # Check if the columns already exist in the transactions table
        cursor.execute("PRAGMA table_info(transactions)")
        columns = [column[1] for column in cursor.fetchall()]
        
        # Add name and sector columns to transactions table if they don't exist
        if 'name' not in columns:
            logger.info("Adding 'name' column to transactions table")
            cursor.execute("ALTER TABLE transactions ADD COLUMN name VARCHAR")
        
        if 'sector' not in columns:
            logger.info("Adding 'sector' column to transactions table")
            cursor.execute("ALTER TABLE transactions ADD COLUMN sector VARCHAR")
        
        # Check if the columns already exist in the watchlist table
        cursor.execute("PRAGMA table_info(watchlist)")
        columns = [column[1] for column in cursor.fetchall()]
        
        # Add name and sector columns to watchlist table if they don't exist
        if 'name' not in columns:
            logger.info("Adding 'name' column to watchlist table")
            cursor.execute("ALTER TABLE watchlist ADD COLUMN name VARCHAR")
        
        if 'sector' not in columns:
            logger.info("Adding 'sector' column to watchlist table")
            cursor.execute("ALTER TABLE watchlist ADD COLUMN sector VARCHAR")
        
        # Commit the changes
        conn.commit()
        logger.info("Database migration completed successfully")
        
        # Close the connection
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Error migrating database: {e}")
        return False

if __name__ == "__main__":
    migrate_database() 