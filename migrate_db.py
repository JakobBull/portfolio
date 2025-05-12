import os
import sqlite3
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database file path
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend', 'portfolio.db')

def migrate_database():
    """Add country columns to the database tables"""
    try:
        # Connect to the database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check if the database file exists
        if not os.path.exists(DB_PATH):
            logger.error(f"Database file not found at {DB_PATH}")
            return False
            
        logger.info(f"Connected to database at {DB_PATH}")
        
        # Add country column to portfolio table
        try:
            cursor.execute("ALTER TABLE portfolio ADD COLUMN country TEXT")
            logger.info("Added country column to portfolio table")
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e):
                logger.info("Country column already exists in portfolio table")
            else:
                logger.error(f"Error adding country column to portfolio table: {e}")
                
        # Add country column to transactions table
        try:
            cursor.execute("ALTER TABLE transactions ADD COLUMN country TEXT")
            logger.info("Added country column to transactions table")
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e):
                logger.info("Country column already exists in transactions table")
            else:
                logger.error(f"Error adding country column to transactions table: {e}")
                
        # Add country column to watchlist table
        try:
            cursor.execute("ALTER TABLE watchlist ADD COLUMN country TEXT")
            logger.info("Added country column to watchlist table")
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e):
                logger.info("Country column already exists in watchlist table")
            else:
                logger.error(f"Error adding country column to watchlist table: {e}")
                
        # Commit changes and close connection
        conn.commit()
        conn.close()
        
        logger.info("Database migration completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error migrating database: {e}")
        return False
        
if __name__ == "__main__":
    migrate_database() 