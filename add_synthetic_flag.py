import os
import sys
import sqlite3
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Database file path
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend', 'portfolio.db')

def add_synthetic_data_column():
    """Add a synthetic_data column to the market_data table"""
    try:
        # Connect to the database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check if the column already exists
        cursor.execute("PRAGMA table_info(market_data)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'is_synthetic' not in columns:
            # Add the column
            cursor.execute("ALTER TABLE market_data ADD COLUMN is_synthetic BOOLEAN DEFAULT 0")
            conn.commit()
            logger.info("Added is_synthetic column to market_data table")
        else:
            logger.info("is_synthetic column already exists in market_data table")
            
        # Close the connection
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Error adding is_synthetic column: {e}")
        return False

if __name__ == "__main__":
    logger.info(f"Running migration on database: {DB_PATH}")
    if add_synthetic_data_column():
        logger.info("Migration completed successfully")
    else:
        logger.error("Migration failed")
        sys.exit(1) 