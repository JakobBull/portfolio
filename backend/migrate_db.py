#!/usr/bin/env python3
"""
Database migration script for the portfolio management system.
This script handles schema changes and data migrations.
"""

import os
import sys
import sqlite3
from datetime import datetime
import logging

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database import db_manager, DB_PATH

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def backup_database():
    """Create a backup of the current database."""
    if not os.path.exists(DB_PATH):
        logger.info("No database found to backup")
        return None
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{DB_PATH}.backup_{timestamp}"
    
    try:
        import shutil
        shutil.copy2(DB_PATH, backup_path)
        logger.info(f"Database backed up to: {backup_path}")
        return backup_path
    except Exception as e:
        logger.error(f"Failed to backup database: {e}")
        return None

def drop_positions_table():
    """Drop the positions table since we now calculate positions from transactions."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Check if positions table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='positions'")
        if cursor.fetchone():
            logger.info("Dropping positions table...")
            cursor.execute("DROP TABLE positions")
            conn.commit()
            logger.info("Positions table dropped successfully")
        else:
            logger.info("Positions table does not exist")
            
        conn.close()
        
    except Exception as e:
        logger.error(f"Error dropping positions table: {e}")
        raise

def update_stock_table():
    """Remove position relationship from stocks table if it exists."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # SQLite doesn't support dropping columns, so we'll just log that the relationship is gone
        logger.info("Stock table positions relationship removed (handled by SQLAlchemy)")
        
        conn.close()
        
    except Exception as e:
        logger.error(f"Error updating stock table: {e}")
        raise

def verify_transactions_table():
    """Verify that the transactions table has all required columns."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get table info
        cursor.execute("PRAGMA table_info(transactions)")
        columns = cursor.fetchall()
        column_names = [col[1] for col in columns]
        
        required_columns = ['id', 'type', 'ticker', 'amount', 'price', 'currency', 'date', 'cost']
        missing_columns = [col for col in required_columns if col not in column_names]
        
        if missing_columns:
            logger.error(f"Missing columns in transactions table: {missing_columns}")
            raise ValueError(f"Transactions table is missing required columns: {missing_columns}")
        else:
            logger.info("Transactions table has all required columns")
            
        conn.close()
        
    except Exception as e:
        logger.error(f"Error verifying transactions table: {e}")
        raise

def main():
    """Main migration function."""
    logger.info("Starting database migration to remove positions table...")
    
    # Create backup
    backup_path = backup_database()
    if backup_path:
        logger.info(f"Backup created at: {backup_path}")
    
    try:
        # Verify transactions table
        verify_transactions_table()
        
        # Drop positions table
        drop_positions_table()
        
        # Update stock table (relationships handled by SQLAlchemy)
        update_stock_table()
        
        # Recreate database schema to ensure consistency
        logger.info("Recreating database schema...")
        from backend.database import Base, engine
        Base.metadata.create_all(bind=engine)
        
        logger.info("Migration completed successfully!")
        logger.info("Portfolio positions will now be calculated dynamically from transactions.")
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        if backup_path and os.path.exists(backup_path):
            logger.info(f"You can restore from backup: {backup_path}")
        sys.exit(1)

if __name__ == "__main__":
    main() 