"""
Database Setup Script for Text Morph Summarization
--------------------------------------------------
This script initializes the SQLite database with the necessary tables.
Run this script once to set up the database before starting the application.
"""

import os
from pathlib import Path
import logging
import sqlite3
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_database():
    """Set up the SQLite database for the Text Morph Summarization application."""
    try:
        # Create database directory if it doesn't exist
        db_dir = Path(__file__).parent / "database"
        db_dir.mkdir(exist_ok=True)
        
        # Path to the database file
        db_path = db_dir / "text_morph.db"
        
        # Connect to SQLite database (creates it if it doesn't exist)
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Create users table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            name TEXT,
            age_group TEXT,
            language TEXT
        )
        ''')
        
        # Create password_resets table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS password_resets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            token TEXT UNIQUE NOT NULL,
            expires_at TIMESTAMP NOT NULL,
            used BOOLEAN NOT NULL DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        ''')
        
        # Create history table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            type TEXT NOT NULL,
            original_text TEXT NOT NULL,
            result_text TEXT NOT NULL,
            model TEXT NOT NULL,
            parameters TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_users_email ON users (email)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_password_resets_token ON password_resets (token)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_history_user_id ON history (user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_history_type ON history (type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_history_created_at ON history (created_at)')
        
        # Commit changes and close connection
        conn.commit()
        conn.close()
        
        logger.info(f"SQLite database initialized at: {db_path.absolute()}")
        return True
    except Exception as e:
        logger.error(f"Error setting up database: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting database setup...")
    if setup_database():
        logger.info("Database setup completed successfully!")
    else:
        logger.error("Database setup failed!")
