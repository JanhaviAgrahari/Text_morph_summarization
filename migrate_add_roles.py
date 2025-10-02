"""
Migration script to add the 'role' column to the users table.
Run this once to update existing database schema.
"""
import sqlite3
import os
from pathlib import Path

# Database path
DB_PATH = Path(__file__).parent / "database" / "text_morph.db"

def migrate_add_role_column():
    """Add role column to users table if it doesn't exist."""
    if not DB_PATH.exists():
        print(f"❌ Database not found at {DB_PATH}")
        return False
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Check if role column already exists
        cursor.execute("PRAGMA table_info(users)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'role' in columns:
            print("✅ Role column already exists in users table")
            return True
        
        # Add role column with default value 'user'
        print("➕ Adding role column to users table...")
        cursor.execute("""
            ALTER TABLE users 
            ADD COLUMN role TEXT NOT NULL DEFAULT 'user'
        """)
        
        # Create index on role column for faster lookups
        print("➕ Creating index on role column...")
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_users_role ON users(role)
        """)
        
        conn.commit()
        print("✅ Migration completed successfully!")
        print("ℹ️  All existing users have been assigned the 'user' role by default.")
        print("ℹ️  To create an admin user, update a user's role manually:")
        print("   UPDATE users SET role = 'admin' WHERE email = 'admin@example.com';")
        
        return True
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        conn.rollback()
        return False
        
    finally:
        conn.close()


if __name__ == "__main__":
    print("=" * 60)
    print("Database Migration: Add User Roles")
    print("=" * 60)
    migrate_add_role_column()
    print("=" * 60)
