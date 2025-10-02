"""
Migration script to add feedback columns to the history table.
Run this once to add feedback_rating, feedback_comment, and feedback_at columns.
"""
import sqlite3
from pathlib import Path

# Database path
DB_PATH = Path(__file__).parent / "database" / "text_morph.db"

def add_feedback_columns():
    """Add feedback columns to the history table."""
    if not DB_PATH.exists():
        print(f"❌ Database not found at {DB_PATH}")
        return
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        print("🔧 Adding feedback columns to history table...")
        
        # Check if columns already exist
        cursor.execute("PRAGMA table_info(history)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'feedback_rating' in columns:
            print("✅ Feedback columns already exist. No migration needed.")
            return
        
        # Add the new columns
        cursor.execute("""
            ALTER TABLE history 
            ADD COLUMN feedback_rating TEXT
        """)
        print("  ✓ Added feedback_rating column")
        
        cursor.execute("""
            ALTER TABLE history 
            ADD COLUMN feedback_comment TEXT
        """)
        print("  ✓ Added feedback_comment column")
        
        cursor.execute("""
            ALTER TABLE history 
            ADD COLUMN feedback_at TIMESTAMP
        """)
        print("  ✓ Added feedback_at column")
        
        conn.commit()
        print("\n✅ Migration completed successfully!")
        print("📊 Feedback columns added: feedback_rating, feedback_comment, feedback_at")
        
    except Exception as e:
        print(f"\n❌ Error during migration: {e}")
        conn.rollback()
        import traceback
        traceback.print_exc()
    finally:
        conn.close()


def main():
    print("=" * 60)
    print("DATABASE MIGRATION: Add Feedback Columns")
    print("=" * 60)
    print()
    
    add_feedback_columns()
    
    print()
    print("=" * 60)
    print("Migration process completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
