"""
Script to remove duplicate history entries from the database.
This happens when both frontend and backend save history.
"""
import sqlite3
from pathlib import Path
from collections import defaultdict

# Database path
DB_PATH = Path(__file__).parent / "database" / "text_morph.db"

def find_and_remove_duplicates():
    """Find and remove duplicate history entries."""
    if not DB_PATH.exists():
        print(f"❌ Database not found at {DB_PATH}")
        return
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Find all history entries
        cursor.execute("""
            SELECT id, user_id, type, original_text, result_text, model, created_at
            FROM history
            ORDER BY created_at
        """)
        
        entries = cursor.fetchall()
        print(f"\n📊 Total history entries: {len(entries)}")
        
        if not entries:
            print("✅ No entries found in database.")
            return
        
        # Group by user_id, type, original_text, result_text, model
        # If created within 5 seconds of each other, consider them duplicates
        duplicates_to_remove = []
        seen = {}
        
        for entry_id, user_id, entry_type, original_text, result_text, model, created_at in entries:
            # Create a unique key for this entry (excluding ID and timestamp)
            key = (user_id, entry_type, original_text[:100], result_text[:100], model)
            
            if key in seen:
                # Check if created within 5 seconds
                prev_id, prev_time = seen[key]
                
                # Parse timestamps (SQLite stores as string)
                try:
                    from datetime import datetime
                    time1 = datetime.fromisoformat(prev_time.replace('Z', '+00:00') if 'Z' in prev_time else prev_time)
                    time2 = datetime.fromisoformat(created_at.replace('Z', '+00:00') if 'Z' in created_at else created_at)
                    time_diff = abs((time2 - time1).total_seconds())
                    
                    if time_diff <= 5:
                        # This is a duplicate - keep the first one, remove this one
                        duplicates_to_remove.append(entry_id)
                        print(f"   Found duplicate: ID {entry_id} (within {time_diff:.1f}s of ID {prev_id})")
                except Exception as e:
                    print(f"   Warning: Could not parse timestamps: {e}")
            else:
                seen[key] = (entry_id, created_at)
        
        if not duplicates_to_remove:
            print("✅ No duplicates found!")
            return
        
        print(f"\n🔍 Found {len(duplicates_to_remove)} duplicate entries")
        print(f"   IDs to remove: {duplicates_to_remove[:10]}{'...' if len(duplicates_to_remove) > 10 else ''}")
        
        # Ask for confirmation
        print(f"\n⚠️  This will permanently delete {len(duplicates_to_remove)} entries.")
        response = input("   Continue? (yes/no): ").strip().lower()
        
        if response != 'yes':
            print("❌ Cancelled. No entries were deleted.")
            return
        
        # Delete duplicates
        cursor.executemany("DELETE FROM history WHERE id = ?", [(id,) for id in duplicates_to_remove])
        conn.commit()
        
        print(f"\n✅ Successfully removed {len(duplicates_to_remove)} duplicate entries!")
        
        # Show final count
        cursor.execute("SELECT COUNT(*) FROM history")
        final_count = cursor.fetchone()[0]
        print(f"📊 Remaining history entries: {final_count}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        conn.rollback()
        import traceback
        traceback.print_exc()
    finally:
        conn.close()


def main():
    print("=" * 60)
    print("History Duplicate Removal Tool")
    print("=" * 60)
    print("\nThis script will:")
    print("1. Find duplicate history entries")
    print("2. Ask for confirmation before deletion")
    print("3. Remove duplicates (keeping the first occurrence)")
    print("\nDuplicates are identified as entries with:")
    print("  - Same user, type, original text, result text, model")
    print("  - Created within 5 seconds of each other")
    
    input("\nPress Enter to continue...")
    find_and_remove_duplicates()
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
