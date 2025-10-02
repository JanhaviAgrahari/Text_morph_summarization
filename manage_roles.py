"""
Script to manage user roles in the Text Morph application.
Allows promoting users to admin or mentor roles.
"""
import sqlite3
import sys
from pathlib import Path

# Database path
DB_PATH = Path(__file__).parent / "database" / "text_morph.db"

def list_users(conn):
    """List all users with their current roles."""
    cursor = conn.cursor()
    cursor.execute("SELECT id, email, role, name FROM users ORDER BY id")
    users = cursor.fetchall()
    
    if not users:
        print("No users found in database.")
        return
    
    print("\n" + "=" * 80)
    print(f"{'ID':<5} {'Email':<35} {'Role':<15} {'Name':<20}")
    print("=" * 80)
    for user in users:
        user_id, email, role, name = user
        name = name or "(not set)"
        print(f"{user_id:<5} {email:<35} {role:<15} {name:<20}")
    print("=" * 80 + "\n")


def update_user_role(conn, email, new_role):
    """Update a user's role."""
    valid_roles = ['user', 'mentor', 'admin']
    if new_role not in valid_roles:
        print(f"❌ Invalid role. Must be one of: {', '.join(valid_roles)}")
        return False
    
    cursor = conn.cursor()
    
    # Check if user exists
    cursor.execute("SELECT id, email, role FROM users WHERE email = ?", (email,))
    user = cursor.fetchone()
    
    if not user:
        print(f"❌ User with email '{email}' not found.")
        return False
    
    user_id, user_email, old_role = user
    
    if old_role == new_role:
        print(f"ℹ️  User '{email}' already has role '{new_role}'")
        return True
    
    # Update role
    cursor.execute("UPDATE users SET role = ? WHERE email = ?", (new_role, email))
    conn.commit()
    
    print(f"✅ User '{email}' role updated: {old_role} → {new_role}")
    return True


def interactive_mode(conn):
    """Interactive mode for updating user roles."""
    while True:
        print("\n" + "=" * 60)
        print("User Role Management - Interactive Mode")
        print("=" * 60)
        print("1. List all users")
        print("2. Update user role")
        print("3. Exit")
        print("=" * 60)
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            list_users(conn)
        
        elif choice == "2":
            email = input("Enter user email: ").strip()
            print("\nAvailable roles:")
            print("  - user (default, regular user)")
            print("  - mentor (can view all summaries/paraphrases)")
            print("  - admin (full access)")
            new_role = input("\nEnter new role: ").strip().lower()
            update_user_role(conn, email, new_role)
        
        elif choice == "3":
            print("Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")


def main():
    """Main entry point."""
    if not DB_PATH.exists():
        print(f"❌ Database not found at {DB_PATH}")
        print("Please run the application first to create the database.")
        sys.exit(1)
    
    conn = sqlite3.connect(DB_PATH)
    
    try:
        # Check if role column exists
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(users)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'role' not in columns:
            print("❌ Role column not found in users table.")
            print("Please run migrate_add_roles.py first.")
            sys.exit(1)
        
        # Command line arguments
        if len(sys.argv) == 3:
            email = sys.argv[1]
            role = sys.argv[2].lower()
            update_user_role(conn, email, role)
        else:
            # Interactive mode
            interactive_mode(conn)
    
    finally:
        conn.close()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help"]:
        print("Usage:")
        print("  python manage_roles.py                    # Interactive mode")
        print("  python manage_roles.py <email> <role>     # Update specific user")
        print()
        print("Examples:")
        print("  python manage_roles.py admin@example.com admin")
        print("  python manage_roles.py mentor@example.com mentor")
        sys.exit(0)
    
    main()
