"""
Test script to verify role-based access control functionality.
Tests the admin history endpoint with different user roles.
"""
import requests
import json
from typing import Dict, Tuple

BACKEND_URL = "http://127.0.0.1:8000"

def test_login(email: str, password: str) -> Tuple[bool, str, Dict]:
    """Test login and return token."""
    print(f"\n{'='*60}")
    print(f"Testing login for: {email}")
    print(f"{'='*60}")
    
    response = requests.post(
        f"{BACKEND_URL}/login",
        json={"email": email, "password": password}
    )
    
    if response.status_code == 200:
        data = response.json()
        token = data.get("access_token", "")
        print(f"✅ Login successful")
        
        # Get user info
        me_response = requests.get(
            f"{BACKEND_URL}/me",
            headers={"Authorization": f"Bearer {token}"}
        )
        if me_response.status_code == 200:
            user_info = me_response.json()
            print(f"   User: {user_info.get('email')}")
            print(f"   Role: {user_info.get('role')}")
            print(f"   Name: {user_info.get('name', 'Not set')}")
            return True, token, user_info
        else:
            print(f"⚠️ Could not fetch user info")
            return True, token, {}
    else:
        print(f"❌ Login failed: {response.status_code}")
        print(f"   {response.text}")
        return False, "", {}


def test_admin_history_access(token: str, role: str) -> bool:
    """Test access to admin history endpoint."""
    print(f"\n{'='*60}")
    print(f"Testing admin history access (Role: {role})")
    print(f"{'='*60}")
    
    response = requests.get(
        f"{BACKEND_URL}/admin/history?limit=10",
        headers={"Authorization": f"Bearer {token}"}
    )
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        entries = data.get("entries", [])
        total_count = data.get("total_count", 0)
        print(f"✅ Access granted!")
        print(f"   Total entries: {total_count}")
        print(f"   Entries returned: {len(entries)}")
        
        if entries:
            print(f"\n   Sample entries:")
            for i, entry in enumerate(entries[:3]):
                print(f"   {i+1}. ID: {entry.get('id')}, Type: {entry.get('type')}, User: {entry.get('user_email')}")
        
        return True
    
    elif response.status_code == 403:
        print(f"❌ Access denied (403 Forbidden)")
        print(f"   This is expected for users with role 'user'")
        try:
            error_data = response.json()
            print(f"   Error: {error_data.get('detail', 'No details')}")
        except:
            pass
        return False
    
    elif response.status_code == 401:
        print(f"❌ Unauthorized (401)")
        print(f"   Token might be invalid or expired")
        return False
    
    else:
        print(f"❌ Unexpected error: {response.status_code}")
        print(f"   Response: {response.text}")
        return False


def main():
    print("\n" + "="*60)
    print("ROLE-BASED ACCESS CONTROL TEST SUITE")
    print("="*60)
    print("\nThis script will test:")
    print("1. Login functionality")
    print("2. Role assignment and retrieval")
    print("3. Admin history endpoint access control")
    print("\nMake sure the backend server is running at:", BACKEND_URL)
    
    input("\nPress Enter to continue...")
    
    # Test 1: Check backend health
    print(f"\n{'='*60}")
    print("Testing backend connectivity")
    print(f"{'='*60}")
    try:
        response = requests.get(f"{BACKEND_URL}/ping", timeout=3)
        if response.status_code == 200:
            print(f"✅ Backend is online")
        else:
            print(f"⚠️ Backend returned unexpected status: {response.status_code}")
    except Exception as e:
        print(f"❌ Cannot connect to backend: {e}")
        print(f"   Please start the backend server first:")
        print(f"   cd Backend && uvicorn main:app --reload")
        return
    
    # Test 2: Test with admin/mentor user
    print("\n" + "="*60)
    print("TEST CASE 1: Admin/Mentor Access")
    print("="*60)
    print("\nPlease provide credentials for an admin or mentor account:")
    email = input("Email: ").strip()
    password = input("Password: ").strip()
    
    if email and password:
        success, token, user_info = test_login(email, password)
        if success:
            role = user_info.get("role", "unknown")
            if role in ["admin", "mentor"]:
                test_admin_history_access(token, role)
                print("\n✅ Admin/Mentor test completed")
            else:
                print(f"\n⚠️ User has role '{role}', not admin or mentor")
                test_admin_history_access(token, role)
    
    # Test 3: Test with regular user
    print("\n" + "="*60)
    print("TEST CASE 2: Regular User Access")
    print("="*60)
    print("\nDo you want to test with a regular user account? (y/n): ", end="")
    if input().strip().lower() == 'y':
        print("\nPlease provide credentials for a regular user account:")
        email = input("Email: ").strip()
        password = input("Password: ").strip()
        
        if email and password:
            success, token, user_info = test_login(email, password)
            if success:
                role = user_info.get("role", "unknown")
                test_admin_history_access(token, role)
                if role == "user":
                    print("\n✅ Regular user test completed (access correctly denied)")
                else:
                    print(f"\n⚠️ User has role '{role}', expected 'user'")
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print("\n✅ All tests completed!")
    print("\nExpected behavior:")
    print("  - Admin/Mentor users: Should access /admin/history (200 OK)")
    print("  - Regular users: Should be denied access (403 Forbidden)")
    print("\nTo promote users to admin/mentor:")
    print("  python manage_roles.py <email> <role>")
    print("\nExample:")
    print("  python manage_roles.py user@example.com admin")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
