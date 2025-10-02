"""
Test script for the feedback system
Run this after starting the backend server to test the feedback endpoint
"""

import requests
import json

# Configuration
BACKEND_URL = "http://localhost:8000"

def test_feedback_system():
    print("=" * 60)
    print("FEEDBACK SYSTEM TEST")
    print("=" * 60)
    
    # Step 1: Login as a test user
    print("\n1. Testing login...")
    login_data = {
        "username": "test@example.com",  # Change to your test user
        "password": "testpassword"       # Change to your test password
    }
    
    try:
        login_response = requests.post(f"{BACKEND_URL}/token", data=login_data)
        if login_response.status_code == 200:
            token = login_response.json().get("access_token")
            print(f"   ✅ Login successful! Token: {token[:20]}...")
        else:
            print(f"   ❌ Login failed: {login_response.status_code}")
            print(f"   Please create a test user first or update credentials")
            return
    except Exception as e:
        print(f"   ❌ Login error: {e}")
        print(f"   Make sure the backend server is running on {BACKEND_URL}")
        return
    
    # Step 2: Create a summary to get an entry_id
    print("\n2. Testing summary generation...")
    headers = {"Authorization": f"Bearer {token}"}
    
    summary_data = {
        "summary_length": "medium",
        "input_type": "text",
        "text_input": "This is a test text for summarization. It needs to be long enough to be summarized properly. This is just a test to get an entry ID for feedback testing. The system will process this and create a history entry.",
        "model_choice": "pegasus"
    }
    
    try:
        summary_response = requests.post(
            f"{BACKEND_URL}/summarize/",
            data=summary_data,
            headers=headers
        )
        
        if summary_response.status_code == 200:
            result = summary_response.json()
            entry_id = result.get("entry_id")
            print(f"   ✅ Summary generated! Entry ID: {entry_id}")
            print(f"   Summary: {result.get('summary', '')[:50]}...")
        else:
            print(f"   ❌ Summary generation failed: {summary_response.status_code}")
            print(f"   Response: {summary_response.text}")
            return
    except Exception as e:
        print(f"   ❌ Summary error: {e}")
        return
    
    if not entry_id:
        print("   ❌ No entry_id returned. Check if user_email is set in session.")
        return
    
    # Step 3: Submit positive feedback
    print("\n3. Testing thumbs up feedback...")
    feedback_payload = {
        "rating": "thumbs_up",
        "comment": "This is a test positive feedback comment!"
    }
    
    try:
        feedback_response = requests.post(
            f"{BACKEND_URL}/history/{entry_id}/feedback",
            json=feedback_payload,
            headers=headers
        )
        
        if feedback_response.status_code == 200:
            feedback_result = feedback_response.json()
            print(f"   ✅ Feedback submitted successfully!")
            print(f"   Response: {json.dumps(feedback_result, indent=2)}")
        else:
            print(f"   ❌ Feedback submission failed: {feedback_response.status_code}")
            print(f"   Response: {feedback_response.text}")
            return
    except Exception as e:
        print(f"   ❌ Feedback error: {e}")
        return
    
    # Step 4: Try to submit feedback again (should update)
    print("\n4. Testing feedback update (thumbs down)...")
    feedback_payload2 = {
        "rating": "thumbs_down",
        "comment": "Changed my mind, this is negative feedback now!"
    }
    
    try:
        feedback_response2 = requests.post(
            f"{BACKEND_URL}/history/{entry_id}/feedback",
            json=feedback_payload2,
            headers=headers
        )
        
        if feedback_response2.status_code == 200:
            feedback_result2 = feedback_response2.json()
            print(f"   ✅ Feedback updated successfully!")
            print(f"   Response: {json.dumps(feedback_result2, indent=2)}")
        else:
            print(f"   ❌ Feedback update failed: {feedback_response2.status_code}")
            print(f"   Response: {feedback_response2.text}")
    except Exception as e:
        print(f"   ❌ Feedback update error: {e}")
    
    # Step 5: Test invalid rating
    print("\n5. Testing invalid rating...")
    invalid_payload = {
        "rating": "invalid_rating",
        "comment": "This should fail"
    }
    
    try:
        invalid_response = requests.post(
            f"{BACKEND_URL}/history/{entry_id}/feedback",
            json=invalid_payload,
            headers=headers
        )
        
        if invalid_response.status_code == 400:
            print(f"   ✅ Invalid rating properly rejected!")
            print(f"   Response: {invalid_response.json()}")
        else:
            print(f"   ❌ Expected 400 error but got: {invalid_response.status_code}")
    except Exception as e:
        print(f"   ❌ Invalid rating test error: {e}")
    
    # Step 6: Test accessing non-existent entry
    print("\n6. Testing non-existent entry...")
    try:
        nonexist_response = requests.post(
            f"{BACKEND_URL}/history/999999/feedback",
            json={"rating": "thumbs_up"},
            headers=headers
        )
        
        if nonexist_response.status_code == 404:
            print(f"   ✅ Non-existent entry properly rejected!")
            print(f"   Response: {nonexist_response.json()}")
        else:
            print(f"   ❌ Expected 404 error but got: {nonexist_response.status_code}")
    except Exception as e:
        print(f"   ❌ Non-existent entry test error: {e}")
    
    print("\n" + "=" * 60)
    print("FEEDBACK SYSTEM TEST COMPLETE")
    print("=" * 60)
    print("\n💡 Next steps:")
    print("1. Check admin panel to view the feedback")
    print("2. Try the feedback UI in the Streamlit app")
    print("3. Test with different users")
    print("=" * 60)

if __name__ == "__main__":
    test_feedback_system()
