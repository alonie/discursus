#!/usr/bin/env python3
"""
Test script to verify MongoDB connection and basic operations.
Run this after setting up your MongoDB connection string.
"""
import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_mongodb_connection():
    """Test basic MongoDB connectivity and operations."""
    print("🧪 Testing MongoDB connection...")
    print("=" * 50)
    
    # Check if MongoDB URI is configured
    if not os.getenv("MONGODB_URI"):
        print("❌ MONGODB_URI not found in environment variables.")
        print("Please set your MongoDB connection string in .env file:")
        print("MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/discursus")
        return False
    
    try:
        # Import and test MongoDB persistence
        from mongodb_persistence import get_mongodb_persistence
        
        print("✅ MongoDB persistence module imported successfully")
        
        # Test connection
        db = get_mongodb_persistence()
        print("✅ MongoDB connection established")
        
        # Test basic operations
        print("\n🔧 Testing basic operations...")
        
        # Test save/load conversation
        test_conversation = [
            {"role": "user", "content": "Hello, this is a test message"},
            {"role": "assistant", "content": "Hello! This is a test response from the assistant."}
        ]
        
        db.save_conversation(test_conversation)
        print("✅ Conversation save test passed")
        
        loaded_conversation = db.load_conversation()
        if len(loaded_conversation) == 2 and loaded_conversation[0]["content"] == "Hello, this is a test message":
            print("✅ Conversation load test passed")
        else:
            print("❌ Conversation load test failed")
            return False
        
        # Test session operations
        test_session_name = "test_session_" + str(int(time.time()))
        db.save_session(test_session_name, test_conversation)
        print("✅ Session save test passed")
        
        sessions = db.list_sessions()
        if test_session_name in sessions:
            print("✅ Session list test passed")
        else:
            print("❌ Session list test failed")
            return False
        
        loaded_session = db.load_session(test_session_name)
        if len(loaded_session) == 2:
            print("✅ Session load test passed")
        else:
            print("❌ Session load test failed")
            return False
        
        # Test autosave flag
        db.set_autosave_flag(True)
        if db._get_autosave_flag():
            print("✅ Autosave flag test passed")
        else:
            print("❌ Autosave flag test failed")
            return False
        
        # Clean up test data
        db.delete_session(test_session_name)
        db.save_conversation([])  # Clear test conversation
        print("✅ Cleanup completed")
        
        # Get statistics
        stats = db.get_session_stats()
        print(f"\n📊 Current MongoDB Statistics:")
        print(f"   Sessions: {stats['session_count']}")
        print(f"   Total messages: {stats['total_messages']}")
        print(f"   Average messages per session: {stats['avg_messages_per_session']}")
        
        print("\n🎉 All tests passed! MongoDB is ready for use.")
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import MongoDB dependencies: {e}")
        print("Make sure you have installed: pip install pymongo")
        return False
        
    except Exception as e:
        print(f"❌ MongoDB test failed: {e}")
        print("\nPlease check:")
        print("1. MongoDB connection string is correct")
        print("2. MongoDB cluster is accessible")
        print("3. Network connectivity to MongoDB")
        print("4. Database permissions are configured properly")
        return False

if __name__ == "__main__":
    import time
    success = test_mongodb_connection()
    sys.exit(0 if success else 1)