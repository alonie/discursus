#!/usr/bin/env python3
"""
Migration script to move existing file-based conversation data to MongoDB.
Run this script after setting up your MongoDB connection in .env file.
"""
import os
import json
import sys
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import MongoDB persistence
from mongodb_persistence import get_mongodb_persistence

def migrate_current_conversation():
    """Migrate the current conversation.json to MongoDB."""
    persist_path = os.path.join(os.path.dirname(__file__), "data", "conversation.json")
    
    if not os.path.exists(persist_path):
        print("ℹ️  No current conversation file found to migrate.")
        return
    
    try:
        with open(persist_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if data:
            db = get_mongodb_persistence()
            db.save_conversation(data)
            print(f"✅ Migrated current conversation ({len(data)} messages)")
        else:
            print("ℹ️  Current conversation file is empty.")
            
    except Exception as e:
        print(f"❌ Failed to migrate current conversation: {e}")

def migrate_sessions():
    """Migrate all session files to MongoDB."""
    sessions_dir = os.path.join(os.path.dirname(__file__), "data", "sessions")
    
    if not os.path.exists(sessions_dir):
        print("ℹ️  No sessions directory found to migrate.")
        return
    
    try:
        files = [f for f in os.listdir(sessions_dir) if f.endswith(".json")]
        
        if not files:
            print("ℹ️  No session files found to migrate.")
            return
        
        db = get_mongodb_persistence()
        migrated_count = 0
        
        for filename in files:
            session_name = os.path.splitext(filename)[0]
            filepath = os.path.join(sessions_dir, filename)
            
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    history = json.load(f)
                
                if history:
                    db.save_session(session_name, history)
                    migrated_count += 1
                    print(f"✅ Migrated session: {session_name} ({len(history)} messages)")
                else:
                    print(f"⚠️  Skipped empty session: {session_name}")
                    
            except Exception as e:
                print(f"❌ Failed to migrate session {session_name}: {e}")
        
        print(f"✅ Migration complete: {migrated_count} sessions migrated")
        
    except Exception as e:
        print(f"❌ Failed to migrate sessions: {e}")

def migrate_autosave_flag():
    """Migrate the autosave flag to MongoDB."""
    autosave_path = os.path.join(os.path.dirname(__file__), "data", "autosave_flag.txt")
    
    if not os.path.exists(autosave_path):
        print("ℹ️  No autosave flag file found to migrate.")
        return
    
    try:
        with open(autosave_path, "r", encoding="utf-8") as f:
            flag_value = f.read().strip() == "1"
        
        db = get_mongodb_persistence()
        db.set_autosave_flag(flag_value)
        print(f"✅ Migrated autosave flag: {flag_value}")
        
    except Exception as e:
        print(f"❌ Failed to migrate autosave flag: {e}")

def migrate_last_session():
    """Migrate the last session pointer to MongoDB."""
    last_session_path = os.path.join(os.path.dirname(__file__), "data", "last_session.txt")
    
    if not os.path.exists(last_session_path):
        print("ℹ️  No last session file found to migrate.")
        return
    
    try:
        with open(last_session_path, "r", encoding="utf-8") as f:
            last_session = f.read().strip()
        
        if last_session:
            db = get_mongodb_persistence()
            db._set_last_session(last_session)
            print(f"✅ Migrated last session pointer: {last_session}")
        else:
            print("ℹ️  Last session file is empty.")
            
    except Exception as e:
        print(f"❌ Failed to migrate last session: {e}")

def main():
    """Run the complete migration."""
    print("🚀 Starting migration from file-based storage to MongoDB...")
    print("=" * 60)
    
    # Check if MongoDB connection is configured
    if not os.getenv("MONGODB_URI"):
        print("❌ MONGODB_URI not found in environment variables.")
        print("Please set your MongoDB connection string in .env file:")
        print("MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/discursus")
        return 1
    
    try:
        # Test MongoDB connection
        db = get_mongodb_persistence()
        print("✅ MongoDB connection successful")
        print()
        
        # Run migrations
        migrate_current_conversation()
        migrate_sessions()
        migrate_autosave_flag()
        migrate_last_session()
        
        print()
        print("=" * 60)
        print("🎉 Migration completed successfully!")
        print()
        print("Next steps:")
        print("1. Test the application to ensure everything works")
        print("2. Once confirmed, you can safely delete the 'data' directory")
        print("3. Your conversations are now stored in MongoDB")
        
        # Show stats
        stats = db.get_session_stats()
        print()
        print("📊 MongoDB Statistics:")
        print(f"   Sessions: {stats['session_count']}")
        print(f"   Total messages: {stats['total_messages']}")
        print(f"   Average messages per session: {stats['avg_messages_per_session']}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        print()
        print("Please check:")
        print("1. MongoDB connection string is correct")
        print("2. MongoDB cluster is accessible")
        print("3. Network connectivity to MongoDB")
        return 1

if __name__ == "__main__":
    sys.exit(main())