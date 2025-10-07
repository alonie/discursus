#!/usr/bin/env python3
"""
Fix the current conversation by loading the most recent meaningful session.
"""
import os
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

from mongodb_persistence import get_mongodb_persistence

def fix_current_conversation():
    """Load the most recent meaningful session as the current conversation."""
    print("🔧 Fixing current conversation...")
    
    try:
        db = get_mongodb_persistence()
        
        # Check current conversation
        current = db.load_conversation()
        print(f"Current conversation: {len(current)} messages")
        
        if not current or len(current) == 0:
            print("Current conversation is empty, looking for meaningful session...")
            
            # Get all sessions
            sessions = db.list_sessions()
            print(f"Available sessions: {len(sessions)}")
            
            # Find the most recent session with actual content
            for session_name in sessions:
                session_data = db.load_session(session_name)
                if session_data and len(session_data) > 0:
                    print(f"✅ Loading session '{session_name}' with {len(session_data)} messages as current conversation")
                    
                    # Set this as the current conversation
                    db.save_conversation(session_data)
                    db._set_last_session(session_name)
                    
                    # Show preview of loaded conversation
                    print("\nLoaded conversation preview:")
                    for i, msg in enumerate(session_data[:3], 1):  # Show first 3 messages
                        role = msg.get('role', 'unknown')
                        content = msg.get('content', '')[:100] + "..." if len(msg.get('content', '')) > 100 else msg.get('content', '')
                        print(f"  {i}. [{role}] {content}")
                    
                    if len(session_data) > 3:
                        print(f"  ... and {len(session_data) - 3} more messages")
                    
                    print(f"\n🎉 Current conversation restored with {len(session_data)} messages!")
                    return True
            
            print("❌ No meaningful sessions found to restore.")
            return False
        else:
            print(f"✅ Current conversation already has {len(current)} messages.")
            return True
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    fix_current_conversation()