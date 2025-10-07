#!/usr/bin/env python3
"""
Inspect MongoDB data to see what's actually stored.
"""
import os
from dotenv import load_dotenv
from mongodb_persistence import get_mongodb_persistence

load_dotenv()

def inspect_mongodb():
    """Inspect all data in MongoDB."""
    print("🔍 Inspecting MongoDB Data...")
    print("=" * 60)
    
    try:
        db = get_mongodb_persistence()
        
        print("\n📝 CURRENT CONVERSATION:")
        print("-" * 30)
        current_conversation = db.load_conversation()
        if current_conversation:
            print(f"Messages: {len(current_conversation)}")
            for i, msg in enumerate(current_conversation, 1):
                role = msg.get('role', 'unknown')
                content = msg.get('content', '')
                preview = content[:100] + "..." if len(content) > 100 else content
                print(f"  {i}. [{role}] {preview}")
        else:
            print("  No current conversation found")
        
        print("\n💾 SAVED SESSIONS:")
        print("-" * 30)
        sessions = db.list_sessions()
        if sessions:
            print(f"Total sessions: {len(sessions)}")
            for i, session_name in enumerate(sessions, 1):
                session_data = db.load_session(session_name)
                msg_count = len(session_data) if session_data else 0
                print(f"  {i}. '{session_name}' - {msg_count} messages")
                
                # Show first message preview if exists
                if session_data and len(session_data) > 0:
                    first_msg = session_data[0]
                    preview = first_msg.get('content', '')[:80] + "..." if len(first_msg.get('content', '')) > 80 else first_msg.get('content', '')
                    print(f"      First message: {preview}")
        else:
            print("  No saved sessions found")
        
        print("\n⚙️  METADATA:")
        print("-" * 30)
        last_session = db._get_last_session()
        autosave = db._get_autosave_flag()
        print(f"  Last session: '{last_session}' (empty if none)")
        print(f"  Autosave enabled: {autosave}")
        
        print("\n📊 STATISTICS:")
        print("-" * 30)
        stats = db.get_session_stats()
        print(f"  Session count: {stats['session_count']}")
        print(f"  Total messages: {stats['total_messages']}")
        print(f"  Average messages per session: {stats['avg_messages_per_session']}")
        
        # Raw collection inspection
        print("\n🔧 RAW COLLECTION DATA:")
        print("-" * 30)
        
        # Check conversations collection
        conv_docs = list(db.conversations_collection.find())
        print(f"  Conversations collection: {len(conv_docs)} documents")
        for doc in conv_docs:
            doc_type = doc.get('type', 'unknown')
            history_len = len(doc.get('history', []))
            print(f"    - Type: {doc_type}, Messages: {history_len}")
        
        # Check sessions collection
        session_docs = list(db.sessions_collection.find())
        print(f"  Sessions collection: {len(session_docs)} documents")
        for doc in session_docs:
            name = doc.get('name', 'unnamed')
            history_len = len(doc.get('history', []))
            created = doc.get('created_at', 'unknown')
            print(f"    - '{name}': {history_len} messages (created: {created})")
        
        # Check metadata collection
        meta_docs = list(db.metadata_collection.find())
        print(f"  Metadata collection: {len(meta_docs)} documents")
        for doc in meta_docs:
            key = doc.get('key', 'unknown')
            value = doc.get('value', 'unknown')
            print(f"    - {key}: {value}")
        
        print("\n✅ Inspection complete!")
        
    except Exception as e:
        print(f"❌ Error during inspection: {e}")

if __name__ == "__main__":
    inspect_mongodb()