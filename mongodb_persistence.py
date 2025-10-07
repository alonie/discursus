"""
MongoDB-based persistence for Discursus conversations and sessions.
"""
import os
import time
from typing import List, Dict, Optional
from datetime import datetime
import pymongo
from pymongo import MongoClient
from pymongo.errors import PyMongoError


class MongoDBPersistence:
    """Handle all MongoDB operations for conversation persistence."""
    
    def __init__(self, connection_string: Optional[str] = None):
        """Initialize MongoDB connection."""
        self.connection_string = connection_string or os.getenv("MONGODB_URI")
        if not self.connection_string:
            raise ValueError("MongoDB connection string not provided. Set MONGODB_URI environment variable.")
        
        self.client = None
        self.db = None
        self.conversations_collection = None
        self.sessions_collection = None
        self.metadata_collection = None
        self._connect()
    
    def _connect(self):
        """Establish connection to MongoDB with optimized settings."""
        try:
            # Optimized connection settings for better performance
            self.client = MongoClient(
                self.connection_string,
                maxPoolSize=10,  # Connection pool size
                minPoolSize=1,   # Minimum connections
                maxIdleTimeMS=30000,  # Close connections after 30s idle
                serverSelectionTimeoutMS=5000,  # 5s timeout for server selection
                socketTimeoutMS=10000,  # 10s socket timeout
                connectTimeoutMS=5000,  # 5s connection timeout
                retryWrites=True,
                w='majority'  # Write concern for data safety
            )
            # Test connection with shorter timeout
            self.client.admin.command('ping')
            
            # Use 'discursus' database
            self.db = self.client.discursus
            
            # Collections
            self.conversations_collection = self.db.conversations
            self.sessions_collection = self.db.sessions
            self.metadata_collection = self.db.metadata
            
            # Create indexes for better performance
            self.conversations_collection.create_index([("created_at", -1)])
            self.sessions_collection.create_index([("name", 1)], unique=True)
            self.sessions_collection.create_index([("created_at", -1)])
            
            print("✅ Connected to MongoDB")
            
        except PyMongoError as e:
            print(f"❌ Failed to connect to MongoDB: {e}")
            raise
    
    def save_conversation(self, history: List[Dict]) -> bool:
        """Save conversation history to MongoDB."""
        try:
            # Serialize conversation data
            serializable = []
            for m in history:
                serializable.append({
                    "role": m.get("role"),
                    "content": m.get("content")
                })
            
            # Update or insert the current conversation
            document = {
                "history": serializable,
                "updated_at": datetime.utcnow(),
                "message_count": len(serializable)
            }
            
            # Use upsert to update existing or create new
            result = self.conversations_collection.update_one(
                {"type": "current"},
                {
                    "$set": document,
                    "$setOnInsert": {
                        "type": "current",
                        "created_at": datetime.utcnow()
                    }
                },
                upsert=True
            )
            
            # Auto-create a named session on first save if none exists
            try:
                if not self._get_last_session():
                    ts_name = time.strftime("session-%A-%d-%B-%Y_%H-%M-%S", time.localtime())
                    self.save_session(ts_name, history)
            except Exception:
                pass
            
            # If autosave flag set, also persist into last named session
            try:
                if self._get_autosave_flag():
                    last_session = self._get_last_session()
                    if last_session:
                        self.save_session(last_session, history)
            except Exception:
                pass
            
            return True
            
        except PyMongoError as e:
            print(f"Error saving conversation: {e}")
            return False
    
    def load_conversation(self) -> List[Dict]:
        """Load current conversation history from MongoDB."""
        try:
            doc = self.conversations_collection.find_one({"type": "current"})
            if doc and "history" in doc and len(doc["history"]) > 0:
                # Ensure data is list of dicts with role/content
                out = []
                for m in doc["history"]:
                    if isinstance(m, dict):
                        out.append({
                            "role": m.get("role", "user"), 
                            "content": m.get("content", "")
                        })
                return out
            
            # If current conversation is empty, try to load the last meaningful session
            last_session = self._get_last_session()
            if last_session:
                session_data = self.load_session(last_session)
                if session_data and len(session_data) > 0:
                    # Restore this session as the current conversation
                    self.save_conversation(session_data)
                    return session_data
            
            # If no last session or it's empty, try to find any session with content
            sessions = self.list_sessions()
            for session_name in sessions:
                session_data = self.load_session(session_name)
                if session_data and len(session_data) > 0:
                    # Use this session as the current conversation
                    self.save_conversation(session_data)
                    self._set_last_session(session_name)
                    return session_data
            
            return []
            
        except PyMongoError as e:
            print(f"Error loading conversation: {e}")
            return []
    
    def save_session(self, name: str, history: List[Dict]) -> bool:
        """Save a named session to MongoDB."""
        if not name:
            name = time.strftime("session-%A-%d-%B-%Y_%H-%M-%S", time.localtime())
        
        try:
            # Serialize conversation data
            serializable = []
            for m in history:
                serializable.append({
                    "role": m.get("role"),
                    "content": m.get("content")
                })
            
            document = {
                "name": name,
                "history": serializable,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "message_count": len(serializable)
            }
            
            # Use upsert to update existing or create new session
            self.sessions_collection.update_one(
                {"name": name},
                {"$set": document},
                upsert=True
            )
            
            # Remember last session
            self._set_last_session(name)
            return True
            
        except PyMongoError as e:
            print(f"Error saving session '{name}': {e}")
            return False
    
    def load_session(self, name: str) -> List[Dict]:
        """Load a named session from MongoDB."""
        if not name:
            return []
        
        try:
            doc = self.sessions_collection.find_one({"name": name})
            if doc and "history" in doc:
                return doc["history"]
            return []
            
        except PyMongoError as e:
            print(f"Error loading session '{name}': {e}")
            return []
    
    def list_sessions(self) -> List[str]:
        """List all saved session names."""
        try:
            sessions = self.sessions_collection.find(
                {}, 
                {"name": 1, "_id": 0}
            ).sort("created_at", -1)
            
            return [session["name"] for session in sessions]
            
        except PyMongoError as e:
            print(f"Error listing sessions: {e}")
            return []
    
    def delete_session(self, name: str) -> bool:
        """Delete a named session."""
        if not name:
            return False
        
        try:
            result = self.sessions_collection.delete_one({"name": name})
            
            # If it was the last session, clear last session pointer
            if self._get_last_session() == name:
                self._set_last_session("")
            
            return result.deleted_count > 0
            
        except PyMongoError as e:
            print(f"Error deleting session '{name}': {e}")
            return False
    
    def _get_last_session(self) -> Optional[str]:
        """Get the last used session name."""
        try:
            doc = self.metadata_collection.find_one({"key": "last_session"})
            return doc.get("value", "") if doc else ""
        except PyMongoError:
            return ""
    
    def _set_last_session(self, name: str):
        """Set the last used session name."""
        try:
            self.metadata_collection.update_one(
                {"key": "last_session"},
                {
                    "$set": {
                        "value": name,
                        "updated_at": datetime.utcnow()
                    },
                    "$setOnInsert": {
                        "created_at": datetime.utcnow()
                    }
                },
                upsert=True
            )
        except PyMongoError:
            pass
    
    def _get_autosave_flag(self) -> bool:
        """Check if autosave is enabled."""
        try:
            doc = self.metadata_collection.find_one({"key": "autosave"})
            return doc.get("value", False) if doc else False
        except PyMongoError:
            return False
    
    def set_autosave_flag(self, enabled: bool):
        """Set autosave flag."""
        try:
            self.metadata_collection.update_one(
                {"key": "autosave"},
                {
                    "$set": {
                        "value": enabled,
                        "updated_at": datetime.utcnow()
                    },
                    "$setOnInsert": {
                        "created_at": datetime.utcnow()
                    }
                },
                upsert=True
            )
        except PyMongoError:
            pass
    
    def get_session_stats(self) -> Dict:
        """Get statistics about sessions and conversations."""
        try:
            session_count = self.sessions_collection.count_documents({})
            
            # Get total messages across all sessions
            pipeline = [
                {"$group": {
                    "_id": None,
                    "total_messages": {"$sum": "$message_count"},
                    "avg_messages": {"$avg": "$message_count"}
                }}
            ]
            
            stats = list(self.sessions_collection.aggregate(pipeline))
            total_messages = stats[0]["total_messages"] if stats else 0
            avg_messages = stats[0]["avg_messages"] if stats else 0
            
            return {
                "session_count": session_count,
                "total_messages": total_messages,
                "avg_messages_per_session": round(avg_messages, 1)
            }
            
        except PyMongoError as e:
            print(f"Error getting stats: {e}")
            return {"session_count": 0, "total_messages": 0, "avg_messages_per_session": 0}
    
    def close(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()


# Global instance - will be initialized when first used
_mongodb_instance = None

def get_mongodb_persistence() -> MongoDBPersistence:
    """Get or create the global MongoDB persistence instance."""
    global _mongodb_instance
    if _mongodb_instance is None:
        _mongodb_instance = MongoDBPersistence()
    return _mongodb_instance