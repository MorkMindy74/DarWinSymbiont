"""
WebSocket manager for real-time evolution updates
"""
import asyncio
import json
from typing import Dict, Set
from fastapi import WebSocket
from datetime import datetime


class ConnectionManager:
    """Manages WebSocket connections for evolution sessions"""
    
    def __init__(self):
        # session_id -> set of WebSocket connections
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        self.lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket, session_id: str):
        """Accept new WebSocket connection"""
        await websocket.accept()
        
        async with self.lock:
            if session_id not in self.active_connections:
                self.active_connections[session_id] = set()
            self.active_connections[session_id].add(websocket)
        
        print(f"✅ WebSocket connected for session {session_id}")
        print(f"   Total connections for session: {len(self.active_connections[session_id])}")
    
    async def disconnect(self, websocket: WebSocket, session_id: str):
        """Remove WebSocket connection"""
        async with self.lock:
            if session_id in self.active_connections:
                self.active_connections[session_id].discard(websocket)
                if not self.active_connections[session_id]:
                    del self.active_connections[session_id]
        
        print(f"❌ WebSocket disconnected for session {session_id}")
    
    async def broadcast_to_session(self, session_id: str, message: dict):
        """Broadcast message to all connections in a session"""
        if session_id not in self.active_connections:
            return
        
        # Add timestamp
        message["timestamp"] = datetime.now().isoformat()
        
        # Get snapshot of connections
        async with self.lock:
            connections = list(self.active_connections.get(session_id, []))
        
        # Send to all connections
        disconnected = []
        for connection in connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                print(f"Error sending to connection: {e}")
                disconnected.append(connection)
        
        # Remove failed connections
        if disconnected:
            async with self.lock:
                for conn in disconnected:
                    if session_id in self.active_connections:
                        self.active_connections[session_id].discard(conn)
    
    def has_connections(self, session_id: str) -> bool:
        """Check if session has active connections"""
        return session_id in self.active_connections and len(self.active_connections[session_id]) > 0


# Global connection manager instance
connection_manager = ConnectionManager()
