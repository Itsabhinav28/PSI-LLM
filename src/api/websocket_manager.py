"""
WebSocket Manager for Real-Time Document Updates
"""

import asyncio
import json
import logging
from typing import Dict, Set, Any, Optional
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

class WebSocketManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        self.active_connections: Dict[str, Any] = {}
        self.connection_groups: Dict[str, Set[str]] = {
            "document_updates": set(),
            "stats_updates": set(),
            "general": set()
        }
        self._running = True
        logger.info("WebSocket Manager initialized")
    
    async def connect(self, websocket: Any, client_id: Optional[str] = None) -> str:
        """Connect a new WebSocket client."""
        if not client_id:
            client_id = str(uuid.uuid4())
        
        self.active_connections[client_id] = websocket
        self.connection_groups["general"].add(client_id)
        
        logger.info(f"WebSocket client connected: {client_id}")
        return client_id
    
    async def disconnect(self, client_id: str):
        """Disconnect a WebSocket client."""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            for group in self.connection_groups.values():
                group.discard(client_id)
            logger.info(f"WebSocket client disconnected: {client_id}")
    
    async def broadcast_to_group(self, group: str, message: Dict[str, Any]):
        """Broadcast a message to all clients in a specific group."""
        if group not in self.connection_groups:
            return
        
        disconnected_clients = []
        
        for client_id in self.connection_groups[group]:
            try:
                websocket = self.active_connections[client_id]
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error broadcasting to {client_id}: {e}")
                disconnected_clients.append(client_id)
        
        for client_id in disconnected_clients:
            await self.disconnect(client_id)
    
    async def get_connection_stats(self) -> Dict[str, Any]:
        """Get current connection statistics."""
        return {
            "total_connections": len(self.active_connections),
            "group_stats": {
                group: len(clients) for group, clients in self.connection_groups.items()
            },
            "timestamp": datetime.now().isoformat()
        }

# Global instance
websocket_manager = WebSocketManager()
