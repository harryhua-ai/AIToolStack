"""WebSocket 连接管理器"""
from typing import Dict, Set
from fastapi import WebSocket
import json


class WebSocketManager:
    """WebSocket 连接管理器"""
    
    def __init__(self):
        # project_id -> Set[WebSocket]
        self.active_connections: Dict[str, Set[WebSocket]] = {}
    
    async def connect(self, websocket: WebSocket, project_id: str):
        """接受新连接"""
        await websocket.accept()
        
        if project_id not in self.active_connections:
            self.active_connections[project_id] = set()
        
        self.active_connections[project_id].add(websocket)
        print(f"[WebSocket] Client connected to project {project_id}. Total: {len(self.active_connections[project_id])}")
    
    def disconnect(self, websocket: WebSocket, project_id: str):
        """断开连接"""
        if project_id in self.active_connections:
            self.active_connections[project_id].discard(websocket)
            
            if not self.active_connections[project_id]:
                del self.active_connections[project_id]
            
            print(f"[WebSocket] Client disconnected from project {project_id}")
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """发送个人消息"""
        try:
            await websocket.send_json(message)
        except Exception as e:
            print(f"[WebSocket] Error sending message: {e}")
    
    async def broadcast_to_project(self, project_id: str, message: dict):
        """向项目内所有客户端广播"""
        if project_id not in self.active_connections:
            print(f"[WebSocket] No active connections for project {project_id}")
            return
        
        connection_count = len(self.active_connections[project_id])
        print(f"[WebSocket] Broadcasting to {connection_count} client(s) in project {project_id}")
        
        disconnected = set()
        success_count = 0
        
        for connection in self.active_connections[project_id]:
            try:
                await connection.send_json(message)
                success_count += 1
            except Exception as e:
                print(f"[WebSocket] Error broadcasting to client: {e}")
                disconnected.add(connection)
        
        # 清理断开的连接
        for conn in disconnected:
            self.disconnect(conn, project_id)
        
        print(f"[WebSocket] Successfully sent message to {success_count} client(s)")
    
    def broadcast_project_update(self, project_id: str, update: dict):
        """广播项目更新（异步调用）"""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 如果事件循环正在运行，创建任务
                asyncio.create_task(self.broadcast_to_project(project_id, update))
                print(f"[WebSocket] Broadcasting update to project {project_id}: {update.get('type', 'unknown')}")
            else:
                # 如果事件循环未运行，直接运行
                loop.run_until_complete(self.broadcast_to_project(project_id, update))
                print(f"[WebSocket] Broadcasting update to project {project_id}: {update.get('type', 'unknown')}")
        except RuntimeError:
            # 如果没有事件循环，创建一个新的
            asyncio.run(self.broadcast_to_project(project_id, update))
            print(f"[WebSocket] Broadcasting update to project {project_id}: {update.get('type', 'unknown')}")
        except Exception as e:
            print(f"[WebSocket] Error broadcasting update: {e}")


# 全局 WebSocket 管理器实例
websocket_manager = WebSocketManager()

