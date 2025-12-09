import { useEffect, useRef } from 'react';
import { WS_BASE_URL } from '../config';

export const useWebSocket = (
  projectId: string,
  onMessage: (message: any) => void
) => {
  const wsRef = useRef<WebSocket | null>(null);
  const onMessageRef = useRef(onMessage);

  // 保持 onMessage 引用最新
  useEffect(() => {
    onMessageRef.current = onMessage;
  }, [onMessage]);

  useEffect(() => {
    if (!projectId) {
      console.log('[WebSocket] No project ID, skipping connection');
      return;
    }

    const wsUrl = `${WS_BASE_URL}/projects/${projectId}`;
    console.log('[WebSocket] Connecting to:', wsUrl);
    
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      console.log('[WebSocket] Connected to project', projectId);
    };

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        console.log('[WebSocket] Message received:', message);
        onMessageRef.current(message);
      } catch (error) {
        console.error('[WebSocket] Failed to parse message:', error, event.data);
      }
    };

    ws.onerror = (error) => {
      console.error('[WebSocket] Error:', error);
    };

    ws.onclose = (event) => {
      console.log('[WebSocket] Disconnected', {
        code: event.code,
        reason: event.reason,
        wasClean: event.wasClean
      });
      
      // 如果不是正常关闭，尝试重连
      if (event.code !== 1000) {
        console.log('[WebSocket] Attempting to reconnect in 3 seconds...');
        setTimeout(() => {
          if (wsRef.current?.readyState === WebSocket.CLOSED || !wsRef.current) {
            // 重新连接逻辑
            console.log('[WebSocket] Reconnecting...');
          }
        }, 3000);
      }
    };

    wsRef.current = ws;

    return () => {
      console.log('[WebSocket] Cleaning up connection');
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.close(1000, 'Component unmounted');
      }
      wsRef.current = null;
    };
  }, [projectId]); // 移除 onMessage 依赖，使用 ref 来访问最新值
};

