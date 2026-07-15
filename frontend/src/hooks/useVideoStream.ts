import { useEffect, type RefObject } from 'react';
import { useAppStore } from '../store';
import { openAuthedWebSocket } from '../lib/auth';

export function useVideoStream(canvasRef: RefObject<HTMLCanvasElement | null>, cameraId = 0) {
  const setWsStatus = useAppStore((s) => s.setWsStatus);

  useEffect(() => {
    let aborted = false;
    let reconnectTimer: ReturnType<typeof setTimeout>;
    let ws: WebSocket | null = null;

    async function connect() {
      const proto = location.protocol === 'https:' ? 'wss' : 'ws';
      try {
        ws = await openAuthedWebSocket(`${proto}://${location.host}/ws/video/${cameraId}`);
      } catch {
        if (!aborted) reconnectTimer = setTimeout(connect, 3000);
        return;
      }
      if (aborted) { ws.close(); return; }
      ws.binaryType = 'blob';

      ws.onopen = () => { if (!aborted) setWsStatus(cameraId, 'connected'); };

      ws.onmessage = (e) => {
        if (aborted) return;
        const url = URL.createObjectURL(e.data as Blob);
        const img = new Image();
        img.onload = () => {
          if (aborted) { URL.revokeObjectURL(url); return; }
          const canvas = canvasRef.current;
          if (canvas) {
            const ctx = canvas.getContext('2d');
            ctx?.drawImage(img, 0, 0, canvas.width, canvas.height);
          }
          URL.revokeObjectURL(url);
        };
        img.src = url;
      };

      ws.onclose = () => {
        if (aborted) return;
        setWsStatus(cameraId, 'disconnected');
        reconnectTimer = setTimeout(connect, 3000);
      };

      ws.onerror = () => ws?.close();
    }

    void connect();
    return () => {
      aborted = true;
      clearTimeout(reconnectTimer);
      ws?.close();
    };
  }, [canvasRef, cameraId, setWsStatus]);
}
