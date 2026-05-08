import { useEffect, useRef, useState } from 'react';
import { useAppStore } from '../store';
import { TYPE_LABELS } from '../types';

function ZoomableImage({ src }: { src: string }) {
  const imgRef      = useRef<HTMLImageElement>(null);
  const zoom        = useRef(1);
  const panX        = useRef(0);
  const panY        = useRef(0);
  const panning     = useRef(false);
  const panOriginX  = useRef(0);
  const panOriginY  = useRef(0);

  function applyTransform() {
    const img = imgRef.current;
    if (!img) return;
    img.style.transform = `scale(${zoom.current}) translate(${panX.current / zoom.current}px, ${panY.current / zoom.current}px)`;
    img.style.cursor    = zoom.current > 1 ? 'grab' : 'zoom-in';
  }

  // Reset when modal opens a new image
  useEffect(() => {
    zoom.current = 1; panX.current = 0; panY.current = 0;
    const img = imgRef.current;
    if (img) { img.style.transform = ''; img.style.cursor = 'zoom-in'; }
  }, [src]);

  // Attach wheel + drag listeners once — never re-attaches
  useEffect(() => {
    const img = imgRef.current;
    if (!img) return;

    function onWheel(e: WheelEvent) {
      e.preventDefault();
      const prev   = zoom.current;
      const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
      zoom.current = Math.min(8, Math.max(1, prev * factor));
      const ratio  = zoom.current / prev;
      const rect   = img!.getBoundingClientRect();
      const cx     = e.clientX - (rect.left + rect.width  / 2);
      const cy     = e.clientY - (rect.top  + rect.height / 2);
      panX.current += cx * (1 - ratio);
      panY.current += cy * (1 - ratio);
      if (zoom.current === 1) { panX.current = 0; panY.current = 0; }
      applyTransform();
    }

    function onMouseDown(e: MouseEvent) {
      if (zoom.current <= 1) return;
      e.preventDefault();
      panning.current  = true;
      panOriginX.current = e.clientX - panX.current;
      panOriginY.current = e.clientY - panY.current;
      img!.style.cursor = 'grabbing';
    }

    function onMouseMove(e: MouseEvent) {
      if (!panning.current) return;
      panX.current = e.clientX - panOriginX.current;
      panY.current = e.clientY - panOriginY.current;
      applyTransform();
    }

    function onMouseUp() {
      if (panning.current) { panning.current = false; applyTransform(); }
    }

    img.addEventListener('wheel', onWheel, { passive: false });
    img.addEventListener('mousedown', onMouseDown);
    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);

    return () => {
      img.removeEventListener('wheel', onWheel);
      img.removeEventListener('mousedown', onMouseDown);
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);
    };
  }, []);

  return (
    <img
      ref={imgRef}
      src={src}
      alt="event snapshot"
      style={{ transformOrigin: 'center center', willChange: 'transform', userSelect: 'none' }}
    />
  );
}

export function EventModal() {
  const ev         = useAppStore((s) => s.modalEvent);
  const closeModal = useAppStore((s) => s.closeModal);
  const demo       = useAppStore((s) => s.stats?.demo_mode ?? false);
  const [alertStatus, setAlertStatus] = useState<{ msg: string; ok: boolean } | null>(null);
  const [alertDisabled, setAlertDisabled] = useState(false);
  const [previewActive, setPreviewActive] = useState(false);
  const previewCanvasRef = useRef<HTMLCanvasElement>(null);

  // Reset state when modal opens a new event
  useEffect(() => {
    setAlertStatus(null);
    setAlertDisabled(false);
    setPreviewActive(false);
  }, [ev]);

  // WebSocket preview — opens its own RTSP capture, never touches the pipeline
  useEffect(() => {
    if (!previewActive || !ev) return;
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const camId = ev.camera_id ?? ev.camera ?? 0;
    const ws = new WebSocket(`${proto}://${location.host}/ws/playback/preview?timestamp=${ev.timestamp}&camera_id=${camId}`);
    ws.binaryType = 'blob';
    ws.onmessage = (e) => {
      const url = URL.createObjectURL(e.data as Blob);
      const img = new Image();
      img.onload = () => {
        const canvas = previewCanvasRef.current;
        if (canvas) {
          if (canvas.width !== img.naturalWidth)   canvas.width  = img.naturalWidth;
          if (canvas.height !== img.naturalHeight) canvas.height = img.naturalHeight;
          canvas.getContext('2d')?.drawImage(img, 0, 0);
        }
        URL.revokeObjectURL(url);
      };
      img.src = url;
    };
    return () => ws.close();
  }, [previewActive, ev]);

  // Close on Escape
  useEffect(() => {
    if (!ev) return;
    function onKey(e: KeyboardEvent) { if (e.key === 'Escape') closeModal(); }
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [ev, closeModal]);

  async function sendAlert() {
    if (!ev) return;
    setAlertDisabled(true);
    try {
      const res = await fetch('/api/alert', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          event_type:   ev.event_type,
          timestamp:    ev.timestamp,
          confidence:   ev.confidence,
          description:  ev.description,
          snapshot_url: ev.snapshot_url,
        }),
      });
      if (res.ok) setAlertStatus({ msg: 'Alert sent!', ok: true });
      else        setAlertStatus({ msg: 'Failed to send', ok: false });
    } catch {
      setAlertStatus({ msg: 'Network error', ok: false });
    }
  }

  if (!ev) return null;

  const pct = Math.round((ev.confidence ?? 0) * 100);

  return (
    <div
      className="event-modal"
      onClick={(e) => { if (e.target === e.currentTarget) closeModal(); }}
    >
      <div className="event-modal-inner">
        <button className="event-modal-close" onClick={closeModal} aria-label="Close">&times;</button>
        <div className="event-modal-body">
          <div className="event-modal-img-wrap">
            {previewActive
              ? <canvas ref={previewCanvasRef} style={{ maxWidth: '100%', maxHeight: '78vh', display: 'block', borderRadius: '6px' }} />
              : ev.snapshot_url
                ? <ZoomableImage src={ev.snapshot_url} />
                : <span style={{ color: 'var(--muted)' }}>No snapshot</span>
            }
          </div>
          <div className="event-modal-info">
            <div className="event-modal-header">
              <span className={`event-modal-badge ${ev.event_type}`}>
                {TYPE_LABELS[ev.event_type] ?? ev.event_type}
              </span>
              <span className="event-modal-conf">{pct}% confidence</span>
            </div>
            <div className="event-modal-timestamp">
              {new Date(ev.timestamp * 1000).toLocaleString()}
            </div>
            <div className="event-modal-section-label">LLM Response</div>
            <div className="event-modal-desc">{ev.description || 'No description'}</div>
            <div className="event-modal-nvr-row">
              {!previewActive
                ? <button className="btn-nvr-play" onClick={() => setPreviewActive(true)}>▶ Play on NVR</button>
                : <button className="btn-nvr-live" onClick={() => setPreviewActive(false)}>⏹ Stop Preview</button>
              }
            </div>
            {(() => {
              const ageMin = (Date.now() / 1000 - ev.timestamp) / 60;
              if (ageMin < 30)
                return <div className="nvr-warning">⚠ Recent recording — NVR may not have sealed this file yet</div>;
              if ((ev.camera_id ?? 0) === 0 && ageMin < 120)
                return <div className="nvr-warning">⚠ Live-camera event — check cam&nbsp;0 panel after seeking</div>;
              return null;
            })()}
            {!demo && (
              <>
                <button className="btn-alert" onClick={sendAlert} disabled={alertDisabled}>
                  &#128241; Send Alert
                </button>
                {alertStatus && (
                  <div className={`event-modal-status ${alertStatus.ok ? 'ok' : 'err'}`}>
                    {alertStatus.msg}
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
