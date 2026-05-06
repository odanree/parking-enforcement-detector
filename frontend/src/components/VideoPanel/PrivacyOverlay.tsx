import { useEffect, useRef, type RefObject } from 'react';
import type { PrivacyRegion } from '../../types';

const W = 1280;
const H = 720;

function toFrame(e: MouseEvent, canvas: HTMLCanvasElement): [number, number] {
  const r = canvas.getBoundingClientRect();
  return [
    Math.round((e.clientX - r.left) * (W / r.width)),
    Math.round((e.clientY - r.top)  * (H / r.height)),
  ];
}

function drawPrivacy(
  canvas: HTMLCanvasElement,
  regions: PrivacyRegion[],
  editing: boolean,
  privacyMode: boolean,
  drag: { x0: number; y0: number; x1: number; y1: number } | null,
) {
  const ctx = canvas.getContext('2d')!;
  ctx.clearRect(0, 0, W, H);

  for (const [x1, y1, x2, y2] of regions) {
    if (editing) {
      ctx.fillStyle   = 'rgba(179,136,255,0.18)';
      ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
      ctx.strokeStyle = '#b388ff'; ctx.lineWidth = 2;
      ctx.setLineDash([5, 3]); ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      ctx.setLineDash([]);
    } else if (privacyMode) {
      ctx.fillStyle = '#000';
      ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
    }
  }

  if (drag) {
    const { x0, y0, x1, y1 } = drag;
    ctx.fillStyle   = 'rgba(179,136,255,0.25)';
    ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
    ctx.strokeStyle = '#b388ff'; ctx.lineWidth = 2;
    ctx.setLineDash([4, 3]); ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
    ctx.setLineDash([]);
  }
}

interface Props {
  canvasRef:   RefObject<HTMLCanvasElement | null>;
  privacyMode: boolean;
  editing:     boolean;
  onDone:      () => void;
}

export function PrivacyOverlay({ canvasRef, privacyMode, editing, onDone }: Props) {
  const regionsRef = useRef<PrivacyRegion[]>([]);
  const draftRef   = useRef<PrivacyRegion[]>([]);
  const dragRef    = useRef<{ x0: number; y0: number; x1: number; y1: number } | null>(null);

  function redraw() {
    const c = canvasRef.current;
    if (c) drawPrivacy(c, editing ? draftRef.current : regionsRef.current, editing, privacyMode, dragRef.current);
  }

  // Load saved regions on mount
  useEffect(() => {
    fetch('/api/privacy/regions')
      .then((r) => r.json())
      .then((d) => { regionsRef.current = d.regions ?? []; redraw(); })
      .catch(() => {});
  }, [canvasRef]);

  // Initialize when entering edit mode
  useEffect(() => {
    if (!editing) return;
    draftRef.current = regionsRef.current.map((r) => [...r] as PrivacyRegion);
    dragRef.current  = null;
  }, [editing]);

  // Re-draw when editing or privacyMode changes
  useEffect(() => { redraw(); });

  // Canvas events — only while editing
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !editing) return;

    function onMouseDown(e: MouseEvent) {
      if (e.button !== 0) return;
      const [x, y] = toFrame(e, canvas!);
      dragRef.current = { x0: x, y0: y, x1: x, y1: y };
    }

    function onMouseMove(e: MouseEvent) {
      if (!dragRef.current) return;
      const [x, y] = toFrame(e, canvas!);
      dragRef.current = { ...dragRef.current, x1: x, y1: y };
      drawPrivacy(canvas!, draftRef.current, true, privacyMode, dragRef.current);
    }

    function onMouseUp() {
      if (!dragRef.current) return;
      const { x0, y0, x1, y1 } = dragRef.current;
      const rx1 = Math.min(x0, x1), ry1 = Math.min(y0, y1);
      const rx2 = Math.max(x0, x1), ry2 = Math.max(y0, y1);
      if (rx2 - rx1 > 8 && ry2 - ry1 > 8)
        draftRef.current = [...draftRef.current, [rx1, ry1, rx2, ry2]];
      dragRef.current = null;
      drawPrivacy(canvas!, draftRef.current, true, privacyMode, null);
    }

    function onContextMenu(e: MouseEvent) {
      e.preventDefault();
      const [cx, cy] = toFrame(e, canvas!);
      draftRef.current = draftRef.current.filter(
        ([x1, y1, x2, y2]) => !(cx >= x1 && cx <= x2 && cy >= y1 && cy <= y2),
      );
      drawPrivacy(canvas!, draftRef.current, true, privacyMode, null);
    }

    canvas.classList.add('editing');
    canvas.addEventListener('mousedown',   onMouseDown);
    canvas.addEventListener('mousemove',   onMouseMove);
    canvas.addEventListener('mouseup',     onMouseUp);
    canvas.addEventListener('contextmenu', onContextMenu);

    return () => {
      canvas.classList.remove('editing');
      canvas.style.cursor = '';
      canvas.removeEventListener('mousedown',   onMouseDown);
      canvas.removeEventListener('mousemove',   onMouseMove);
      canvas.removeEventListener('mouseup',     onMouseUp);
      canvas.removeEventListener('contextmenu', onContextMenu);
    };
  }, [canvasRef, editing, privacyMode]);

  function cancelEdit() {
    dragRef.current = null;
    onDone();
  }

  async function saveEdit() {
    try {
      await fetch('/api/privacy/regions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ regions: draftRef.current }),
      });
      regionsRef.current = draftRef.current;
    } catch { /* ignore */ }
    cancelEdit();
  }

  if (!editing) return null;

  return (
    <div className="privacy-toolbar">
      <span className="zone-hint">&#9632; Drag to draw a redaction box &nbsp;·&nbsp; Right-click a box to delete it</span>
      <div className="zone-actions">
        <button className="btn-reset" onClick={() => { draftRef.current = []; redraw(); }}>Clear All</button>
        <button className="btn-save"   onClick={saveEdit}>Save</button>
        <button className="btn-cancel" onClick={cancelEdit}>Cancel</button>
      </div>
    </div>
  );
}
