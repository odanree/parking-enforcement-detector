import { useEffect, useRef, type RefObject } from 'react';
import type { Polygon } from '../../types';

const W = 1280;
const H = 720;
const HIT_R      = 18;
const EDGE_HIT_R = 14;

function toFrame(e: MouseEvent, canvas: HTMLCanvasElement): [number, number] {
  const r = canvas.getBoundingClientRect();
  return [
    Math.round((e.clientX - r.left) * (W / r.width)),
    Math.round((e.clientY - r.top)  * (H / r.height)),
  ];
}

function hitVertex(points: Polygon, cx: number, cy: number): number {
  return points.findIndex(([px, py]) => Math.hypot(px - cx, py - cy) < HIT_R);
}

function hitEdge(points: Polygon, cx: number, cy: number): number {
  const n = points.length;
  for (let i = 0; i < n; i++) {
    const [ax, ay] = points[i];
    const [bx, by] = points[(i + 1) % n];
    if (Math.hypot((ax + bx) / 2 - cx, (ay + by) / 2 - cy) < EDGE_HIT_R) return i;
  }
  return -1;
}

function drawZone(canvas: HTMLCanvasElement, points: Polygon, editing: boolean, dragIdx: number, dragEdge: number) {
  const ctx = canvas.getContext('2d')!;
  ctx.clearRect(0, 0, W, H);
  if (!points.length) return;

  ctx.beginPath();
  ctx.moveTo(points[0][0], points[0][1]);
  for (let i = 1; i < points.length; i++) ctx.lineTo(points[i][0], points[i][1]);
  ctx.closePath();
  ctx.fillStyle   = editing ? 'rgba(0,230,118,0.10)' : 'rgba(0,230,118,0.06)';
  ctx.fill();
  ctx.strokeStyle = editing ? '#00e676' : '#00e676aa';
  ctx.lineWidth   = editing ? 2 : 1.5;
  ctx.setLineDash(editing ? [] : [6, 4]);
  ctx.stroke();
  ctx.setLineDash([]);

  const xs = points.map((p) => p[0]);
  const ys = points.map((p) => p[1]);
  ctx.font      = 'bold 11px Inter, system-ui, sans-serif';
  ctx.fillStyle = editing ? '#00e676' : '#00e676aa';
  ctx.fillText('STREET ZONE', Math.min(...xs), Math.max(Math.min(...ys) - 6, 12));

  if (!editing) return;
  const n = points.length;

  for (let i = 0; i < n; i++) {
    const [ax, ay] = points[i];
    const [bx, by] = points[(i + 1) % n];
    const mx = (ax + bx) / 2, my = (ay + by) / 2;
    const s = 14;
    ctx.beginPath();
    ctx.moveTo(mx, my - s); ctx.lineTo(mx + s, my);
    ctx.lineTo(mx, my + s); ctx.lineTo(mx - s, my);
    ctx.closePath();
    ctx.fillStyle   = i === dragEdge ? '#ffffff' : '#00b0ff';
    ctx.strokeStyle = '#000'; ctx.lineWidth = 2;
    ctx.fill(); ctx.stroke();
  }

  points.forEach(([px, py], i) => {
    ctx.beginPath();
    ctx.arc(px, py, 7, 0, Math.PI * 2);
    ctx.fillStyle   = i === dragIdx ? '#ffffff' : '#00e676';
    ctx.strokeStyle = '#000'; ctx.lineWidth = 1.5;
    ctx.fill(); ctx.stroke();
  });
}

interface Props {
  canvasRef: RefObject<HTMLCanvasElement | null>;
  editing:   boolean;
  onDone:    () => void;
}

export function ZoneOverlay({ canvasRef, editing, onDone }: Props) {
  const pointsRef  = useRef<Polygon>([]);
  const savedRef   = useRef<Polygon>([]);
  const historyRef = useRef<Polygon[]>([]);
  const dragIdx    = useRef(-1);
  const dragEdge   = useRef(-1);
  const dragPrev   = useRef<[number, number] | null>(null);
  const didDrag    = useRef(false);
  const undoBtnRef = useRef<HTMLButtonElement>(null);

  function redraw() {
    const c = canvasRef.current;
    if (c) drawZone(c, pointsRef.current, editing, dragIdx.current, dragEdge.current);
  }

  function pushHistory() {
    historyRef.current.push(pointsRef.current.map((p) => [...p] as [number, number]));
    if (undoBtnRef.current) undoBtnRef.current.disabled = false;
  }

  // Load zone on mount
  useEffect(() => {
    fetch('/api/zone')
      .then((r) => r.json())
      .then((d) => {
        pointsRef.current = d.polygon ?? [];
        const c = canvasRef.current;
        if (c) drawZone(c, pointsRef.current, false, -1, -1);
      })
      .catch(() => {});
  }, [canvasRef]);

  // Initialize when entering edit mode
  useEffect(() => {
    if (!editing) return;
    savedRef.current   = pointsRef.current.map((p) => [...p] as [number, number]);
    historyRef.current = [];
    if (undoBtnRef.current) undoBtnRef.current.disabled = true;
  }, [editing]);

  // Re-draw whenever editing toggles
  useEffect(() => { redraw(); });

  // Keyboard undo
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (!editing) return;
      if ((e.key === 'z' || e.key === 'Z') && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        if (!historyRef.current.length) return;
        pointsRef.current = historyRef.current.pop()!;
        if (undoBtnRef.current) undoBtnRef.current.disabled = historyRef.current.length === 0;
        redraw();
      }
    }
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [editing]);

  // Canvas mouse events — only active while editing
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !editing) return;

    function onClick(e: MouseEvent) {
      if (didDrag.current) return;
      const [cx, cy] = toFrame(e, canvas!);
      if (hitVertex(pointsRef.current, cx, cy) !== -1) return;
      if (hitEdge(pointsRef.current, cx, cy) !== -1) return;
      pushHistory();
      pointsRef.current = [...pointsRef.current, [cx, cy]];
      drawZone(canvas!, pointsRef.current, true, -1, -1);
    }

    function onContextMenu(e: MouseEvent) {
      e.preventDefault();
      const [cx, cy] = toFrame(e, canvas!);
      const idx = hitVertex(pointsRef.current, cx, cy);
      if (idx !== -1) {
        pushHistory();
        pointsRef.current = pointsRef.current.filter((_, i) => i !== idx);
        drawZone(canvas!, pointsRef.current, true, dragIdx.current, dragEdge.current);
      }
    }

    function onMouseDown(e: MouseEvent) {
      if (e.button !== 0) return;
      didDrag.current = false;
      const [cx, cy] = toFrame(e, canvas!);
      dragIdx.current = hitVertex(pointsRef.current, cx, cy);
      if (dragIdx.current !== -1) { dragPrev.current = [cx, cy]; canvas!.style.cursor = 'grabbing'; return; }
      dragEdge.current = hitEdge(pointsRef.current, cx, cy);
      if (dragEdge.current !== -1) { dragPrev.current = [cx, cy]; canvas!.style.cursor = 'grabbing'; pushHistory(); }
    }

    function onMouseMove(e: MouseEvent) {
      const [cx, cy] = toFrame(e, canvas!);
      if (dragIdx.current !== -1) {
        pointsRef.current[dragIdx.current] = [cx, cy];
        drawZone(canvas!, pointsRef.current, true, dragIdx.current, dragEdge.current);
        dragPrev.current = [cx, cy]; didDrag.current = true;
      } else if (dragEdge.current !== -1 && dragPrev.current) {
        const [px, py] = dragPrev.current;
        const dx = cx - px, dy = cy - py;
        const n = pointsRef.current.length;
        const j = (dragEdge.current + 1) % n;
        pointsRef.current[dragEdge.current] = [pointsRef.current[dragEdge.current][0] + dx, pointsRef.current[dragEdge.current][1] + dy];
        pointsRef.current[j]                = [pointsRef.current[j][0]                + dx, pointsRef.current[j][1]                + dy];
        drawZone(canvas!, pointsRef.current, true, dragIdx.current, dragEdge.current);
        dragPrev.current = [cx, cy]; didDrag.current = true;
      } else {
        if (hitVertex(pointsRef.current, cx, cy) !== -1)     canvas!.style.cursor = 'grab';
        else if (hitEdge(pointsRef.current, cx, cy) !== -1) canvas!.style.cursor = 'move';
        else                                                  canvas!.style.cursor = 'crosshair';
      }
    }

    function onMouseUp() {
      dragIdx.current = -1; dragEdge.current = -1; dragPrev.current = null;
      canvas!.style.cursor = 'crosshair';
    }

    function onMouseLeave() {
      dragIdx.current = -1; dragEdge.current = -1; dragPrev.current = null;
      canvas!.style.cursor = '';
    }

    canvas.classList.add('editing');
    canvas.addEventListener('click',       onClick);
    canvas.addEventListener('contextmenu', onContextMenu);
    canvas.addEventListener('mousedown',   onMouseDown);
    canvas.addEventListener('mousemove',   onMouseMove);
    canvas.addEventListener('mouseup',     onMouseUp);
    canvas.addEventListener('mouseleave',  onMouseLeave);

    return () => {
      canvas.classList.remove('editing');
      canvas.style.cursor = '';
      canvas.removeEventListener('click',       onClick);
      canvas.removeEventListener('contextmenu', onContextMenu);
      canvas.removeEventListener('mousedown',   onMouseDown);
      canvas.removeEventListener('mousemove',   onMouseMove);
      canvas.removeEventListener('mouseup',     onMouseUp);
      canvas.removeEventListener('mouseleave',  onMouseLeave);
    };
  }, [canvasRef, editing]);

  function cancelEdit() {
    pointsRef.current  = savedRef.current.map((p) => [...p] as [number, number]);
    historyRef.current = [];
    onDone();
  }

  async function saveEdit() {
    if (pointsRef.current.length < 3) { alert('Zone needs at least 3 points.'); return; }
    try {
      const res = await fetch('/api/zone', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ polygon: pointsRef.current }),
      });
      if (!res.ok) throw new Error(await res.text());
      onDone();
    } catch (err: unknown) {
      alert('Save failed: ' + (err instanceof Error ? err.message : String(err)));
    }
  }

  if (!editing) return null;

  return (
    <div className="zone-toolbar">
      <span className="zone-hint">
        Click to add &nbsp;·&nbsp; Drag vertex or edge &nbsp;·&nbsp; Right-click to remove &nbsp;·&nbsp; Ctrl-Z to undo
      </span>
      <div className="zone-actions">
        <button ref={undoBtnRef} className="btn-undo" disabled onClick={() => {
          if (!historyRef.current.length) return;
          pointsRef.current = historyRef.current.pop()!;
          if (undoBtnRef.current) undoBtnRef.current.disabled = historyRef.current.length === 0;
          redraw();
        }}>Undo</button>
        <button className="btn-reset" onClick={() => {
          pushHistory();
          pointsRef.current = [[0,0],[W,0],[W,H],[0,H]];
          redraw();
        }}>Reset</button>
        <button className="btn-save"   onClick={saveEdit}>Save</button>
        <button className="btn-cancel" onClick={cancelEdit}>Cancel</button>
      </div>
    </div>
  );
}
