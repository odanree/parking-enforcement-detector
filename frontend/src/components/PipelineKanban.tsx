import { useEffect, useMemo, useRef, useState, useCallback } from 'react';
import { useAppStore } from '../store';
import type { PipelineStage, RagNeighbor, RagStage } from '../types';

type TimeFilter  = '5m' | '1h' | '12h' | '24h' | 'all';
type LabelFilter = 'unlabeled' | 'labeled' | 'all';
type CamFilter   = 0 | 1 | 'all';

interface ModelEvalEntry {
  model: string;
  backend: string;
  detected: number;
  confidence: number;
  description: string;
  ts: number;
}

interface TraceCard {
  id: string;
  source: 'historical' | 'live';
  outcome: 'detected' | 'rejected' | 'people_alert' | 'chalking';
  timestamp: number;
  camera_id: number;
  confidence: number;
  description: string;
  thumbnail: string | null;   // base64 for live rejected items
  thumb_url: string | null;   // /dataset/<file> for historical items
  hires_url: string | null;   // /dataset/<file>_hires.jpg when HIRES_SNAPSHOT=true
  rag_neighbors: RagNeighbor[];
  track_id?: number | null;
  phash?: string | null;
  yolo?: { confidence: number; track_id?: number; bbox?: number[] } | null;
  first_pass: PipelineStage | null;
  rag: RagStage | null;
  confirm: PipelineStage | null;
  label?: string;
  person_type?: string;
  capture_source?: string;
  rejection_reason?: string;
  model_evals?: ModelEvalEntry[];
  model_primary?: string;
  model_confirm?: string;
}

function pct(v: number) { return `${Math.round(v * 100)}%`; }

interface CardGroup { rep: TraceCard; count: number; all: TraceCard[]; }

const HASH_THRESHOLD = 10; // max Hamming distance (out of 64 bits) to be "same scene"

function hammingDist(a: string, b: string): number {
  let d = 0;
  for (let i = 0; i < Math.min(a.length, b.length); i++) {
    const xor = parseInt(a[i], 16) ^ parseInt(b[i], 16);
    d += xor.toString(2).split('1').length - 1;
  }
  return d;
}

function groupCards(cards: TraceCard[]): CardGroup[] {
  // Greedy hash-similarity clustering per camera.
  // Cards without a phash each get their own group.
  const groups: { camId: number; hash: string; cards: TraceCard[] }[] = [];

  for (const card of cards) {
    const h = card.phash;
    if (h) {
      const existing = groups.find(
        (g) => g.camId === card.camera_id && hammingDist(g.hash, h) <= HASH_THRESHOLD,
      );
      if (existing) { existing.cards.push(card); continue; }
      groups.push({ camId: card.camera_id, hash: h, cards: [card] });
    } else {
      groups.push({ camId: card.camera_id, hash: '', cards: [card] });
    }
  }

  return groups.map(({ cards: all }) => {
    const rep = all.reduce((best, c) => c.confidence > best.confidence ? c : best, all[0]);
    return { rep, count: all.length, all };
  });
}

const PERSON_TYPES: { key: string; label: string; emoji: string }[] = [
  { key: 'pedestrian',       label: 'Pedestrian',  emoji: '🚶' },
  { key: 'occupant',         label: 'Occupant',    emoji: '🚗' },
  { key: 'worker_landscape', label: 'Landscaper',  emoji: '🌿' },
  { key: 'worker_delivery',  label: 'Delivery',    emoji: '📦' },
  { key: 'chalker',          label: 'Chalker',     emoji: '✏️' },
];

function ConfBadge({ v, detected }: { v: number; detected: boolean }) {
  const cls = detected ? 'kanban-conf-pos' : 'kanban-conf-neg';
  return <span className={`kanban-conf ${cls}`}>{pct(v)}</span>;
}

function RagSignalBadge({ signal, fpClose, tpClose }: { signal: string; fpClose: number; tpClose: number }) {
  if (signal === 'none') return <span className="kanban-rag-none">no labeled matches</span>;
  if (signal === 'fp')   return <span className="kanban-rag-fp">{fpClose} FP match{fpClose !== 1 ? 'es' : ''}</span>;
  return <span className="kanban-rag-tp">{tpClose} TP match{tpClose !== 1 ? 'es' : ''}</span>;
}

function NeighborList({ neighbors }: { neighbors: RagNeighbor[] }) {
  const labeled = neighbors.filter((n) => n.label);
  if (!labeled.length) return null;
  return (
    <div className="kanban-neighbors">
      {labeled.slice(0, 3).map((n, i) => (
        <div key={i} className={`kanban-neighbor ${n.label === 'true_positive' ? 'nb-tp' : 'nb-fp'}`}>
          <span className="nb-badge">{n.label === 'true_positive' ? 'TP' : 'FP'}</span>
          <span className="nb-dist">{n.distance.toFixed(2)}</span>
          <span className="nb-desc">{n.description.slice(0, 60)}…</span>
        </div>
      ))}
    </div>
  );
}

function ZoomableImage({ src }: { src: string }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const zoomRef      = useRef({ scale: 1, tx: 0, ty: 0 });
  const dragRef      = useRef<{ x: number; y: number } | null>(null);
  const [xform, setXform] = useState({ scale: 1, tx: 0, ty: 0 });
  const [dims, setDims]   = useState<{ w: number; h: number } | null>(null);

  // Reset zoom and dimensions whenever the image changes (prev/next navigation)
  useEffect(() => {
    const z = { scale: 1, tx: 0, ty: 0 };
    zoomRef.current = z;
    setXform(z);
    setDims(null);
  }, [src]);

  // Wheel handler — must be non-passive to call preventDefault
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    function onWheel(e: WheelEvent) {
      e.preventDefault();
      const rect = el!.getBoundingClientRect();
      const cx = e.clientX - rect.left;
      const cy = e.clientY - rect.top;
      const { scale, tx, ty } = zoomRef.current;
      const factor   = e.deltaY < 0 ? 1.15 : 1 / 1.15;
      const newScale = Math.min(Math.max(scale * factor, 1), 10);
      if (newScale <= 1) {
        const z = { scale: 1, tx: 0, ty: 0 };
        zoomRef.current = z; setXform(z); return;
      }
      const ratio = newScale / scale;
      const z = { scale: newScale, tx: cx - (cx - tx) * ratio, ty: cy - (cy - ty) * ratio };
      zoomRef.current = z; setXform(z);
    }
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, []);

  const onPointerDown = useCallback((e: React.PointerEvent) => {
    if (zoomRef.current.scale <= 1) return;
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    dragRef.current = { x: e.clientX, y: e.clientY };
  }, []);

  const onPointerMove = useCallback((e: React.PointerEvent) => {
    if (!dragRef.current) return;
    const dx = e.clientX - dragRef.current.x;
    const dy = e.clientY - dragRef.current.y;
    dragRef.current = { x: e.clientX, y: e.clientY };
    const { scale, tx, ty } = zoomRef.current;
    const z = { scale, tx: tx + dx, ty: ty + dy };
    zoomRef.current = z; setXform(z);
  }, []);

  const onPointerUp = useCallback(() => { dragRef.current = null; }, []);

  const onDoubleClick = useCallback(() => {
    const z = { scale: 1, tx: 0, ty: 0 };
    zoomRef.current = z; setXform(z);
  }, []);

  return (
    <div
      ref={containerRef}
      className="kanban-zoom-wrap"
      style={{ cursor: xform.scale > 1 ? 'grab' : 'zoom-in' }}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onDoubleClick={onDoubleClick}
    >
      <img
        src={src}
        alt=""
        draggable={false}
        className="kanban-zoom-img"
        style={{ transformOrigin: '0 0', transform: `translate(${xform.tx}px,${xform.ty}px) scale(${xform.scale})` }}
        onLoad={(e) => {
          const img = e.currentTarget;
          setDims({ w: img.naturalWidth, h: img.naturalHeight });
        }}
      />
      {dims && (
        <span className="img-resolution-badge">{dims.w}×{dims.h}</span>
      )}
      {xform.scale > 1 && (
        <span className="kanban-zoom-hint">double-click to reset</span>
      )}
    </div>
  );
}

function thumbSrc(card: TraceCard): string | null {
  if (card.thumb_url)  return card.thumb_url;
  if (card.thumbnail)  return `data:image/jpeg;base64,${card.thumbnail}`;
  return null;
}

function hiresSrc(card: TraceCard): string | null {
  return card.hires_url ?? thumbSrc(card);
}

function KanbanCard({ card, count, onClick }: { card: TraceCard; count?: number; onClick: () => void }) {
  const timeStr = new Date(card.timestamp * 1000).toLocaleTimeString([], {
    hour: '2-digit', minute: '2-digit', second: '2-digit',
  });
  const src = thumbSrc(card);
  const isLive = card.id.startsWith('live-');

  return (
    <div className={`kanban-card ${card.outcome}`} onClick={onClick}>
      {src && (
        <div className="kanban-thumb-wrap">
          <img src={src} alt="" className="kanban-thumb" loading="lazy" />
          {count != null && count > 1 && (
            <span className="kanban-group-badge">×{count}</span>
          )}
        </div>
      )}
      <div className="kanban-card-meta">
        <div className="kanban-card-top">
          <span className="kanban-cam">Cam {card.camera_id}</span>
          {isLive && <span className="kanban-live-dot" title="Pending database index" />}
          <span className="kanban-time">{timeStr}</span>
        </div>
        <div className="kanban-desc">{card.description.slice(0, 80)}{card.description.length > 80 ? '…' : ''}</div>
      </div>
    </div>
  );
}

function CardDetail({ card }: { card: TraceCard }) {
  return (
    <div className="kanban-pipeline-flow">
      {card.rejection_reason === 'out_of_hours' && (
        <div className="kanban-stage-block stage-time">
          <div className="kanban-stage-label">⏰ Time-Window Gate</div>
          <div className="kanban-stage-result"><span className="stage-time-text">✗ Off hours</span></div>
          <div className="kanban-stage-desc">{card.description}</div>
        </div>
      )}
      {card.yolo && (
        <div className="kanban-stage-block stage-yolo">
          <div className="kanban-stage-label">YOLO Detection</div>
          <div className="kanban-stage-result">
            <span className="stage-pos-text">✓ person detected</span>
            <ConfBadge v={card.yolo.confidence} detected />
            {card.yolo.track_id != null && (
              <span className="kanban-yolo-meta">#{card.yolo.track_id}</span>
            )}
          </div>
          {card.yolo.bbox && (
            <div className="kanban-stage-desc">bbox [{card.yolo.bbox.join(', ')}]</div>
          )}
        </div>
      )}
      {card.first_pass && (
        <div className={`kanban-stage-block ${card.first_pass.detected ? 'stage-pos' : 'stage-neg'}`}>
          <div className="kanban-stage-label">1st Pass — People Detection ({card.first_pass.backend})</div>
          <div className="kanban-stage-result">
            {card.first_pass.detected ? '✓ detected' : '✗ not detected'}
            <ConfBadge v={card.first_pass.confidence} detected={card.first_pass.detected} />
          </div>
          <div className="kanban-stage-desc">{card.first_pass.description}</div>
        </div>
      )}
      {card.rag && (
        <div className="kanban-stage-block stage-rag">
          <div className="kanban-stage-label">
            RAG Retrieval
            {card.rag.skipped_confirm && <span className="kanban-skip-badge">⚡ skipped confirm</span>}
          </div>
          <div className="kanban-stage-result">
            <RagSignalBadge signal={card.rag.signal} fpClose={card.rag.fp_close} tpClose={card.rag.tp_close} />
            <span className="kanban-rag-total">{card.rag.labeled_count} labeled neighbors</span>
          </div>
          <NeighborList neighbors={card.rag.neighbors} />
        </div>
      )}
      {card.confirm ? (
        <div className={`kanban-stage-block ${card.confirm.detected ? 'stage-pos' : 'stage-neg'}`}>
          <div className="kanban-stage-label">Confirm ({card.confirm.backend})</div>
          <div className="kanban-stage-result">
            {card.confirm.detected ? '✓ detected' : '✗ not detected'}
            <ConfBadge v={card.confirm.confidence} detected={card.confirm.detected} />
          </div>
          <div className="kanban-stage-desc">{card.confirm.description}</div>
        </div>
      ) : (
        card.first_pass && (
          <div className="kanban-stage-block stage-skipped">
            <div className="kanban-stage-label">Confirm</div>
            <div className="kanban-stage-result">skipped</div>
          </div>
        )
      )}
      <div className={`kanban-stage-block stage-outcome ${card.outcome}`}>
        <div className="kanban-stage-label">Outcome</div>
        <div className="kanban-stage-result">
          {card.outcome === 'chalking'
            ? <><span className="stage-pos-text">✓ Chalking alert</span> <ConfBadge v={card.confidence} detected /></>
            : card.outcome === 'people_alert' || card.outcome === 'detected'
            ? <><span className="stage-pos-text">✓ {COLUMNS.find(c => c.key === cardColumn(card))?.label ?? 'People Alert'}</span> <ConfBadge v={card.confidence} detected /></>
            : <><span className="stage-neg-text">✗ Rejected</span> <ConfBadge v={card.confidence} detected={false} /></>
          }
        </div>
      </div>
      {card.model_evals && card.model_evals.length > 0 && (
        <div className="kanban-stage-block stage-model-evals">
          <div className="kanban-stage-label">Model Comparison</div>
          <table className="model-eval-table">
            <thead>
              <tr><th>Model</th><th>Result</th><th>Conf</th></tr>
            </thead>
            <tbody>
              {card.model_evals.map((e) => (
                <tr key={e.model} className={e.detected ? 'eval-pos' : 'eval-neg'}>
                  <td className="eval-model-name">{e.model}</td>
                  <td>{e.detected ? '✓' : '✗'}</td>
                  <td>{pct(e.confidence)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function PersonTypeLabeler({ card }: { card: TraceCard }) {
  const [current, setCurrent] = useState(card.person_type ?? '');
  const [saving, setSaving] = useState(false);

  async function applyLabel(pt: string) {
    const next = current === pt ? '' : pt; // toggle off if already set
    setSaving(true);
    try {
      await fetch(`/api/dataset/${card.id}/person-type`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ person_type: next }),
      });
      setCurrent(next);
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="person-type-labeler">
      <span className="person-type-label-title">Person Type</span>
      <div className="person-type-buttons">
        {PERSON_TYPES.map((pt) => (
          <button
            key={pt.key}
            className={`btn-person-type btn-pt-${pt.key}${current === pt.key ? ' active' : ''}`}
            onClick={() => applyLabel(pt.key)}
            disabled={saving}
            title={pt.label}
          >
            {pt.emoji} {pt.label}
          </button>
        ))}
        {current && (
          <button className="btn-person-type btn-pt-clear" onClick={() => applyLabel('')} disabled={saving}>
            ✕ Clear
          </button>
        )}
      </div>
    </div>
  );
}

function TpFpLabeler({ card, onLabeled }: { card: TraceCard; onLabeled?: (label: string) => void }) {
  const [current, setCurrent] = useState(card.label ?? '');
  const [saving, setSaving] = useState(false);
  const isLive = card.id.startsWith('live-');

  async function applyLabel(next: string) {
    if (isLive) return;
    const val = current === next ? '' : next;
    setSaving(true);
    try {
      await fetch(`/api/dataset/${card.id}/label`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ label: val }),
      });
      setCurrent(val);
      onLabeled?.(val);
    } finally {
      setSaving(false);
    }
  }

  const LABELS = [
    { val: 'true_positive',  short: 'TP', cls: 'tp' },
    { val: 'false_positive', short: 'FP', cls: 'fp' },
    { val: 'true_negative',  short: 'TN', cls: 'tn' },
    { val: 'false_negative', short: 'FN', cls: 'fn' },
  ];

  return (
    <div className="kanban-tpfp-row">
      {LABELS.map(({ val, short, cls }) => (
        <button
          key={val}
          className={`kanban-tpfp-btn ${cls}${current === val ? ' active' : ''}`}
          onClick={() => applyLabel(val)}
          disabled={saving || isLive}
          title={isLive ? 'Available once event is saved to dataset' : val.replace('_', ' ')}
        >{short}</button>
      ))}
      {isLive && <span className="kanban-tpfp-pending">Pending index…</span>}
    </div>
  );
}

function StageDetailModal({ cards: allCards, startIndex, onClose, onLabeled }: { cards: TraceCard[]; startIndex: number; onClose: () => void; onLabeled?: (label: string, cardId: string) => void }) {
  const [idx, setIdx] = useState(startIndex);
  const card = allCards[idx];
  const timeStr = new Date(card.timestamp * 1000).toLocaleString();
  const total = allCards.length;
  const captureTag = card.capture_source === 'zone_pedestrian' ? '🚶 Zone pedestrian' : null;

  return (
    <div className="kanban-modal-overlay" onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <div className="kanban-modal">
        <button className="kanban-modal-close" onClick={onClose}>&times;</button>
        <div className="kanban-modal-header">
          <span className={`kanban-outcome-badge ${card.outcome}`}>
            {card.outcome === 'chalking'
              ? '✓ Chalking'
              : card.outcome === 'people_alert' || card.outcome === 'detected'
              ? `✓ ${COLUMNS.find(c => c.key === cardColumn(card))?.label ?? 'People Alert'}`
              : `✗ ${COLUMNS.find(c => c.key === cardColumn(card))?.label ?? 'Rejected'}`}
          </span>
          {captureTag && <span className="kanban-capture-tag">{captureTag}</span>}
          <span className="kanban-modal-time">Cam {card.camera_id} · {timeStr}</span>
        </div>

        {total > 1 && (
          <div className="kanban-modal-nav-row">
            <button className="kanban-modal-nav-btn" onClick={() => setIdx((i) => (i - 1 + total) % total)}>&#8592;</button>
            <span className="kanban-modal-nav-count">{idx + 1} / {total}</span>
            <button className="kanban-modal-nav-btn" onClick={() => setIdx((i) => (i + 1) % total)}>&#8594;</button>
          </div>
        )}

        {hiresSrc(card) && (
          <div style={{ position: 'relative' }}>
            <ZoomableImage src={hiresSrc(card)!} />
            {card.hires_url && (
              <span className="hires-badge">HI-RES</span>
            )}
          </div>
        )}

        <TpFpLabeler card={card} onLabeled={onLabeled ? (label) => onLabeled(label, card.id) : undefined} />

        <PersonTypeLabeler card={card} />

        <CardDetail card={card} />
      </div>
    </div>
  );
}

const COLUMNS: { key: string; label: string; desc: string }[] = [
  { key: 'no_person',    label: 'No Person ✗',    desc: 'Rejected — no person near vehicle'       },
  { key: 'vlm_error',    label: 'VLM Error ⚠',    desc: 'Timeout / connection failure'            },
  { key: 'not_chalking', label: 'False Detect ✗', desc: 'Model disagreement — not a real detection' },
  { key: 'pedestrian',   label: 'Pedestrian',     desc: 'Person on sidewalk / walking by'         },
  { key: 'occupant',     label: 'Occupant',       desc: 'Person exiting or entering vehicle'      },
  { key: 'worker',       label: 'Worker',         desc: 'Delivery driver, landscaper, or maintenance' },
  { key: 'chalking',     label: 'Chalking ✓',     desc: 'Confirmed chalking activity'             },
];

const _NO_PERSON_RE   = /no person|not visible|nobody|no one|not anyone|not present|no individual|no human|cannot see|can't see/i;
const _PEDESTRIAN_RE  = /sidewalk|pedestrian|walking|walk by|passing|foot traffic|passer|stroll/i;

function cardColumn(card: TraceCard): string {
  if (card.rejection_reason === 'suppressed')   return 'no_person';
  if (card.rejection_reason === 'vlm_error')    return 'vlm_error';

  const desc = card.description || card.first_pass?.description || '';
  const detected = card.outcome === 'people_alert' || card.outcome === 'detected';

  // Confirmed chalking always wins.
  if (detected && card.confirm?.detected) return 'chalking';

  // Route by the classifier's person_type when known. This covers classifier
  // short-circuits (pedestrian/occupant/delivery skip the VLM and are "rejected"
  // by design but ARE correctly-classified people, not false detections).
  switch (card.person_type) {
    case 'pedestrian':       return 'pedestrian';
    case 'occupant':         return 'occupant';
    case 'worker_delivery':  return 'worker';
    case 'worker_landscape': return 'worker';
    case 'chalker':          return detected ? 'chalking' : 'not_chalking';
  }

  // No classifier label → fall back to outcome/description heuristics.
  if (detected) return _PEDESTRIAN_RE.test(desc) ? 'pedestrian' : 'occupant';
  if (card.outcome === 'rejected') return _NO_PERSON_RE.test(desc) ? 'no_person' : 'not_chalking';
  return 'no_person';
}

export function PipelineKanban() {
  const kanbanOpen    = useAppStore((s) => s.kanbanOpen);
  const setKanbanOpen = useAppStore((s) => s.setKanbanOpen);

  const [cards, setCards]       = useState<TraceCard[]>([]);
  const [loading, setLoading]   = useState(false);
  const [selected, setSelected] = useState<{ cards: TraceCard[]; index: number } | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const [timeFilter,  setTimeFilter]  = useState<TimeFilter>('1h');
  const [labelFilter, setLabelFilter] = useState<LabelFilter>('unlabeled');
  const [camFilter,   setCamFilter]   = useState<CamFilter>('all');

  const timeWindowSecs = useMemo(() => {
    if (timeFilter === 'all') return null;
    return timeFilter === '5m' ? 300 : timeFilter === '1h' ? 3600
         : timeFilter === '12h' ? 12 * 3600 : 24 * 3600;
  }, [timeFilter]);

  function buildUrl() {
    const p = new URLSearchParams({ limit: '500' });
    if (timeWindowSecs !== null) p.set('since', String(Math.floor(Date.now() / 1000) - timeWindowSecs));
    if (labelFilter === 'unlabeled') p.set('label', '');
    if (camFilter !== 'all') p.set('camera_id', String(camFilter));
    return `/api/pipeline/trace?${p}`;
  }

  function fetchCards() {
    fetch(buildUrl())
      .then((r) => r.json())
      .then((d) => setCards(d.cards ?? []))
      .catch(() => {});
  }

  useEffect(() => {
    if (!kanbanOpen) return;
    setLoading(true);
    fetch(buildUrl())
      .then((r) => r.json())
      .then((d) => setCards(d.cards ?? []))
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [kanbanOpen, timeWindowSecs, labelFilter, camFilter]);

  useEffect(() => {
    if (!kanbanOpen || !autoRefresh) {
      if (timerRef.current) clearInterval(timerRef.current);
      return;
    }
    timerRef.current = setInterval(fetchCards, 5000);
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, [kanbanOpen, autoRefresh, timeWindowSecs, labelFilter, camFilter]);

  // Keep the open modal in sync with the latest card fetch (e.g. hires_url backfill after restart).
  // Only refresh cards that belong to the same kanban column as the originally-selected group
  // so that a False Detect and an Occupant sharing the same timestamp don't merge in the modal.
  useEffect(() => {
    if (!selected) return;
    const selectedCol = cardColumn(selected.cards[0]);
    const keys = new Set(selected.cards.map(sc => `${sc.camera_id}:${Math.round(sc.timestamp)}`));
    const refreshed = cards.filter(c =>
      cardColumn(c) === selectedCol &&
      keys.has(`${c.camera_id}:${Math.round(c.timestamp)}`),
    );
    if (refreshed.length > 0) {
      setSelected(prev => prev ? { index: Math.min(prev.index, refreshed.length - 1), cards: refreshed } : prev);
    }
  }, [cards]);

  useEffect(() => {
    if (!kanbanOpen) return;
    function onKey(e: KeyboardEvent) { if (e.key === 'Escape') { setSelected(null); setKanbanOpen(false); } }
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [kanbanOpen, setKanbanOpen]);

  // Client-side: 'labeled' filter = only cards that have a label set
  const visibleCards = labelFilter === 'labeled'
    ? cards.filter((c) => c.label && c.label !== '')
    : cards;

  const oldestTs = cards.length > 0 ? Math.min(...cards.map(c => c.timestamp)) : null;
  const oldestLabel = oldestTs !== null ? (() => {
    const date = new Date(oldestTs * 1000).toLocaleString([], {
      month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
    });
    const diffSec = Math.max(0, Math.floor(Date.now() / 1000 - oldestTs));
    const d = Math.floor(diffSec / 86400);
    const h = Math.floor((diffSec % 86400) / 3600);
    const m = Math.floor((diffSec % 3600) / 60);
    const s = diffSec % 60;
    const rel = d > 0 ? `${d}d ${h}h ago` : h > 0 ? `${h}h ${m}m ago` : m > 0 ? `${m}m ${s}s ago` : `${s}s ago`;
    return `back to ${date} (${rel})`;
  })() : null;

  const detected = visibleCards.filter((c) => c.outcome === 'detected' || c.outcome === 'people_alert');
  const rejected  = visibleCards.filter((c) => c.outcome === 'rejected');

  const byColumn: Record<string, TraceCard[]> = {
    no_person: [], vlm_error: [], not_chalking: [],
    pedestrian: [], occupant: [], worker: [], chalking: [],
  };
  for (const card of visibleCards) {
    const col = cardColumn(card);
    (byColumn[col] ??= []).push(card);
  }

  if (!kanbanOpen) return null;

  return (
    <>
      {selected && (
        <StageDetailModal
          cards={selected.cards}
          startIndex={selected.index}
          onClose={() => setSelected(null)}
          onLabeled={(label, cardId) => {
            if (label && labelFilter === 'unlabeled') setLabelFilter('all');
            setCards(prev => prev.map(c => c.id === cardId ? { ...c, label } : c));
          }}
        />
      )}
      <div className="kanban-overlay" onClick={(e) => { if (e.target === e.currentTarget) setKanbanOpen(false); }}>
        <div className="kanban-panel">
          <div className="kanban-panel-header">
            <h2 className="kanban-title">Pipeline Kanban</h2>
            <div className="kanban-filters">
              {/* Time */}
              {(['5m','1h','12h','24h','all'] as TimeFilter[]).map((t) => (
                <button key={t} className={`kanban-filter-btn${timeFilter === t ? ' active' : ''}`}
                  onClick={() => setTimeFilter(t)}>
                  {t === '5m' ? '5 min' : t === '1h' ? '1 hr' : t === '12h' ? '12 hr' : t === '24h' ? '24 hr' : 'All time'}
                </button>
              ))}
              <span className="kanban-filter-sep" />
              {/* Label */}
              {(['unlabeled','labeled','all'] as LabelFilter[]).map((l) => (
                <button key={l} className={`kanban-filter-btn${labelFilter === l ? ' active' : ''}`}
                  onClick={() => setLabelFilter(l)}>
                  {l === 'unlabeled' ? 'Unlabeled' : l === 'labeled' ? 'Labeled' : 'Any label'}
                </button>
              ))}
              <span className="kanban-filter-sep" />
              {/* Camera */}
              {(['all', 0, 1] as CamFilter[]).map((c) => (
                <button key={String(c)} className={`kanban-filter-btn${camFilter === c ? ' active' : ''}`}
                  onClick={() => setCamFilter(c)}>
                  {c === 'all' ? 'All cams' : `Cam ${c}`}
                </button>
              ))}
            </div>
            <div className="kanban-header-right">
              <span className="kanban-stat kanban-stat-detected">{detected.length} detected</span>
              <span className="kanban-stat kanban-stat-rejected">{rejected.length} rejected</span>
              {oldestLabel && (
                <span className="kanban-stat kanban-stat-oldest" title={`Oldest of ${cards.length} cards`}>
                  back to {oldestLabel}
                </span>
              )}
              <label className="kanban-auto-label">
                <input type="checkbox" checked={autoRefresh} onChange={(e) => setAutoRefresh(e.target.checked)} />
                Live
              </label>
              <button className="btn-kanban-clear" onClick={async () => {
                if (!confirm('Delete all pipeline history? This cannot be undone.')) return;
                await fetch('/api/pipeline/history', { method: 'DELETE' });
                setCards([]);
              }}>Clear History</button>
              <button className="btn-debug-close" onClick={() => setKanbanOpen(false)}>&times;</button>
            </div>
          </div>

          {loading && <div className="kanban-loading">Loading…</div>}

          <div className="kanban-board">
            {COLUMNS.map((col) => {
              const colCards = byColumn[col.key] ?? [];
              return (
                <div key={col.key} className="kanban-col">
                  <div className="kanban-col-header">
                    <span className="kanban-col-title">{col.label}</span>
                    <span className="kanban-col-desc">{col.desc}</span>
                    <span className="kanban-col-count">{colCards.length}</span>
                  </div>
                  <div className="kanban-col-cards">
                    {colCards.length === 0
                      ? <div className="kanban-col-empty">—</div>
                      : groupCards(colCards).sort((a, b) => b.count - a.count).map(({ rep, count, all }) => (
                          <KanbanCard key={rep.id} card={rep} count={count} onClick={() => setSelected({ cards: all, index: all.indexOf(rep) })} />
                        ))
                    }
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </>
  );
}
