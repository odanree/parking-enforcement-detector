import { useEffect, useMemo, useRef, useState, useCallback } from 'react';
import { useAppStore } from '../store';
import type { PipelineStage, RagNeighbor, RagStage } from '../types';

type TimeFilter  = '5m' | '1h' | '6h' | '12h' | '7d' | '30d' | 'all';
type LabelFilter = 'unlabeled' | 'labeled' | 'all';
type CamFilter   = 0 | 1 | 'all';

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
  rag_neighbors: RagNeighbor[];
  track_id?: number | null;
  phash?: string | null;
  yolo?: { confidence: number; track_id?: number; bbox?: number[] } | null;
  first_pass: PipelineStage | null;
  rag: RagStage | null;
  confirm: PipelineStage | null;
  label?: string;
  rejection_reason?: string;
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

  // Reset whenever the image changes (prev/next navigation)
  useEffect(() => {
    const z = { scale: 1, tx: 0, ty: 0 };
    zoomRef.current = z;
    setXform(z);
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
      />
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

function KanbanCard({ card, count, onClick }: { card: TraceCard; count?: number; onClick: () => void }) {
  const timeStr = new Date(card.timestamp * 1000).toLocaleTimeString([], {
    hour: '2-digit', minute: '2-digit', second: '2-digit',
  });
  const src = thumbSrc(card);

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
            ? <><span className="stage-pos-text">✓ People alert</span> <ConfBadge v={card.confidence} detected /></>
            : <><span className="stage-neg-text">✗ Rejected</span> <ConfBadge v={card.confidence} detected={false} /></>
          }
        </div>
      </div>
    </div>
  );
}

function StageDetailModal({ cards: allCards, startIndex, onClose }: { cards: TraceCard[]; startIndex: number; onClose: () => void }) {
  const [idx, setIdx] = useState(startIndex);
  const card = allCards[idx];
  const timeStr = new Date(card.timestamp * 1000).toLocaleString();
  const total = allCards.length;

  return (
    <div className="kanban-modal-overlay" onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <div className="kanban-modal">
        <button className="kanban-modal-close" onClick={onClose}>&times;</button>
        <div className="kanban-modal-header">
          <span className={`kanban-outcome-badge ${card.outcome}`}>
            {card.outcome === 'chalking' ? '✓ Chalking' : card.outcome === 'people_alert' || card.outcome === 'detected' ? '✓ People Alert' : '✗ Rejected'}
          </span>
          <span className="kanban-modal-time">Cam {card.camera_id} · {timeStr}</span>
        </div>

        {total > 1 && (
          <div className="kanban-modal-nav-row">
            <button className="kanban-modal-nav-btn" onClick={() => setIdx((i) => (i - 1 + total) % total)}>&#8592;</button>
            <span className="kanban-modal-nav-count">{idx + 1} / {total}</span>
            <button className="kanban-modal-nav-btn" onClick={() => setIdx((i) => (i + 1) % total)}>&#8594;</button>
          </div>
        )}

        {thumbSrc(card) && <ZoomableImage src={thumbSrc(card)!} />}

        {card.label && (
          <div style={{ marginBottom: 8 }}>
            <span className={`cmp-label-badge ${card.label === 'true_positive' ? 'cmp-label-tp' : card.label === 'false_positive' ? 'cmp-label-fp' : 'cmp-label-other'}`}>
              {card.label === 'true_positive' ? 'TP' : card.label === 'false_positive' ? 'FP' : card.label}
            </span>
          </div>
        )}

        <CardDetail card={card} />
      </div>
    </div>
  );
}

const COLUMNS: { key: string; label: string; desc: string }[] = [
  { key: 'time_window',   label: 'Off Hours ✗',    desc: 'Outside PE enforcement window'         },
  { key: 'first_pass',    label: 'Primary ✗',      desc: 'Rejected by 1st-pass VLM'             },
  { key: 'rag',           label: 'RAG Blocked ✗',  desc: 'Auto-rejected by FP matches'          },
  { key: 'confirm',       label: 'Reeval ✗',       desc: 'Rejected by confirm/reeval'           },
  { key: 'people_alert',  label: 'People Alert ✓', desc: 'Person near vehicle (VLM positive)'   },
  { key: 'chalking',      label: 'Chalking ✓',     desc: 'Confirmed by both VLM stages'         },
];

function cardColumn(card: TraceCard): string {
  if (card.rejection_reason === 'out_of_hours') return 'time_window';

  // Explicit people_alert outcome (two-stage: first-pass positive before confirm)
  if (card.outcome === 'people_alert') return 'people_alert';

  if (card.outcome === 'rejected') {
    if (!card.first_pass?.detected) return 'first_pass';
    if (card.rag?.skipped_confirm) return 'rag';
    if (card.confirm && !card.confirm.detected) return 'confirm';
    return 'first_pass';
  }

  // outcome === 'detected'
  // Two-stage confirm positive → chalking (rare)
  if (card.confirm?.detected) return 'chalking';
  // Single-stage positive (no confirm stage ran) → people alert (common)
  return 'people_alert';
}

export function PipelineKanban() {
  const kanbanOpen    = useAppStore((s) => s.kanbanOpen);
  const setKanbanOpen = useAppStore((s) => s.setKanbanOpen);

  const [cards, setCards]       = useState<TraceCard[]>([]);
  const [loading, setLoading]   = useState(false);
  const [selected, setSelected] = useState<{ cards: TraceCard[]; index: number } | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const [timeFilter,  setTimeFilter]  = useState<TimeFilter>('12h');
  const [labelFilter, setLabelFilter] = useState<LabelFilter>('unlabeled');
  const [camFilter,   setCamFilter]   = useState<CamFilter>('all');

  const timeWindowSecs = useMemo(() => {
    if (timeFilter === 'all') return null;
    return timeFilter === '5m' ? 300 : timeFilter === '1h' ? 3600 : timeFilter === '6h' ? 6 * 3600
         : timeFilter === '12h' ? 12 * 3600 : timeFilter === '7d' ? 7 * 86400 : 30 * 86400;
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

  const detected = visibleCards.filter((c) => c.outcome === 'detected' || c.outcome === 'people_alert');
  const rejected  = visibleCards.filter((c) => c.outcome === 'rejected');

  const byColumn: Record<string, TraceCard[]> = {
    time_window: [], first_pass: [], rag: [], confirm: [], people_alert: [], chalking: [],
  };
  for (const card of visibleCards) {
    const col = cardColumn(card);
    byColumn[col].push(card);
  }

  if (!kanbanOpen) return null;

  return (
    <>
      {selected && (
        <StageDetailModal cards={selected.cards} startIndex={selected.index} onClose={() => setSelected(null)} />
      )}
      <div className="kanban-overlay" onClick={(e) => { if (e.target === e.currentTarget) setKanbanOpen(false); }}>
        <div className="kanban-panel">
          <div className="kanban-panel-header">
            <h2 className="kanban-title">Pipeline Kanban</h2>
            <div className="kanban-filters">
              {/* Time */}
              {(['5m','1h','6h','12h','7d','30d','all'] as TimeFilter[]).map((t) => (
                <button key={t} className={`kanban-filter-btn${timeFilter === t ? ' active' : ''}`}
                  onClick={() => setTimeFilter(t)}>
                  {t === '5m' ? '5 min' : t === '1h' ? '1 hour' : t === '6h' ? '6 hours' : t === '12h' ? '12 hours' : t === '7d' ? '7 days' : t === '30d' ? '30 days' : 'All time'}
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
              <label className="kanban-auto-label">
                <input type="checkbox" checked={autoRefresh} onChange={(e) => setAutoRefresh(e.target.checked)} />
                Live
              </label>
              <button className="btn-debug-close" onClick={() => setKanbanOpen(false)}>&times;</button>
            </div>
          </div>

          {loading && <div className="kanban-loading">Loading…</div>}

          <div className="kanban-board">
            {COLUMNS.map((col) => (
              <div key={col.key} className="kanban-col">
                <div className="kanban-col-header">
                  <span className="kanban-col-title">{col.label}</span>
                  <span className="kanban-col-desc">{col.desc}</span>
                  <span className="kanban-col-count">{byColumn[col.key].length}</span>
                </div>
                <div className="kanban-col-cards">
                  {byColumn[col.key].length === 0
                    ? <div className="kanban-col-empty">—</div>
                    : groupCards(byColumn[col.key]).map(({ rep, count, all }) => (
                        <KanbanCard key={rep.id} card={rep} count={count} onClick={() => setSelected({ cards: all, index: all.indexOf(rep) })} />
                      ))
                  }
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </>
  );
}
