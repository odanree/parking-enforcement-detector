import { useEffect, useState } from 'react';
import { useAppStore } from '../store';

interface ComparisonEvent {
  id: string;
  description: string;
  detected: number;
  confidence: number;
  reeval_detected: number;
  reeval_confidence: number;
  reeval_description: string;
  reeval_backend: string;
  thumb_file: string;
  label: string;
  timestamp: number;
  camera_id: number;
  agreement: boolean;
}

interface ComparisonData {
  total: number;
  agreement_rate: number | null;
  disagreements: ComparisonEvent[];
  agreements: ComparisonEvent[];
  progress: { completed: number; total: number; running: boolean };
}

type Filter = 'fp_fixed' | 'new_detections' | 'all';

function thumb_url(thumb_file: string) {
  return thumb_file ? `/dataset/${thumb_file}` : '';
}

function pct(v: number) {
  return `${Math.round(v * 100)}%`;
}

function LabelBadge({ label }: { label: string }) {
  if (!label) return null;
  const cls = label === 'true_positive' ? 'cmp-label-tp'
            : label === 'false_positive' ? 'cmp-label-fp'
            : 'cmp-label-other';
  const text = label === 'true_positive' ? 'TP' : label === 'false_positive' ? 'FP' : label;
  return <span className={`cmp-label-badge ${cls}`}>{text}</span>;
}

function ComparisonCard({ ev, onLabel }: { ev: ComparisonEvent; onLabel: (id: string, label: string) => void }) {
  const timeStr = new Date(ev.timestamp * 1000).toLocaleTimeString([], {
    hour: '2-digit', minute: '2-digit', second: '2-digit',
  });
  const fpFixed     = ev.detected && !ev.reeval_detected;
  const newDetected = !ev.detected && ev.reeval_detected;

  return (
    <div className={`cmp-card${fpFixed ? ' fp-fixed' : newDetected ? ' new-detect' : ''}`}>
      {thumb_url(ev.thumb_file) && (
        <img src={thumb_url(ev.thumb_file)} alt="" loading="lazy" className="cmp-thumb" />
      )}
      <div className="cmp-meta">
        <div className="cmp-header">
          <span className={`cmp-tag ${fpFixed ? 'tag-fp-fixed' : 'tag-new-detect'}`}>
            {fpFixed ? 'FP Fixed' : newDetected ? 'New Detection' : 'Changed'}
          </span>
          <span className="cmp-time">Cam {ev.camera_id} · {timeStr}</span>
          <LabelBadge label={ev.label} />
        </div>

        <div className="cmp-row">
          <div className="cmp-col">
            <div className="cmp-col-label">Original · {pct(ev.confidence)}</div>
            <div className="cmp-col-detect">
              {ev.detected ? '✓ detected' : '✗ not detected'}
            </div>
            <div className="cmp-desc">{ev.description || '—'}</div>
          </div>
          <div className="cmp-divider" />
          <div className="cmp-col">
            <div className="cmp-col-label">Reeval · {pct(ev.reeval_confidence)}</div>
            <div className="cmp-col-detect">
              {ev.reeval_detected ? '✓ detected' : '✗ not detected'}
            </div>
            <div className="cmp-desc">{ev.reeval_description || '—'}</div>
          </div>
        </div>

        <div className="cmp-actions">
          <button
            className={`cmp-btn-label ${ev.label === 'true_positive' ? 'active' : ''}`}
            onClick={() => onLabel(ev.id, ev.label === 'true_positive' ? '' : 'true_positive')}
          >TP</button>
          <button
            className={`cmp-btn-label ${ev.label === 'false_positive' ? 'active active-fp' : ''}`}
            onClick={() => onLabel(ev.id, ev.label === 'false_positive' ? '' : 'false_positive')}
          >FP</button>
        </div>
      </div>
    </div>
  );
}

export function ComparisonDrawer() {
  const comparisonOpen    = useAppStore((s) => s.comparisonOpen);
  const setComparisonOpen = useAppStore((s) => s.setComparisonOpen);

  const [data, setData]     = useState<ComparisonData | null>(null);
  const [loading, setLoading] = useState(false);
  const [filter, setFilter]  = useState<Filter>('fp_fixed');
  const [labels, setLabels]  = useState<Record<string, string>>({});

  useEffect(() => {
    if (!comparisonOpen) return;
    setLoading(true);
    fetch('/api/dataset/comparison')
      .then((r) => r.json())
      .then((d) => {
        setData(d);
        const init: Record<string, string> = {};
        for (const ev of [...d.disagreements, ...d.agreements]) {
          if (ev.label) init[ev.id] = ev.label;
        }
        setLabels(init);
      })
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [comparisonOpen]);

  useEffect(() => {
    if (!comparisonOpen) return;
    function onKey(e: KeyboardEvent) { if (e.key === 'Escape') setComparisonOpen(false); }
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [comparisonOpen, setComparisonOpen]);

  async function handleLabel(id: string, label: string) {
    try {
      await fetch(`/api/dataset/${id}/label`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ label }),
      });
      setLabels((prev) => ({ ...prev, [id]: label }));
    } catch { /* ignore */ }
  }

  const allItems = data ? [...data.disagreements, ...data.agreements] : [];

  const items = allItems
    .map((ev) => ({ ...ev, label: labels[ev.id] ?? ev.label }))
    .filter((ev) => {
      if (filter === 'fp_fixed')       return ev.detected && !ev.reeval_detected;
      if (filter === 'new_detections') return !ev.detected && ev.reeval_detected;
      return true;
    });

  const fpCount  = allItems.filter((e) => e.detected && !e.reeval_detected).length;
  const ndCount  = allItems.filter((e) => !e.detected && e.reeval_detected).length;
  const agrPct   = data?.agreement_rate != null ? `${Math.round(data.agreement_rate * 100)}%` : '—';

  return (
    <>
      <div
        className={`debug-backdrop${comparisonOpen ? ' open' : ''}`}
        onClick={() => setComparisonOpen(false)}
      />
      <aside className={`cmp-drawer${comparisonOpen ? ' open' : ''}`} aria-label="Prompt comparison">
        <div className="debug-drawer-header">
          <h3>
            Reeval Comparison
            {data && (
              <span className="debug-count" title="agreement rate">{agrPct} agree</span>
            )}
          </h3>
          <div className="debug-drawer-actions">
            <button className="btn-debug-close" aria-label="Close" onClick={() => setComparisonOpen(false)}>
              &times;
            </button>
          </div>
        </div>

        <div className="cmp-filter-bar">
          <button
            className={`cmp-filter-btn${filter === 'fp_fixed' ? ' active' : ''}`}
            onClick={() => setFilter('fp_fixed')}
          >FP Fixed ({fpCount})</button>
          <button
            className={`cmp-filter-btn${filter === 'new_detections' ? ' active' : ''}`}
            onClick={() => setFilter('new_detections')}
          >New Detections ({ndCount})</button>
          <button
            className={`cmp-filter-btn${filter === 'all' ? ' active' : ''}`}
            onClick={() => setFilter('all')}
          >All ({allItems.length})</button>
        </div>

        <div className="debug-list">
          {loading && <div className="debug-empty">Loading…</div>}
          {!loading && items.length === 0 && (
            <div className="debug-empty">No items for this filter</div>
          )}
          {!loading && items.map((ev) => (
            <ComparisonCard key={ev.id} ev={ev} onLabel={handleLabel} />
          ))}
        </div>
      </aside>
    </>
  );
}
