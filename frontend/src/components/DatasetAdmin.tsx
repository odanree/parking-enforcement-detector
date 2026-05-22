import { useEffect, useState, useCallback, useRef } from 'react';
import { useAppStore } from '../store';

const PAGE_SIZE = 50;

type LabelVal = '' | 'true_positive' | 'false_positive' | 'true_negative' | 'false_negative';
type ViewMode  = 'list' | 'grouped';

interface DbItem {
  id:             string;
  description:    string;
  detected:       number;
  confidence:     number;
  camera_id:      number;
  timestamp:      number;
  label:          string;
  person_type:    string;
  capture_source: string;
  thumb_file:     string;
  hires_file:     string;
  model_primary:  string;
  yolo_confidence: number;
  thumb_phash?:   string;
}

interface DbGroup { rep: DbItem; count: number; all: DbItem[] }

const HASH_THRESHOLD = 10;

function hammingDist(a: string, b: string): number {
  let d = 0;
  for (let i = 0; i < Math.min(a.length, b.length); i++) {
    const xor = parseInt(a[i], 16) ^ parseInt(b[i], 16);
    d += xor.toString(2).split('1').length - 1;
  }
  return d;
}

function groupItems(items: DbItem[]): DbGroup[] {
  const clusters: { camId: number; hash: string; items: DbItem[] }[] = [];
  for (const item of items) {
    const h = item.thumb_phash;
    if (h) {
      const ex = clusters.find(
        g => g.camId === item.camera_id && hammingDist(g.hash, h) <= HASH_THRESHOLD,
      );
      if (ex) { ex.items.push(item); continue; }
      clusters.push({ camId: item.camera_id, hash: h, items: [item] });
    } else {
      clusters.push({ camId: item.camera_id, hash: '', items: [item] });
    }
  }
  return clusters.map(({ items: all }) => {
    const rep = all.reduce((best, c) => c.confidence > best.confidence ? c : best, all[0]);
    return { rep, count: all.length, all };
  });
}

function fmt(ts: number) {
  return new Date(ts * 1000).toLocaleString([], {
    month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit', second: '2-digit',
  });
}

function AdminRow({
  item,
  rowIndex,
  onLabel,
  onDelete,
  onPreview,
}: {
  item: DbItem;
  rowIndex: number;
  onLabel: (id: string, label: LabelVal) => void;
  onDelete: (id: string) => void;
  onPreview: (idx: number) => void;
}) {
  const [labelBusy, setLabelBusy] = useState(false);

  async function changeLabel(val: string) {
    setLabelBusy(true);
    try {
      await fetch(`/api/dataset/${item.id}/label`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ label: val }),
      });
      onLabel(item.id, val as LabelVal);
    } finally {
      setLabelBusy(false);
    }
  }

  const thumbUrl = item.thumb_file ? `/dataset/${item.thumb_file}` : null;
  const hiresUrl = item.hires_file ? `/dataset/${item.hires_file}` : null;
  const previewUrl = hiresUrl || thumbUrl;

  return (
    <tr className="da-row">
      <td className="da-cell da-cell-thumb">
        {thumbUrl
          ? <img
              className="da-thumb"
              src={thumbUrl}
              alt=""
              onClick={() => previewUrl && onPreview(rowIndex)}
              title={hiresUrl ? 'Click for hi-res' : 'Click to enlarge'}
            />
          : <span className="da-no-thumb">—</span>
        }
      </td>
      <td className="da-cell da-cell-time">{fmt(item.timestamp)}</td>
      <td className="da-cell da-cell-cam">Cam {item.camera_id}</td>
      <td className={`da-cell da-cell-det ${item.detected ? 'da-yes' : 'da-no'}`}>
        {item.detected ? '✓' : '✗'}
      </td>
      <td className="da-cell da-cell-conf">
        {Math.round(item.confidence * 100)}%
      </td>
      <td className="da-cell da-cell-label">
        <select
          className={`da-label-select da-label-${item.label || 'none'}`}
          value={item.label}
          onChange={e => changeLabel(e.target.value)}
          disabled={labelBusy}
        >
          <option value="">—</option>
          <option value="true_positive">TP</option>
          <option value="false_positive">FP</option>
          <option value="true_negative">TN</option>
          <option value="false_negative">FN</option>
        </select>
      </td>
      <td className="da-cell da-cell-source">
        {item.capture_source === 'zone_pedestrian' ? 'zone' : item.capture_source || '—'}
      </td>
      <td className="da-cell da-cell-desc" title={item.description}>
        {item.description.slice(0, 90)}{item.description.length > 90 ? '…' : ''}
      </td>
      <td className="da-cell da-cell-del">
        <button
          className="da-del-btn"
          title="Delete from database"
          onClick={() => {
            if (confirm('Delete this record from the database?')) onDelete(item.id);
          }}
        >✕</button>
      </td>
    </tr>
  );
}

function GroupCard({
  group,
  onOpen,
}: {
  group: DbGroup;
  onOpen: () => void;
}) {
  const { rep, count, all } = group;
  const thumbUrl = rep.thumb_file ? `/dataset/${rep.thumb_file}` : null;

  const lc: Record<string, number> = {};
  for (const item of all) lc[item.label || 'none'] = (lc[item.label || 'none'] ?? 0) + 1;
  const hasLabels = Object.keys(lc).some(k => k !== 'none');

  return (
    <div className="da-group-card" onClick={onOpen}>
      <div className="da-group-thumb-wrap">
        {thumbUrl
          ? <img className="da-group-thumb" src={thumbUrl} alt="" />
          : <span className="da-no-thumb">—</span>
        }
        {count > 1 && <span className="da-group-count">{count}</span>}
        <span className={`da-group-det ${rep.detected ? 'da-yes' : 'da-no'}`}>
          {rep.detected ? '✓' : '✗'}
        </span>
      </div>
      <div className="da-group-meta">
        <span>Cam {rep.camera_id}</span>
        <span>{Math.round(rep.confidence * 100)}%</span>
      </div>
      <div className="da-group-time">{fmt(rep.timestamp)}</div>
      {hasLabels && (
        <div className="da-group-labels">
          {(lc['true_positive']  ?? 0) > 0 && <span className="da-gl da-gl-tp">TP×{lc['true_positive']}</span>}
          {(lc['false_positive'] ?? 0) > 0 && <span className="da-gl da-gl-fp">FP×{lc['false_positive']}</span>}
          {(lc['true_negative']  ?? 0) > 0 && <span className="da-gl da-gl-tn">TN×{lc['true_negative']}</span>}
          {(lc['false_negative'] ?? 0) > 0 && <span className="da-gl da-gl-fn">FN×{lc['false_negative']}</span>}
        </div>
      )}
    </div>
  );
}

export function DatasetAdmin() {
  const adminOpen    = useAppStore((s) => s.adminOpen);
  const setAdminOpen = useAppStore((s) => s.setAdminOpen);

  const [items,   setItems]   = useState<DbItem[]>([]);
  const [total,   setTotal]   = useState(0);
  const [offset,  setOffset]  = useState(0);
  const [loading, setLoading] = useState(false);

  const [filterLabel,    setFilterLabel]    = useState('all');
  const [filterDetected, setFilterDetected] = useState('all');
  const [filterCamera,   setFilterCamera]   = useState('all');
  const [filterSource,   setFilterSource]   = useState('hires');  // default: hide lo-res zone_pedestrian

  const [viewMode,    setViewMode]    = useState<ViewMode>('list');
  const [previewIdx,  setPreviewIdx]  = useState<number | null>(null);
  const [groupPreview, setGroupPreview] = useState<{ all: DbItem[]; idx: number } | null>(null);

  const [deduping,  setDeduping]  = useState(false);
  const [exporting, setExporting] = useState(false);
  const [clearing,  setClearing]  = useState(false);

  const fetchRef = useRef(0);

  const fetchPage = useCallback(async (off: number) => {
    const seq = ++fetchRef.current;
    setLoading(true);
    const isGrouped = viewMode === 'grouped';
    const lim = isGrouped ? 500 : PAGE_SIZE;
    const actualOff = isGrouped ? 0 : off;
    const p = new URLSearchParams({ offset: String(actualOff), limit: String(lim) });
    if (filterLabel    !== 'all') p.set('label',          filterLabel === 'unlabeled' ? '' : filterLabel);
    if (filterDetected !== 'all') p.set('detected',       filterDetected);
    if (filterCamera   !== 'all') p.set('camera_id',      filterCamera);
    if (filterSource === 'hires')      p.set('exclude_source', 'zone_pedestrian');
    else if (filterSource !== 'all')   p.set('capture_source', filterSource);
    try {
      const res  = await fetch(`/api/dataset?${p}`);
      const data = await res.json();
      if (fetchRef.current !== seq) return;
      const sorted = (data.items ?? []).sort((a: DbItem, b: DbItem) => b.timestamp - a.timestamp);
      setItems(sorted);
      setTotal(data.total ?? 0);
      setOffset(actualOff);
    } finally {
      if (fetchRef.current === seq) setLoading(false);
    }
  }, [filterLabel, filterDetected, filterCamera, filterSource, viewMode]);

  useEffect(() => {
    if (!adminOpen) return;
    fetchPage(0);
  }, [adminOpen, filterLabel, filterDetected, filterCamera, filterSource, viewMode]);

  useEffect(() => {
    if (!adminOpen) return;
    function onKey(e: KeyboardEvent) {
      if (groupPreview !== null) {
        if (e.key === 'ArrowLeft')  { setGroupPreview(g => g ? { ...g, idx: Math.max(0, g.idx - 1) } : g); return; }
        if (e.key === 'ArrowRight') { setGroupPreview(g => g ? { ...g, idx: Math.min(g.all.length - 1, g.idx + 1) } : g); return; }
        if (e.key === 'Escape')     { setGroupPreview(null); return; }
      } else if (previewIdx !== null) {
        if (e.key === 'ArrowLeft')  { setPreviewIdx(i => Math.max(0, (i ?? 0) - 1)); return; }
        if (e.key === 'ArrowRight') { setPreviewIdx(i => Math.min(items.length - 1, (i ?? 0) + 1)); return; }
        if (e.key === 'Escape')     { setPreviewIdx(null); return; }
      } else {
        if (e.key === 'Escape') setAdminOpen(false);
      }
    }
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [adminOpen, previewIdx, groupPreview, items.length, setAdminOpen]);

  function handleLabel(id: string, label: LabelVal) {
    setItems(prev => prev.map(it => it.id === id ? { ...it, label } : it));
    setGroupPreview(g => g ? { ...g, all: g.all.map(it => it.id === id ? { ...it, label } : it) } : g);
  }

  const [hiresBusy, setHiresBusy] = useState<string | null>(null);

  async function handleCaptureHires(id: string) {
    setHiresBusy(id);
    try {
      const res = await fetch(`/api/dataset/${id}/capture-hires`, { method: 'POST' });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        alert(`Hi-res capture failed: ${err.detail || res.status}`);
        return;
      }
      const data = await res.json();
      const hires_file = (data.hires_url || '').replace('/dataset/', '');
      const patch = (it: DbItem) => it.id === id ? { ...it, hires_file, capture_source: 'manual_hires' } : it;
      setItems(prev => prev.map(patch));
      setGroupPreview(g => g ? { ...g, all: g.all.map(patch) } : g);
    } catch (e) {
      alert(`Hi-res capture error: ${e}`);
    } finally {
      setHiresBusy(null);
    }
  }

  async function handleDelete(id: string) {
    await fetch(`/api/dataset/${id}`, { method: 'DELETE' });
    setItems(prev => prev.filter(it => it.id !== id));
    setTotal(t => t - 1);
    setGroupPreview(g => {
      if (!g) return g;
      const next = g.all.filter(it => it.id !== id);
      if (next.length === 0) return null;
      return { all: next, idx: Math.min(g.idx, next.length - 1) };
    });
  }

  async function deduplicate() {
    setDeduping(true);
    try {
      const res  = await fetch('/api/dataset/deduplicate', { method: 'POST' });
      const data = await res.json();
      alert(`Removed ${data.removed} duplicate(s). ${data.remaining} records remain.`);
      fetchPage(0);
    } finally {
      setDeduping(false);
    }
  }

  async function clearAll() {
    const answer = prompt(`This will permanently delete ALL ${total} records and their images.\n\nType DELETE to confirm:`);
    if (answer !== 'DELETE') return;
    setClearing(true);
    try {
      const res  = await fetch('/api/dataset', { method: 'DELETE' });
      const data = await res.json();
      alert(`Cleared ${data.removed} records.`);
      setItems([]);
      setTotal(0);
      setOffset(0);
    } finally {
      setClearing(false);
    }
  }

  async function exportData() {
    setExporting(true);
    try {
      const res  = await fetch('/api/dataset/export');
      const data = await res.json();
      const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
      const url  = URL.createObjectURL(blob);
      const a    = document.createElement('a');
      a.href     = url;
      a.download = `dataset-${new Date().toISOString().slice(0, 10)}.json`;
      a.click();
      URL.revokeObjectURL(url);
    } finally {
      setExporting(false);
    }
  }

  if (!adminOpen) return null;

  const page       = Math.floor(offset / PAGE_SIZE) + 1;
  const totalPages = Math.ceil(total / PAGE_SIZE) || 1;

  const previewItem = previewIdx !== null ? items[previewIdx] : null;
  const previewUrl  = previewItem
    ? (previewItem.hires_file ? `/dataset/${previewItem.hires_file}` : previewItem.thumb_file ? `/dataset/${previewItem.thumb_file}` : null)
    : null;

  const gpItem = groupPreview ? groupPreview.all[groupPreview.idx] : null;
  const gpUrl  = gpItem
    ? (gpItem.hires_file ? `/dataset/${gpItem.hires_file}` : gpItem.thumb_file ? `/dataset/${gpItem.thumb_file}` : null)
    : null;

  const groups = viewMode === 'grouped' ? groupItems(items) : [];

  return (
    <>
      {/* ── Group lightbox ── */}
      {gpItem && gpUrl && (
        <div className="da-preview-backdrop" onClick={() => setGroupPreview(null)}>
          <button
            className="da-preview-nav da-preview-prev"
            onClick={e => { e.stopPropagation(); setGroupPreview(g => g ? { ...g, idx: Math.max(0, g.idx - 1) } : g); }}
            disabled={groupPreview!.idx === 0}
          >‹</button>
          <div className="da-preview-content" onClick={e => e.stopPropagation()}>
            <img className="da-preview-img" src={gpUrl} alt="" />
            <div className="da-preview-caption">
              <span>{fmt(gpItem.timestamp)}</span>
              <span>Cam {gpItem.camera_id}</span>
              <span className={gpItem.detected ? 'da-yes' : 'da-no'}>{gpItem.detected ? '✓ Detected' : '✗ Rejected'}</span>
              <GroupPreviewLabel item={gpItem} onLabel={handleLabel} />
              <span className="da-preview-counter">{groupPreview!.idx + 1} / {groupPreview!.all.length}</span>
              {!gpItem.hires_file && (
                <button
                  className="da-hires-btn"
                  title="Re-grab a hi-res frame from the NVR recording at this timestamp"
                  disabled={hiresBusy === gpItem.id}
                  onClick={() => handleCaptureHires(gpItem.id)}
                >{hiresBusy === gpItem.id ? '…' : '⤓ hi-res'}</button>
              )}
              <button
                className="da-del-btn"
                title="Delete from database"
                onClick={() => { if (confirm('Delete this record?')) handleDelete(gpItem.id); }}
              >✕</button>
            </div>
          </div>
          <button
            className="da-preview-nav da-preview-next"
            onClick={e => { e.stopPropagation(); setGroupPreview(g => g ? { ...g, idx: Math.min(g.all.length - 1, g.idx + 1) } : g); }}
            disabled={groupPreview!.idx === groupPreview!.all.length - 1}
          >›</button>
        </div>
      )}

      {/* ── List lightbox ── */}
      {previewItem && previewUrl && (
        <div className="da-preview-backdrop" onClick={() => setPreviewIdx(null)}>
          <button
            className="da-preview-nav da-preview-prev"
            onClick={e => { e.stopPropagation(); setPreviewIdx(i => Math.max(0, (i ?? 0) - 1)); }}
            disabled={previewIdx === 0}
          >‹</button>
          <div className="da-preview-content" onClick={e => e.stopPropagation()}>
            <img className="da-preview-img" src={previewUrl} alt="" />
            <div className="da-preview-caption">
              <span>{fmt(previewItem.timestamp)}</span>
              <span>Cam {previewItem.camera_id}</span>
              <span className={previewItem.detected ? 'da-yes' : 'da-no'}>{previewItem.detected ? '✓ Detected' : '✗ Rejected'}</span>
              {previewItem.label && <span className={`cmp-label-badge cmp-label-${previewItem.label === 'true_positive' ? 'tp' : previewItem.label === 'false_positive' ? 'fp' : 'other'}`}>{previewItem.label === 'true_positive' ? 'TP' : previewItem.label === 'false_positive' ? 'FP' : previewItem.label}</span>}
              <span className="da-preview-counter">{(previewIdx ?? 0) + 1} / {items.length}</span>
              {!previewItem.hires_file && (
                <button
                  className="da-hires-btn"
                  title="Re-grab a hi-res frame from the NVR recording at this timestamp"
                  disabled={hiresBusy === previewItem.id}
                  onClick={() => handleCaptureHires(previewItem.id)}
                >{hiresBusy === previewItem.id ? '…' : '⤓ hi-res'}</button>
              )}
            </div>
          </div>
          <button
            className="da-preview-nav da-preview-next"
            onClick={e => { e.stopPropagation(); setPreviewIdx(i => Math.min(items.length - 1, (i ?? 0) + 1)); }}
            disabled={previewIdx === items.length - 1}
          >›</button>
        </div>
      )}

      <div className="dataset-admin-overlay" onClick={e => { if (e.target === e.currentTarget) setAdminOpen(false); }}>
        <div className="dataset-admin-panel">

          {/* ── Header ── */}
          <div className="dataset-admin-header">
            <div className="da-title-row">
              <h2 className="da-title">Dataset Admin</h2>
              <span className="da-total">{total.toLocaleString()} records</span>
            </div>

            {/* ── View toggle ── */}
            <div className="da-view-toggle">
              <button
                className={`da-view-btn${viewMode === 'list' ? ' active' : ''}`}
                onClick={() => setViewMode('list')}
              >≡ List</button>
              <button
                className={`da-view-btn${viewMode === 'grouped' ? ' active' : ''}`}
                onClick={() => setViewMode('grouped')}
              >⊟ Grouped</button>
            </div>

            {/* ── Filters ── */}
            <div className="da-filters">
              <select value={filterLabel} onChange={e => setFilterLabel(e.target.value)}>
                <option value="all">All labels</option>
                <option value="unlabeled">Unlabeled</option>
                <option value="true_positive">True Positive</option>
                <option value="false_positive">False Positive</option>
                <option value="true_negative">True Negative</option>
                <option value="false_negative">False Negative</option>
              </select>
              <select value={filterDetected} onChange={e => setFilterDetected(e.target.value)}>
                <option value="all">All outcomes</option>
                <option value="1">Detected ✓</option>
                <option value="0">Rejected ✗</option>
              </select>
              <select value={filterCamera} onChange={e => setFilterCamera(e.target.value)}>
                <option value="all">All cameras</option>
                <option value="0">Cam 0</option>
                <option value="1">Cam 1</option>
              </select>
              <select value={filterSource} onChange={e => setFilterSource(e.target.value)}>
                <option value="hires">Hi-res only</option>
                <option value="all">All sources</option>
                <option value="chalking">Chalking</option>
                <option value="manual_hires">Manual hi-res</option>
                <option value="zone_pedestrian">Zone pedestrian (lo-res)</option>
              </select>
            </div>

            {/* ── Actions ── */}
            <div className="da-actions">
              <button className="da-action-btn" onClick={() => fetchPage(offset)} disabled={loading}>
                {loading ? '…' : '↺ Refresh'}
              </button>
              <button className="da-action-btn" onClick={deduplicate} disabled={deduping}>
                {deduping ? 'Deduplicating…' : 'Deduplicate'}
              </button>
              <button className="da-action-btn" onClick={exportData} disabled={exporting}>
                {exporting ? 'Exporting…' : '↓ Export JSON'}
              </button>
              <button className="da-action-btn da-action-danger" onClick={clearAll} disabled={clearing || total === 0}>
                {clearing ? 'Clearing…' : '✕ Clear All'}
              </button>
              <button className="btn-debug-close" onClick={() => setAdminOpen(false)}>&times;</button>
            </div>
          </div>

          {/* ── Content ── */}
          <div className="da-table-wrap">
            {items.length === 0 && !loading && (
              <div className="da-empty">No records match the current filters.</div>
            )}

            {/* List view */}
            {viewMode === 'list' && items.length > 0 && (
              <table className="da-table">
                <thead>
                  <tr>
                    <th>Thumb</th>
                    <th>Time</th>
                    <th>Cam</th>
                    <th>Det</th>
                    <th>Conf</th>
                    <th>Label</th>
                    <th>Source</th>
                    <th>Description</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  {items.map((item, idx) => (
                    <AdminRow
                      key={item.id}
                      item={item}
                      rowIndex={idx}
                      onLabel={handleLabel}
                      onDelete={handleDelete}
                      onPreview={setPreviewIdx}
                    />
                  ))}
                </tbody>
              </table>
            )}

            {/* Grouped view */}
            {viewMode === 'grouped' && items.length > 0 && (
              <div className="da-group-grid">
                {groups.map((group) => (
                  <GroupCard
                    key={group.rep.id}
                    group={group}
                    onOpen={() => setGroupPreview({ all: group.all, idx: 0 })}
                  />
                ))}
              </div>
            )}
          </div>

          {/* ── Pagination (list mode only) ── */}
          {viewMode === 'list' && (
            <div className="da-pagination">
              <button
                className="da-page-btn"
                onClick={() => fetchPage(offset - PAGE_SIZE)}
                disabled={offset === 0 || loading}
              >← Prev</button>
              <span className="da-page-info">
                {offset + 1}–{Math.min(offset + PAGE_SIZE, total)} of {total.toLocaleString()}
                &nbsp;&nbsp;(page {page} / {totalPages})
              </span>
              <button
                className="da-page-btn"
                onClick={() => fetchPage(offset + PAGE_SIZE)}
                disabled={offset + PAGE_SIZE >= total || loading}
              >Next →</button>
            </div>
          )}

          {viewMode === 'grouped' && items.length > 0 && (
            <div className="da-pagination">
              <span className="da-page-info">
                {groups.length} group{groups.length !== 1 ? 's' : ''} from {items.length} records
                {items.length >= 500 ? ' (showing first 500)' : ''}
              </span>
            </div>
          )}

        </div>
      </div>
    </>
  );
}

function GroupPreviewLabel({ item, onLabel }: { item: DbItem; onLabel: (id: string, label: LabelVal) => void }) {
  const [busy, setBusy] = useState(false);

  async function changeLabel(val: string) {
    setBusy(true);
    try {
      await fetch(`/api/dataset/${item.id}/label`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ label: val }),
      });
      onLabel(item.id, val as LabelVal);
    } finally {
      setBusy(false);
    }
  }

  return (
    <select
      className={`da-label-select da-label-${item.label || 'none'}`}
      value={item.label}
      onChange={e => changeLabel(e.target.value)}
      disabled={busy}
    >
      <option value="">—</option>
      <option value="true_positive">TP</option>
      <option value="false_positive">FP</option>
      <option value="true_negative">TN</option>
      <option value="false_negative">FN</option>
    </select>
  );
}
