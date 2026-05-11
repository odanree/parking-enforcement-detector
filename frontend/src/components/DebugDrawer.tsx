import { useEffect, useRef, useState } from 'react';
import { useAppStore } from '../store';
import type { DebugItem } from '../types';
import { TYPE_LABELS } from '../types';

function DebugThumb({ item }: { item: DebugItem }) {
  const hasFrames = item.kind === 'chalking' && item.frames && item.frames.length > 1;
  const [idx, setIdx] = useState(0);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (!hasFrames) return;
    timerRef.current = setInterval(() => {
      setIdx((i) => (i + 1) % item.frames!.length);
    }, 500);
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, [hasFrames, item.frames]);

  if (hasFrames) {
    return (
      <img
        src={`data:image/jpeg;base64,${item.frames![idx]}`}
        alt={`frame ${idx + 1}/${item.frames!.length}`}
        loading="lazy"
      />
    );
  }
  if (item.thumbnail) {
    return <img src={`data:image/jpeg;base64,${item.thumbnail}`} alt="" loading="lazy" />;
  }
  return null;
}

export function DebugDrawer() {
  const debugItems   = useAppStore((s) => s.debugItems);
  const debugOpen    = useAppStore((s) => s.debugOpen);
  const setDebugOpen = useAppStore((s) => s.setDebugOpen);

  useEffect(() => {
    if (!debugOpen) return;
    function onKey(e: KeyboardEvent) { if (e.key === 'Escape') setDebugOpen(false); }
    document.addEventListener('keydown', onKey);
    return () => document.removeEventListener('keydown', onKey);
  }, [debugOpen, setDebugOpen]);

  async function clearAll() {
    try {
      await fetch('/api/debug/rejected', { method: 'DELETE' });
      useAppStore.getState().setDebugItems([]);
    } catch { /* ignore */ }
  }

  const offHoursCount = debugItems.filter((i) => i.rejection_reason === 'out_of_hours').length;
  const vlmCount      = debugItems.filter((i) => i.rejection_reason !== 'out_of_hours' && !i.suppressed_by_session).length;
  const suppressedCount = debugItems.filter((i) => !!i.suppressed_by_session).length;

  return (
    <>
      <div
        className={`debug-backdrop${debugOpen ? ' open' : ''}`}
        onClick={() => setDebugOpen(false)}
      />
      <aside className={`debug-drawer${debugOpen ? ' open' : ''}`} aria-label="VLM debug">
        <div className="debug-drawer-header">
          <h3>
            Not Detected{' '}
            <span className="debug-count">{debugItems.length}</span>
          </h3>
          <div className="debug-drawer-actions">
            <button className="btn-debug-clear" onClick={clearAll}>Clear</button>
            <button id="btn-debug-close" className="btn-debug-close" aria-label="Close" onClick={() => setDebugOpen(false)}>
              &times;
            </button>
          </div>
        </div>
        {debugItems.length > 0 && (
          <div className="debug-breakdown">
            {offHoursCount > 0 && <span className="dbk-offhours">⏰ {offHoursCount} off-hours</span>}
            {vlmCount > 0      && <span className="dbk-vlm">🔍 {vlmCount} VLM</span>}
            {suppressedCount > 0 && <span className="dbk-suppressed">⛔ {suppressedCount} suppressed</span>}
          </div>
        )}
        <div className="debug-list">
          {debugItems.length === 0
            ? <div className="debug-empty">No rejected frames yet</div>
            : debugItems.map((item, i) => {
                const pct          = Math.round((item.confidence ?? 0) * 100);
                const timeStr      = new Date(item.timestamp * 1000).toLocaleTimeString([], {
                  hour: '2-digit', minute: '2-digit', second: '2-digit',
                });
                const isSuppressed = !!item.suppressed_by_session;
                const isOffHours   = item.rejection_reason === 'out_of_hours';
                return (
                  <div key={i} className={`debug-item${isSuppressed ? ' suppressed' : ''}${isOffHours ? ' off-hours' : ''}`}>
                    <DebugThumb item={item} />
                    <div className="debug-item-meta">
                      <div className="debug-item-header">
                        <span className={`debug-kind ${item.kind}`}>{TYPE_LABELS[item.kind] ?? item.kind}</span>
                        {isOffHours
                          ? <span className="debug-offhours-badge">⏰ Off Hours</span>
                          : isSuppressed
                            ? <span className="debug-suppressed-badge">⛔ S:{item.suppressed_by_session}</span>
                            : <span className="debug-conf">{pct}%</span>
                        }
                        <span className="debug-time">{timeStr}</span>
                      </div>
                      <div className="debug-desc">
                        {isOffHours   ? (item.description || 'Outside PE enforcement window') :
                         isSuppressed ? 'Suppressed by user downvote' :
                                        (item.description || 'No description')}
                      </div>
                    </div>
                  </div>
                );
              })
          }
        </div>
      </aside>
    </>
  );
}
