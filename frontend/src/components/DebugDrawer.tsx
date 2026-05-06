import { useEffect } from 'react';
import { useAppStore } from '../store';
import { TYPE_LABELS } from '../types';

export function DebugDrawer() {
  const debugItems  = useAppStore((s) => s.debugItems);
  const debugOpen   = useAppStore((s) => s.debugOpen);
  const setDebugOpen = useAppStore((s) => s.setDebugOpen);

  // Close on Escape
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
        <div className="debug-list">
          {debugItems.length === 0
            ? <div className="debug-empty">No rejected frames yet</div>
            : debugItems.map((item, i) => {
                const pct     = Math.round((item.confidence ?? 0) * 100);
                const timeStr = new Date(item.timestamp * 1000).toLocaleTimeString([], {
                  hour: '2-digit', minute: '2-digit', second: '2-digit',
                });
                return (
                  <div key={i} className="debug-item">
                    {item.thumbnail && (
                      <img src={`data:image/jpeg;base64,${item.thumbnail}`} alt="" loading="lazy" />
                    )}
                    <div className="debug-item-meta">
                      <div className="debug-item-header">
                        <span className={`debug-kind ${item.kind}`}>{TYPE_LABELS[item.kind] ?? item.kind}</span>
                        <span className="debug-conf">{pct}%</span>
                        <span className="debug-time">{timeStr}</span>
                      </div>
                      <div className="debug-desc">{item.description || 'No description'}</div>
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
