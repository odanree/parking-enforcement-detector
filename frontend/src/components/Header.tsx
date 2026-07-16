import { useAppStore } from '../store';

export function Header() {
  const stats             = useAppStore((s) => s.stats);
  const wsStatuses        = useAppStore((s) => s.wsStatuses);
  const debugItems        = useAppStore((s) => s.debugItems);
  const setDebugOpen      = useAppStore((s) => s.setDebugOpen);
  const setKanbanOpen     = useAppStore((s) => s.setKanbanOpen);
  const setAdminOpen      = useAppStore((s) => s.setAdminOpen);
  const setTimelineOpen   = useAppStore((s) => s.setTimelineOpen);
  const demo = stats?.demo_mode ?? false;

  return (
    <header>
      <span className="logo">&#x1F6D1; Parking Enforcement Detector</span>
      <div className="header-badges">
        <span id="badge-pipeline" className={`badge ${stats?.pipeline_running ? 'on' : 'off'}`}>
          &#x25CF; Pipeline
        </span>
        <span className={`badge badge-ws ${wsStatuses[0]}`}>
          &#x25CF; Cam 0
        </span>
        <span className={`badge badge-ws ${wsStatuses[1]}`}>
          &#x25CF; Cam 1
        </span>
        {!demo && (
          <>
            <button className="btn-debug-open" onClick={() => setKanbanOpen(true)}>
              Pipeline
            </button>
            <button className="btn-debug-open" onClick={() => setAdminOpen(true)}>
              Dataset
            </button>
            <button className="btn-debug-open" onClick={() => setTimelineOpen(true)}>
              Timeline
            </button>
            <button id="btn-debug-open" className="btn-debug-open" onClick={() => setDebugOpen(true)}>
              Debug
              {debugItems.length > 0 && (
                <span className="debug-badge visible">{debugItems.length}</span>
              )}
            </button>
          </>
        )}
      </div>
    </header>
  );
}
