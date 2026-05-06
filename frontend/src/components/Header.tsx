import { useAppStore } from '../store';

export function Header() {
  const stats      = useAppStore((s) => s.stats);
  const wsStatus   = useAppStore((s) => s.wsStatus);
  const debugItems = useAppStore((s) => s.debugItems);
  const setDebugOpen = useAppStore((s) => s.setDebugOpen);
  const demo = stats?.demo_mode ?? false;

  return (
    <header>
      <span className="logo">&#x1F6D1; Parking Enforcement Detector</span>
      <div className="header-badges">
        <span id="badge-pipeline" className={`badge ${stats?.pipeline_running ? 'on' : 'off'}`}>
          &#x25CF; Pipeline
        </span>
        <span id="badge-sweep" className={`badge ${stats?.sweep_window_active ? 'on' : 'off'}`}>
          &#x25CF; Sweep Window
        </span>
        <span id="badge-ws" className={`badge badge-ws ${wsStatus}`}>
          &#x25CF; Stream
        </span>
        {!demo && (
          <button id="btn-debug-open" className="btn-debug-open" onClick={() => setDebugOpen(true)}>
            Debug
            {debugItems.length > 0 && (
              <span className="debug-badge visible">{debugItems.length}</span>
            )}
          </button>
        )}
      </div>
    </header>
  );
}
