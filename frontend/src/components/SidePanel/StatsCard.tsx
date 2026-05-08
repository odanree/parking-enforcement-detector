import { useAppStore } from '../../store';

function fmtUptime(secs: number) {
  const h = String(Math.floor(secs / 3600)).padStart(2, '0');
  const m = String(Math.floor((secs % 3600) / 60)).padStart(2, '0');
  const s = String(secs % 60).padStart(2, '0');
  return `${h}:${m}:${s}`;
}

function fmtTime(ts: number | null) {
  if (!ts) return '—';
  return new Date(ts * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function fmtCost(usd: number) {
  if (usd < 0.001) return `$${(usd * 1000).toFixed(3)}m`;
  return `$${usd.toFixed(4)}`;
}

export function StatsCard() {
  const stats = useAppStore((s) => s.stats);

  return (
    <div className="card">
      <h2>Stats</h2>
      <div className="stat-grid">
        <div className="stat-block">
          <span id="stat-chalking" className="stat-value">{stats?.total_chalking ?? 0}</span>
          <span className="stat-label">Chalking alerts</span>
        </div>
        <div className="stat-block wide">
          <span id="stat-uptime" className="stat-value mono">{fmtUptime(stats?.uptime_seconds ?? 0)}</span>
          <span className="stat-label">Uptime</span>
        </div>
        <div className="stat-block wide">
          <span className="stat-value mono small">{fmtTime(stats?.last_chalking ?? null)}</span>
          <span className="stat-label">Last chalking</span>
        </div>
        <div className="stat-block">
          <span className="stat-value">{stats?.vlm_calls ?? 0}</span>
          <span className="stat-label">VLM evals</span>
        </div>
        <div className="stat-block">
          <span className="stat-value mono">{(stats?.vlm_cost_usd ?? 0) === 0 ? '$0' : fmtCost(stats?.vlm_cost_usd ?? 0)}</span>
          <span className="stat-label">Claude cost</span>
        </div>
      </div>
    </div>
  );
}
