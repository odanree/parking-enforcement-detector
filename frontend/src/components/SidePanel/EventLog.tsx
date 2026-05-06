import { useAppStore } from '../../store';
import type { AppEvent } from '../../types';
import { TYPE_LABELS } from '../../types';

function EventItem({ ev }: { ev: AppEvent }) {
  const openModal = useAppStore((s) => s.openModal);
  const pct     = Math.round((ev.confidence ?? 0) * 100);
  const timeStr = new Date(ev.timestamp * 1000).toLocaleTimeString([], {
    hour: '2-digit', minute: '2-digit', second: '2-digit',
  });

  return (
    <li className={`event-item ${ev.event_type}`}>
      <div className="event-header">
        <span className={`event-type ${ev.event_type}`}>{TYPE_LABELS[ev.event_type] ?? ev.event_type}</span>
        <span className="event-conf">{pct}%</span>
        <span className="event-time">{timeStr}</span>
      </div>
      <div className="event-desc" title={ev.description ?? ''}>{ev.description || 'No description'}</div>
      <div className="conf-bar"><div className="conf-fill" style={{ width: `${pct}%` }} /></div>
      {ev.snapshot_url && (
        <img
          className="event-thumb"
          src={ev.snapshot_url}
          alt="snapshot"
          loading="lazy"
          onClick={() => openModal(ev)}
        />
      )}
    </li>
  );
}

export function EventLog() {
  const events = useAppStore((s) => s.events);

  return (
    <div className="card events-card">
      <h2>
        Event Log{' '}
        <span className="event-count">{events.length}</span>
      </h2>
      <ul className="event-list">
        {events.length === 0
          ? <li className="event-empty">No alerts yet</li>
          : events.map((ev) => <EventItem key={`${ev.timestamp}-${ev.event_type}`} ev={ev} />)
        }
      </ul>
    </div>
  );
}
