import { useEffect, useRef, useState } from 'react';
import { useAppStore } from '../../store';
import type { AppEvent, Vote } from '../../types';
import { TYPE_LABELS } from '../../types';

function FrameAnimator({ frames, onClick }: { frames: string[]; onClick: () => void }) {
  const [idx, setIdx] = useState(0);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (frames.length <= 1) return;
    timerRef.current = setInterval(() => {
      setIdx((i) => (i + 1) % frames.length);
    }, 500); // 2 fps
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, [frames]);

  return (
    <img
      className="event-thumb"
      src={`data:image/jpeg;base64,${frames[idx]}`}
      alt={`frame ${idx + 1}/${frames.length}`}
      onClick={onClick}
    />
  );
}

async function postVote(camId: number, timestamp: number, vote: Vote) {
  await fetch(`/api/events/${camId}/vote?timestamp=${timestamp}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ vote }),
  });
}

function VoteButtons({ ev, camId }: { ev: AppEvent; camId: number }) {
  const setEvents  = useAppStore((s) => s.setEvents);
  const allEvents  = useAppStore((s) => s.events);
  const [pending, setPending] = useState(false);

  const cast = async (vote: Vote) => {
    if (pending) return;
    setPending(true);
    if (vote === 'archive') {
      await postVote(camId, ev.timestamp, 'archive');
      setEvents(allEvents.filter((e) => !(e.timestamp === ev.timestamp && e.camera === ev.camera)));
    } else {
      const next = ev.vote === vote ? null : vote;  // toggle off if same
      await postVote(camId, ev.timestamp, next);
      setEvents(allEvents.map((e) =>
        e.timestamp === ev.timestamp && e.camera === ev.camera ? { ...e, vote: next } : e
      ));
    }
    setPending(false);
  };

  return (
    <div className="vote-buttons">
      <button
        className={`vote-btn up   ${ev.vote === 'up'      ? 'active' : ''}`}
        title="True positive"
        onClick={() => cast('up')}
        disabled={pending}
      >👍</button>
      <button
        className={`vote-btn down ${ev.vote === 'down'    ? 'active' : ''}`}
        title="False positive"
        onClick={() => cast('down')}
        disabled={pending}
      >👎</button>
      <button
        className={`vote-btn arc  ${ev.vote === 'archive' ? 'active' : ''}`}
        title="Archive / reviewed"
        onClick={() => cast('archive')}
        disabled={pending}
      >📁</button>
    </div>
  );
}

function EventItem({ ev }: { ev: AppEvent }) {
  const openModal       = useAppStore((s) => s.openModal);
  const sessions        = useAppStore((s) => s.sessions);
  const sessionFilter   = useAppStore((s) => s.sessionFilter);
  const setSessionFilter = useAppStore((s) => s.setSessionFilter);
  const pct     = Math.round((ev.confidence ?? 0) * 100);
  const timeStr = new Date(ev.timestamp * 1000).toLocaleTimeString([], {
    hour: '2-digit', minute: '2-digit', second: '2-digit',
  });

  const hasFrames = ev.frames && ev.frames.length > 0;
  const camId = ev.camera ?? 0;
  const session = ev.session_id ? sessions.find((s) => s.session_id === ev.session_id) : null;
  const isFiltered = sessionFilter === ev.session_id;

  return (
    <li className={`event-item ${ev.event_type}`}>
      <div className="event-header">
        <span className={`event-type ${ev.event_type}`}>{TYPE_LABELS[ev.event_type] ?? ev.event_type}</span>
        {ev.camera != null && ev.camera > 0 && <span className="event-cam">Cam {ev.camera}</span>}
        {ev.track_id != null && <span className="event-cam" title="Track ID">#{ev.track_id}</span>}
        {session && (
          <span
            className={`event-session ${isFiltered ? 'active' : ''}`}
            title={`Session ${session.session_id} · ${session.event_count} events · ${session.track_ids.length} track IDs — click to filter`}
            onClick={() => setSessionFilter(isFiltered ? null : session.session_id)}
          >
            S:{session.session_id} ×{session.event_count}
          </span>
        )}
        <span className="event-conf">{pct}%</span>
        <span className="event-time">{timeStr}</span>
      </div>
      <div className="event-desc" title={ev.description ?? ''}>{ev.description || 'No description'}</div>
      <div className="conf-bar"><div className="conf-fill" style={{ width: `${pct}%` }} /></div>
      {hasFrames
        ? <FrameAnimator frames={ev.frames!} onClick={() => openModal(ev)} />
        : ev.snapshot_url && (
          <img
            className="event-thumb"
            src={ev.snapshot_url}
            alt="snapshot"
            loading="lazy"
            onClick={() => openModal(ev)}
          />
        )
      }
      <VoteButtons ev={ev} camId={camId} />
    </li>
  );
}

export function EventLog() {
  const events          = useAppStore((s) => s.events);
  const sessions        = useAppStore((s) => s.sessions);
  const sessionFilter   = useAppStore((s) => s.sessionFilter);
  const setSessionFilter = useAppStore((s) => s.setSessionFilter);

  const displayed = sessionFilter
    ? events.filter((ev) => ev.session_id === sessionFilter)
    : events;

  const filterSession = sessionFilter ? sessions.find((s) => s.session_id === sessionFilter) : null;

  return (
    <div className="card events-card">
      <h2>
        Event Log{' '}
        <span className="event-count">{displayed.length}</span>
      </h2>
      {filterSession && (
        <div className="session-filter-bar">
          <span>Session {filterSession.session_id} · {filterSession.event_count} events · {filterSession.duration_seconds}s</span>
          <button className="session-filter-clear" onClick={() => setSessionFilter(null)}>✕ All</button>
        </div>
      )}
      <ul className="event-list">
        {displayed.length === 0
          ? <li className="event-empty">No alerts yet</li>
          : displayed.map((ev) => <EventItem key={`${ev.timestamp}-${ev.event_type}`} ev={ev} />)
        }
      </ul>
    </div>
  );
}
