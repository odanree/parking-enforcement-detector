'use strict';

// ── Canvas video stream ───────────────────────────────────────────────────────
const canvas    = document.getElementById('feed');
const ctx       = canvas.getContext('2d');
const noSignal  = document.getElementById('no-signal');
const badgeWs   = document.getElementById('badge-ws');

let wsActive = false;

function connectVideoStream() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${proto}://${location.host}/ws/video`);
  ws.binaryType = 'blob';

  ws.onopen = () => {
    badgeWs.textContent = '● Stream';
    badgeWs.className = 'badge badge-ws connected';
  };

  ws.onmessage = (event) => {
    if (!wsActive) {
      wsActive = true;
      noSignal.classList.add('hidden');
    }
    const url = URL.createObjectURL(event.data);
    const img = new Image();
    img.onload = () => {
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      URL.revokeObjectURL(url);
    };
    img.src = url;
  };

  ws.onclose = () => {
    wsActive = false;
    badgeWs.textContent = '● Stream';
    badgeWs.className = 'badge badge-ws disconnected';
    noSignal.classList.remove('hidden');
    // Auto-reconnect after 3 s
    setTimeout(connectVideoStream, 3000);
  };

  ws.onerror = () => ws.close();
}

connectVideoStream();

// ── Stats polling ─────────────────────────────────────────────────────────────
const elPipeline    = document.getElementById('badge-pipeline');
const elSweep       = document.getElementById('badge-sweep');
const elChalking    = document.getElementById('stat-chalking');
const elSweeper     = document.getElementById('stat-sweeper');
const elUptime      = document.getElementById('stat-uptime');
const elLastChalk   = document.getElementById('stat-last-chalking');
const elLastSweeper = document.getElementById('stat-last-sweeper');

function fmtUptime(secs) {
  const h = String(Math.floor(secs / 3600)).padStart(2, '0');
  const m = String(Math.floor((secs % 3600) / 60)).padStart(2, '0');
  const s = String(secs % 60).padStart(2, '0');
  return `${h}:${m}:${s}`;
}

function fmtTime(ts) {
  if (!ts) return '—';
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

async function fetchStats() {
  try {
    const res = await fetch('/api/stats');
    const s   = await res.json();

    elPipeline.textContent = `● Pipeline`;
    elPipeline.className = `badge ${s.pipeline_running ? 'on' : 'off'}`;

    elSweep.textContent = `● Sweep Window`;
    elSweep.className = `badge ${s.sweep_window_active ? 'on' : 'off'}`;

    elChalking.textContent    = s.total_chalking;
    elSweeper.textContent     = s.total_sweeper;
    elUptime.textContent      = fmtUptime(s.uptime_seconds);
    elLastChalk.textContent   = fmtTime(s.last_chalking);
    elLastSweeper.textContent = fmtTime(s.last_sweeper);
  } catch (_) { /* ignore transient fetch errors */ }
}

fetchStats();
setInterval(fetchStats, 2000);

// ── Event log ─────────────────────────────────────────────────────────────────
const eventList  = document.getElementById('event-list');
const eventCount = document.getElementById('event-count');
let lastEventTs  = 0;

function buildEventItem(ev) {
  const li   = document.createElement('li');
  li.className = `event-item ${ev.event_type}`;

  const pct = Math.round((ev.confidence ?? 0) * 100);
  const timeStr = new Date(ev.timestamp * 1000).toLocaleTimeString([], {
    hour: '2-digit', minute: '2-digit', second: '2-digit'
  });

  li.innerHTML = `
    <div class="event-header">
      <span class="event-type ${ev.event_type}">${ev.event_type === 'chalking' ? 'Chalking' : 'Sweeper'}</span>
      <span class="event-conf">${pct}%</span>
      <span class="event-time">${timeStr}</span>
    </div>
    <div class="event-desc" title="${ev.description ?? ''}">${ev.description || 'No description'}</div>
    <div class="conf-bar"><div class="conf-fill" style="width:${pct}%"></div></div>
  `;
  return li;
}

async function fetchEvents() {
  try {
    const res    = await fetch('/api/events');
    const events = await res.json();

    if (!events.length) return;

    const newest = events[0].timestamp;
    if (newest <= lastEventTs) return;   // nothing new
    lastEventTs = newest;

    // Remove placeholder
    const empty = eventList.querySelector('.event-empty');
    if (empty) empty.remove();

    // Prepend only new events (those newer than the previous known newest)
    const newOnes = events.filter(e => e.timestamp > (lastEventTs - 0.001));
    newOnes.forEach(ev => {
      eventList.insertBefore(buildEventItem(ev), eventList.firstChild);
    });

    // Trim list to 30 items
    while (eventList.children.length > 30) {
      eventList.removeChild(eventList.lastChild);
    }

    eventCount.textContent = events.length;
  } catch (_) { /* ignore */ }
}

fetchEvents();
setInterval(fetchEvents, 3000);
