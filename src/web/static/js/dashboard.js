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
    if (newest <= lastEventTs) return;
    lastEventTs = newest;

    const empty = eventList.querySelector('.event-empty');
    if (empty) empty.remove();

    const newOnes = events.filter(e => e.timestamp > (lastEventTs - 0.001));
    newOnes.forEach(ev => {
      eventList.insertBefore(buildEventItem(ev), eventList.firstChild);
    });

    while (eventList.children.length > 30) eventList.removeChild(eventList.lastChild);
    eventCount.textContent = events.length;
  } catch (_) { /* ignore */ }
}

fetchEvents();
setInterval(fetchEvents, 3000);

// ── Zone editor ───────────────────────────────────────────────────────────────
const overlay     = document.getElementById('zone-overlay');
const octx        = overlay.getContext('2d');
const btnEdit     = document.getElementById('btn-edit-zone');
const zoneControls = document.getElementById('zone-controls');
const btnSave     = document.getElementById('btn-save-zone');
const btnCancel   = document.getElementById('btn-cancel-zone');

const W = overlay.width;   // 1280
const H = overlay.height;  // 720
const HIT_R = 18;          // vertex hit radius in canvas pixels

let editing   = false;
let points    = [];   // [[x,y], ...] in frame coordinates
let saved     = [];   // snapshot of points before edit session
let dragIdx   = -1;

// Convert a MouseEvent to canvas (frame) coordinates
function toFrame(e) {
  const r = overlay.getBoundingClientRect();
  return [
    Math.round((e.clientX - r.left) * (W / r.width)),
    Math.round((e.clientY - r.top)  * (H / r.height)),
  ];
}

function hitIndex(cx, cy) {
  return points.findIndex(([px, py]) => {
    const dx = px - cx, dy = py - cy;
    return Math.sqrt(dx * dx + dy * dy) < HIT_R;
  });
}

function drawOverlay() {
  octx.clearRect(0, 0, W, H);
  if (!points.length) return;

  // Polygon fill
  octx.beginPath();
  octx.moveTo(points[0][0], points[0][1]);
  for (let i = 1; i < points.length; i++) octx.lineTo(points[i][0], points[i][1]);
  octx.closePath();
  octx.fillStyle = editing ? 'rgba(0,230,118,0.10)' : 'rgba(0,230,118,0.06)';
  octx.fill();

  // Polygon outline
  octx.strokeStyle = editing ? '#00e676' : '#00e676aa';
  octx.lineWidth   = editing ? 2 : 1.5;
  octx.setLineDash(editing ? [] : [6, 4]);
  octx.stroke();
  octx.setLineDash([]);

  if (!editing) return;

  // Vertex handles
  points.forEach(([px, py], i) => {
    octx.beginPath();
    octx.arc(px, py, 7, 0, Math.PI * 2);
    octx.fillStyle   = i === dragIdx ? '#ffffff' : '#00e676';
    octx.strokeStyle = '#000';
    octx.lineWidth   = 1.5;
    octx.fill();
    octx.stroke();
  });
}

// Load current zone from API
async function loadZone() {
  try {
    const res  = await fetch('/api/zone');
    const data = await res.json();
    points = data.polygon || [];
    drawOverlay();
  } catch (_) {}
}

loadZone();

function enterEditMode() {
  editing = true;
  saved   = points.map(p => [...p]);
  overlay.classList.add('editing');
  btnEdit.classList.add('active');
  btnEdit.textContent = 'Editing…';
  zoneControls.classList.remove('hidden');
  drawOverlay();
}

function exitEditMode() {
  editing = false;
  dragIdx = -1;
  overlay.classList.remove('editing');
  btnEdit.classList.remove('active');
  btnEdit.textContent = 'Edit Zone';
  zoneControls.classList.add('hidden');
  drawOverlay();
}

btnEdit.addEventListener('click', () => {
  if (!editing) enterEditMode();
});

btnCancel.addEventListener('click', () => {
  points = saved.map(p => [...p]);
  exitEditMode();
});

btnSave.addEventListener('click', async () => {
  if (points.length < 3) {
    alert('Zone needs at least 3 points.');
    return;
  }
  try {
    const res = await fetch('/api/zone', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ polygon: points }),
    });
    if (!res.ok) throw new Error(await res.text());
    exitEditMode();
  } catch (err) {
    alert('Save failed: ' + err.message);
  }
});

// Click → add point (unless we just finished a drag)
overlay.addEventListener('click', (e) => {
  if (!editing || dragIdx !== -1) return;
  const [cx, cy] = toFrame(e);
  if (hitIndex(cx, cy) !== -1) return;  // clicked an existing vertex
  points.push([cx, cy]);
  drawOverlay();
});

// Right-click → remove nearest vertex
overlay.addEventListener('contextmenu', (e) => {
  e.preventDefault();
  if (!editing) return;
  const [cx, cy] = toFrame(e);
  const idx = hitIndex(cx, cy);
  if (idx !== -1) { points.splice(idx, 1); drawOverlay(); }
});

overlay.addEventListener('mousedown', (e) => {
  if (!editing || e.button !== 0) return;
  const [cx, cy] = toFrame(e);
  dragIdx = hitIndex(cx, cy);
});

overlay.addEventListener('mousemove', (e) => {
  if (!editing || dragIdx === -1) return;
  const [cx, cy] = toFrame(e);
  points[dragIdx] = [cx, cy];
  drawOverlay();
});

overlay.addEventListener('mouseup', () => { dragIdx = -1; });
overlay.addEventListener('mouseleave', () => { dragIdx = -1; });
