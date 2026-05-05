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
const elFps         = document.getElementById('fps-badge');

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

// ── Pause button ──────────────────────────────────────────────────────────────
const btnPause = document.getElementById('btn-pause');
let _paused = false;

btnPause.addEventListener('click', async () => {
  try {
    const res = await fetch('/api/pipeline/pause', { method: 'POST' });
    const data = await res.json();
    _paused = data.paused;
    syncPauseBtn();
  } catch (_) {}
});

function syncPauseBtn() {
  btnPause.textContent = _paused ? '▶ Resume' : '⏸ Pause';
  btnPause.classList.toggle('paused', _paused);
}

document.addEventListener('keydown', (e) => {
  if (e.code === 'Space' && !e.target.closest('input, textarea, select')) {
    e.preventDefault();
    btnPause.click();
  }
});

// ── Seek buttons ──────────────────────────────────────────────────────────────
document.querySelectorAll('.btn-seek').forEach(btn => {
  btn.addEventListener('click', async () => {
    const seconds = parseFloat(btn.dataset.seconds);
    try {
      await fetch('/api/playback/seek', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ seconds }),
      });
    } catch (_) {}
  });
});

// ── Playback speed ────────────────────────────────────────────────────────────
const speedBtns = document.querySelectorAll('.btn-speed');

function syncSpeedBtns(speed) {
  speedBtns.forEach(btn => {
    btn.classList.toggle('active', parseFloat(btn.dataset.speed) === speed);
  });
}

speedBtns.forEach(btn => {
  btn.addEventListener('click', async () => {
    const speed = parseFloat(btn.dataset.speed);
    try {
      const res  = await fetch('/api/playback/speed', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ speed }),
      });
      const data = await res.json();
      syncSpeedBtns(data.speed);
    } catch (_) {}
  });
});

// ── Motion detect toggle ──────────────────────────────────────────────────────
const btnMotion = document.getElementById('btn-motion');
let _motionEnabled = false;

btnMotion.addEventListener('click', async () => {
  try {
    const res = await fetch('/api/motion/toggle', { method: 'POST' });
    const data = await res.json();
    _motionEnabled = data.motion_detect_enabled;
    syncMotionBtn();
  } catch (_) {}
});

function syncMotionBtn() {
  btnMotion.textContent = _motionEnabled ? 'Motion Detect ON' : 'Motion Detect';
  btnMotion.classList.toggle('active', _motionEnabled);
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

    if (_paused !== s.paused) {
      _paused = s.paused;
      syncPauseBtn();
    }
    if (_motionEnabled !== s.motion_detect_enabled) {
      _motionEnabled = s.motion_detect_enabled;
      syncMotionBtn();
    }
    if (s.playback_speed  !== undefined) syncSpeedBtns(s.playback_speed);
    if (s.privacy_mode    !== undefined) syncPrivacy(s.privacy_mode);
    if (s.fps !== undefined) elFps.textContent = `${Math.round(s.fps)} fps`;
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

  const thumbHtml = ev.snapshot_url
    ? `<img class="event-thumb" src="${ev.snapshot_url}" alt="snapshot" loading="lazy">`
    : '';

  li.innerHTML = `
    <div class="event-header">
      <span class="event-type ${ev.event_type}">${{chalking:'Chalking',sweeper:'Sweeper',pe_vehicle:'PE Vehicle'}[ev.event_type] ?? ev.event_type}</span>
      <span class="event-conf">${pct}%</span>
      <span class="event-time">${timeStr}</span>
    </div>
    <div class="event-desc" title="${ev.description ?? ''}">${ev.description || 'No description'}</div>
    <div class="conf-bar"><div class="conf-fill" style="width:${pct}%"></div></div>
    ${thumbHtml}
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
const overlay      = document.getElementById('zone-overlay');
const octx         = overlay.getContext('2d');
const btnEdit      = document.getElementById('btn-edit-zone');
const zoneControls = document.getElementById('zone-controls');
const btnSave      = document.getElementById('btn-save-zone');
const btnCancel    = document.getElementById('btn-cancel-zone');
const btnUndo      = document.getElementById('btn-undo-zone');
const btnReset     = document.getElementById('btn-reset-zone');

const W = overlay.width;   // 1280
const H = overlay.height;  // 720
const HIT_R      = 18;     // vertex hit radius
const EDGE_HIT_R = 14;     // edge midpoint hit radius

let editing   = false;
let points    = [];   // [[x,y], ...] in frame coordinates
let saved     = [];   // snapshot of points before edit session
let history   = [];   // undo stack — each entry is a snapshot of points
let dragIdx   = -1;   // vertex being dragged (-1 = none)
let dragEdge  = -1;   // edge being dragged, index i = edge points[i]→points[i+1] (-1 = none)
let _dragPrev = null; // [cx,cy] at last mousemove
let _didDrag  = false;

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

// Returns edge index i (edge from points[i] to points[(i+1)%n]) or -1
function hitEdge(cx, cy) {
  const n = points.length;
  for (let i = 0; i < n; i++) {
    const [ax, ay] = points[i];
    const [bx, by] = points[(i + 1) % n];
    const mx = (ax + bx) / 2;
    const my = (ay + by) / 2;
    if (Math.sqrt((mx - cx) ** 2 + (my - cy) ** 2) < EDGE_HIT_R) return i;
  }
  return -1;
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

  // Zone label — top-left vertex of the bounding box
  const xs = points.map(p => p[0]);
  const ys = points.map(p => p[1]);
  const labelX = Math.min(...xs);
  const labelY = Math.min(...ys) - 6;
  octx.font = 'bold 11px Inter, system-ui, sans-serif';
  octx.fillStyle = editing ? '#00e676' : '#00e676aa';
  octx.fillText('STREET ZONE', labelX, Math.max(labelY, 12));

  if (!editing) return;

  const n = points.length;

  // Edge midpoint handles (blue diamonds) — drag to slide the whole edge
  for (let i = 0; i < n; i++) {
    const [ax, ay] = points[i];
    const [bx, by] = points[(i + 1) % n];
    const mx = (ax + bx) / 2;
    const my = (ay + by) / 2;
    const s = 14;
    octx.beginPath();
    octx.moveTo(mx,     my - s);
    octx.lineTo(mx + s, my);
    octx.lineTo(mx,     my + s);
    octx.lineTo(mx - s, my);
    octx.closePath();
    octx.fillStyle   = i === dragEdge ? '#ffffff' : '#00b0ff';
    octx.strokeStyle = '#000';
    octx.lineWidth   = 2;
    octx.fill();
    octx.stroke();
  }

  // Vertex handles (green circles) — drag to move a single corner
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

function pushHistory() {
  history.push(points.map(p => [...p]));
  btnUndo.disabled = false;
}

function applyUndo() {
  if (!history.length) return;
  points = history.pop();
  btnUndo.disabled = history.length === 0;
  drawOverlay();
}

function enterEditMode() {
  editing = true;
  saved   = points.map(p => [...p]);
  history = [];
  btnUndo.disabled = true;
  overlay.classList.add('editing');
  btnEdit.classList.add('active');
  btnEdit.textContent = 'Editing…';
  zoneControls.classList.remove('hidden');
  drawOverlay();
}

function exitEditMode() {
  editing = false;
  dragIdx = -1;
  history = [];
  btnUndo.disabled = true;
  overlay.style.cursor = '';
  overlay.classList.remove('editing');
  btnEdit.classList.remove('active');
  btnEdit.textContent = 'Edit Zone';
  zoneControls.classList.add('hidden');
  drawOverlay();
}

btnEdit.addEventListener('click', () => {
  if (!editing) enterEditMode();
});

btnUndo.addEventListener('click', applyUndo);

btnReset.addEventListener('click', () => {
  pushHistory();
  points = [[0, 0], [W, 0], [W, H], [0, H]];
  drawOverlay();
});

btnCancel.addEventListener('click', () => {
  points = saved.map(p => [...p]);
  exitEditMode();
});

document.addEventListener('keydown', (e) => {
  if (editing && (e.key === 'z' || e.key === 'Z') && (e.ctrlKey || e.metaKey)) {
    e.preventDefault();
    applyUndo();
  }
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
  if (!editing || _didDrag) return;
  const [cx, cy] = toFrame(e);
  if (hitIndex(cx, cy) !== -1) return;
  if (hitEdge(cx, cy) !== -1) return;
  pushHistory();
  points.push([cx, cy]);
  drawOverlay();
});

// Right-click → remove nearest vertex
overlay.addEventListener('contextmenu', (e) => {
  e.preventDefault();
  if (!editing) return;
  const [cx, cy] = toFrame(e);
  const idx = hitIndex(cx, cy);
  if (idx !== -1) {
    pushHistory();
    points.splice(idx, 1);
    drawOverlay();
  }
});

overlay.addEventListener('mousedown', (e) => {
  if (!editing || e.button !== 0) return;
  _didDrag = false;
  const [cx, cy] = toFrame(e);
  // Vertex takes priority over edge midpoint
  dragIdx = hitIndex(cx, cy);
  if (dragIdx !== -1) {
    _dragPrev = [cx, cy];
    overlay.style.cursor = 'grabbing';
    return;
  }
  dragEdge = hitEdge(cx, cy);
  if (dragEdge !== -1) {
    _dragPrev = [cx, cy];
    overlay.style.cursor = 'grabbing';
    pushHistory();
  }
});

overlay.addEventListener('mousemove', (e) => {
  if (!editing) return;
  const [cx, cy] = toFrame(e);
  if (dragIdx !== -1) {
    points[dragIdx] = [cx, cy];
    drawOverlay();
    _dragPrev = [cx, cy];
    _didDrag = true;
  } else if (dragEdge !== -1) {
    const [px, py] = _dragPrev;
    const dx = cx - px, dy = cy - py;
    const n = points.length;
    const j = (dragEdge + 1) % n;
    points[dragEdge] = [points[dragEdge][0] + dx, points[dragEdge][1] + dy];
    points[j]        = [points[j][0]        + dx, points[j][1]        + dy];
    drawOverlay();
    _dragPrev = [cx, cy];
    _didDrag = true;
  } else {
    if (hitIndex(cx, cy) !== -1)     overlay.style.cursor = 'grab';
    else if (hitEdge(cx, cy) !== -1) overlay.style.cursor = 'move';
    else                             overlay.style.cursor = 'crosshair';
  }
});

overlay.addEventListener('mouseup', () => {
  dragIdx  = -1;
  dragEdge = -1;
  _dragPrev = null;
  overlay.style.cursor = 'crosshair';
});

overlay.addEventListener('mouseleave', () => {
  dragIdx  = -1;
  dragEdge = -1;
  _dragPrev = null;
  overlay.style.cursor = '';
});

// ── Privacy overlay ───────────────────────────────────────────────────────────
const privacyCanvas   = document.getElementById('privacy-overlay');
const pctx            = privacyCanvas.getContext('2d');
const btnPrivacy      = document.getElementById('btn-privacy');
const privacyControls = document.getElementById('privacy-controls');
const btnSavePrivacy  = document.getElementById('btn-save-privacy');
const btnCancelPrivacy = document.getElementById('btn-cancel-privacy');

let _privacyMode    = false;
let _privacyEditing = false;
let _privacyRegions = [];   // [[x1,y1,x2,y2], ...]
let _privacyDrag    = null; // {x0,y0,x1,y1} while drawing

// Load saved regions from server
(async () => {
  try {
    const r = await fetch('/api/privacy/regions');
    const d = await r.json();
    _privacyRegions = d.regions || [];
    drawPrivacy();
  } catch (_) {}
})();

function drawPrivacy() {
  pctx.clearRect(0, 0, privacyCanvas.width, privacyCanvas.height);
  if (!_privacyMode && !_privacyEditing) return;

  for (const [x1, y1, x2, y2] of _privacyRegions) {
    if (_privacyEditing) {
      pctx.strokeStyle = '#b388ff';
      pctx.lineWidth = 2;
      pctx.setLineDash([6, 3]);
      pctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      pctx.setLineDash([]);
      pctx.fillStyle = 'rgba(179,136,255,0.15)';
      pctx.fillRect(x1, y1, x2 - x1, y2 - y1);
    } else {
      pctx.fillStyle = '#000';
      pctx.fillRect(x1, y1, x2 - x1, y2 - y1);
    }
  }

  if (_privacyDrag) {
    const { x0, y0, x1, y1 } = _privacyDrag;
    pctx.strokeStyle = '#b388ff';
    pctx.lineWidth = 2;
    pctx.setLineDash([4, 3]);
    pctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
    pctx.setLineDash([]);
    pctx.fillStyle = 'rgba(179,136,255,0.2)';
    pctx.fillRect(x0, y0, x1 - x0, y1 - y0);
  }
}

function toPrivacyFrame(e) {
  const r = privacyCanvas.getBoundingClientRect();
  return [
    Math.round((e.clientX - r.left) * (privacyCanvas.width  / r.width)),
    Math.round((e.clientY - r.top)  * (privacyCanvas.height / r.height)),
  ];
}

btnPrivacy.addEventListener('click', async () => {
  if (!_privacyEditing) {
    // Simple toggle privacy mode
    try {
      const r = await fetch('/api/privacy/toggle', { method: 'POST' });
      const d = await r.json();
      _privacyMode = d.privacy_mode;
      btnPrivacy.classList.toggle('active', _privacyMode);
      drawPrivacy();
    } catch (_) {}
  }
});

btnPrivacy.addEventListener('contextmenu', (e) => {
  e.preventDefault();
  // Right-click Privacy button → enter edit mode
  _privacyEditing = true;
  privacyCanvas.classList.add('editing');
  privacyControls.classList.remove('hidden');
  btnPrivacy.textContent = '🚫 Editing…';
  drawPrivacy();
});

btnSavePrivacy.addEventListener('click', async () => {
  try {
    await fetch('/api/privacy/regions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ regions: _privacyRegions }),
    });
  } catch (_) {}
  exitPrivacyEdit();
});

btnCancelPrivacy.addEventListener('click', () => {
  // Reload from server to discard unsaved changes
  fetch('/api/privacy/regions').then(r => r.json()).then(d => {
    _privacyRegions = d.regions || [];
    exitPrivacyEdit();
  });
});

function exitPrivacyEdit() {
  _privacyEditing = false;
  privacyCanvas.classList.remove('editing');
  privacyCanvas.style.cursor = '';
  privacyControls.classList.add('hidden');
  btnPrivacy.textContent = '🚫 Privacy';
  drawPrivacy();
}

privacyCanvas.addEventListener('mousedown', (e) => {
  if (!_privacyEditing || e.button !== 0) return;
  const [x, y] = toPrivacyFrame(e);
  _privacyDrag = { x0: x, y0: y, x1: x, y1: y };
});

privacyCanvas.addEventListener('mousemove', (e) => {
  if (!_privacyEditing || !_privacyDrag) return;
  const [x, y] = toPrivacyFrame(e);
  _privacyDrag.x1 = x;
  _privacyDrag.y1 = y;
  drawPrivacy();
});

privacyCanvas.addEventListener('mouseup', () => {
  if (!_privacyEditing || !_privacyDrag) return;
  const { x0, y0, x1, y1 } = _privacyDrag;
  const rx1 = Math.min(x0, x1), ry1 = Math.min(y0, y1);
  const rx2 = Math.max(x0, x1), ry2 = Math.max(y0, y1);
  if (rx2 - rx1 > 8 && ry2 - ry1 > 8) {
    _privacyRegions.push([rx1, ry1, rx2, ry2]);
  }
  _privacyDrag = null;
  drawPrivacy();
});

privacyCanvas.addEventListener('contextmenu', (e) => {
  e.preventDefault();
  if (!_privacyEditing) return;
  const [cx, cy] = toPrivacyFrame(e);
  _privacyRegions = _privacyRegions.filter(([x1, y1, x2, y2]) =>
    !(cx >= x1 && cx <= x2 && cy >= y1 && cy <= y2)
  );
  drawPrivacy();
});

// Sync privacy mode from stats poll
function syncPrivacy(mode) {
  if (_privacyMode !== mode) {
    _privacyMode = mode;
    btnPrivacy.classList.toggle('active', mode);
    drawPrivacy();
  }
}

// ── Snapshot lightbox ─────────────────────────────────────────────────────────
const lightbox      = document.getElementById('lightbox');
const lightboxImg   = document.getElementById('lightbox-img');
const lightboxClose = document.getElementById('lightbox-close');

function openLightbox(src) {
  lightboxImg.src = src;
  lightbox.classList.remove('hidden');
}

function closeLightbox() {
  lightbox.classList.add('hidden');
  lightboxImg.src = '';
}

lightboxClose.addEventListener('click', closeLightbox);
lightbox.addEventListener('click', (e) => { if (e.target === lightbox) closeLightbox(); });
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeLightbox(); });

// Delegate thumbnail clicks from the event list
eventList.addEventListener('click', (e) => {
  const thumb = e.target.closest('.event-thumb');
  if (thumb) openLightbox(thumb.src);
});
