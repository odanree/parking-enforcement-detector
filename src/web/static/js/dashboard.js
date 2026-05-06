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
const _eventDataMap = new Map(); // snapshot_url (relative) → event object

function buildEventItem(ev) {
  const li   = document.createElement('li');
  li.className = `event-item ${ev.event_type}`;

  const pct = Math.round((ev.confidence ?? 0) * 100);
  const timeStr = new Date(ev.timestamp * 1000).toLocaleTimeString([], {
    hour: '2-digit', minute: '2-digit', second: '2-digit'
  });

  if (ev.snapshot_url) _eventDataMap.set(ev.snapshot_url, ev);

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

    const prevLastTs = lastEventTs;
    lastEventTs = newest;

    const empty = eventList.querySelector('.event-empty');
    if (empty) empty.remove();

    // prevLastTs === 0 means first load — render all events from server.
    // Otherwise only add events strictly newer than what's already shown.
    const newOnes = prevLastTs === 0 ? events : events.filter(e => e.timestamp > prevLastTs);
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
const privacyCanvas    = document.getElementById('privacy-overlay');
const pctx             = privacyCanvas.getContext('2d');
const btnPrivacy       = document.getElementById('btn-privacy');
const btnEditPrivacy   = document.getElementById('btn-edit-privacy');
const privacyControls  = document.getElementById('privacy-controls');
const btnSavePrivacy   = document.getElementById('btn-save-privacy');
const btnCancelPrivacy = document.getElementById('btn-cancel-privacy');
const btnClearPrivacy  = document.getElementById('btn-clear-privacy');

let _privacyMode    = false;
let _privacyEditing = false;
let _privacyRegions = [];       // saved [[x1,y1,x2,y2], ...]
let _privacyDraft   = [];       // unsaved copy while editing
let _privacyDrag    = null;     // {x0,y0,x1,y1} while drawing

// Load saved regions from server on startup
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
  const regions = _privacyEditing ? _privacyDraft : _privacyRegions;

  for (const [x1, y1, x2, y2] of regions) {
    if (_privacyEditing) {
      pctx.fillStyle = 'rgba(179,136,255,0.18)';
      pctx.fillRect(x1, y1, x2 - x1, y2 - y1);
      pctx.strokeStyle = '#b388ff';
      pctx.lineWidth = 2;
      pctx.setLineDash([5, 3]);
      pctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      pctx.setLineDash([]);
    } else if (_privacyMode) {
      pctx.fillStyle = '#000';
      pctx.fillRect(x1, y1, x2 - x1, y2 - y1);
    }
  }

  // Live drag preview
  if (_privacyDrag) {
    const { x0, y0, x1, y1 } = _privacyDrag;
    pctx.fillStyle = 'rgba(179,136,255,0.25)';
    pctx.fillRect(x0, y0, x1 - x0, y1 - y0);
    pctx.strokeStyle = '#b388ff';
    pctx.lineWidth = 2;
    pctx.setLineDash([4, 3]);
    pctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
    pctx.setLineDash([]);
  }
}

function toPrivacyFrame(e) {
  const r = privacyCanvas.getBoundingClientRect();
  return [
    Math.round((e.clientX - r.left) * (privacyCanvas.width  / r.width)),
    Math.round((e.clientY - r.top)  * (privacyCanvas.height / r.height)),
  ];
}

// Privacy toggle — simple on/off
btnPrivacy.addEventListener('click', async () => {
  try {
    const r = await fetch('/api/privacy/toggle', { method: 'POST' });
    const d = await r.json();
    _privacyMode = d.privacy_mode;
    btnPrivacy.classList.toggle('active', _privacyMode);
    drawPrivacy();
  } catch (_) {}
});

// Edit Regions — opens the draw toolbar
btnEditPrivacy.addEventListener('click', () => {
  _privacyDraft   = _privacyRegions.map(r => [...r]);
  _privacyEditing = true;
  privacyCanvas.classList.add('editing');
  privacyControls.classList.remove('hidden');
  btnEditPrivacy.classList.add('active');
  drawPrivacy();
});

btnClearPrivacy.addEventListener('click', () => {
  _privacyDraft = [];
  drawPrivacy();
});

btnSavePrivacy.addEventListener('click', async () => {
  try {
    await fetch('/api/privacy/regions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ regions: _privacyDraft }),
    });
    _privacyRegions = _privacyDraft;
  } catch (_) {}
  exitPrivacyEdit();
});

btnCancelPrivacy.addEventListener('click', exitPrivacyEdit);

function exitPrivacyEdit() {
  _privacyEditing = false;
  _privacyDrag    = null;
  privacyCanvas.classList.remove('editing');
  privacyCanvas.style.cursor = '';
  privacyControls.classList.add('hidden');
  btnEditPrivacy.classList.remove('active');
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
  if (rx2 - rx1 > 8 && ry2 - ry1 > 8) _privacyDraft.push([rx1, ry1, rx2, ry2]);
  _privacyDrag = null;
  drawPrivacy();
});

// Right-click a drawn box to delete it
privacyCanvas.addEventListener('contextmenu', (e) => {
  e.preventDefault();
  if (!_privacyEditing) return;
  const [cx, cy] = toPrivacyFrame(e);
  _privacyDraft = _privacyDraft.filter(([x1, y1, x2, y2]) =>
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

// ── VLM processing queue ──────────────────────────────────────────────────────
const processingCard = document.getElementById('processing-card');
const vlmJobList     = document.getElementById('vlm-job-list');
const KIND_LABELS    = { chalking: 'Chalking', sweeper: 'Sweeper', pe_vehicle: 'PE Vehicle' };

function renderPending(jobs) {
  // Remove jobs no longer returned by the server
  const currentIds = new Set(jobs.map(j => j.id));
  vlmJobList.querySelectorAll('.vlm-job').forEach(el => {
    if (!currentIds.has(el.dataset.id)) el.remove();
  });

  for (const job of jobs) {
    const done = job.completed_at != null;
    let li = vlmJobList.querySelector(`[data-id="${job.id}"]`);

    if (!li) {
      li = document.createElement('li');
      li.className = 'vlm-job';
      li.dataset.id = job.id;
      li.dataset.submittedAt = job.submitted_at;
      const sampleLabel = job.sample_num > 1 ? ` · Sample #${job.sample_num}` : '';
      li.innerHTML = `
        <div class="vlm-thumb-wrap">
          ${job.thumbnail ? `<img src="data:image/jpeg;base64,${job.thumbnail}" alt="">` : ''}
          <div class="vlm-spinner"><div class="vlm-spinner-ring"></div></div>
        </div>
        <div class="vlm-job-meta">
          <div class="vlm-kind ${job.kind}">${KIND_LABELS[job.kind] ?? job.kind}${sampleLabel}</div>
          <div class="vlm-elapsed">0.0s</div>
          <div class="vlm-result"></div>
        </div>
      `;
      vlmJobList.appendChild(li);
    }

    // Update result state once completed
    if (done) {
      const spinner = li.querySelector('.vlm-spinner');
      const resultEl = li.querySelector('.vlm-result');
      if (spinner && !spinner.dataset.resolved) {
        spinner.dataset.resolved = '1';
        spinner.innerHTML = job.detected
          ? '<span style="font-size:16px">✓</span>'
          : '<span style="font-size:16px;color:#888">✗</span>';
        spinner.style.background = job.detected ? 'rgba(0,200,80,0.55)' : 'rgba(80,80,80,0.55)';
      }
      if (resultEl && !resultEl.textContent) {
        resultEl.textContent = job.detected ? 'Detected' : 'Not detected';
        resultEl.style.color = job.detected ? 'var(--green)' : 'var(--muted)';
        resultEl.style.fontSize = '11px';
        resultEl.style.marginTop = '2px';
      }
    }
  }

  if (vlmJobList.children.length) {
    processingCard.classList.remove('hidden');
  } else {
    processingCard.classList.add('hidden');
  }
}

// Tick elapsed times independently of fetch
setInterval(() => {
  vlmJobList.querySelectorAll('.vlm-job').forEach(li => {
    const el = li.querySelector('.vlm-elapsed');
    if (el) el.textContent = (Date.now() / 1000 - parseFloat(li.dataset.submittedAt)).toFixed(1) + 's';
  });
}, 100);

async function pollPending() {
  try {
    const res = await fetch('/api/pending');
    const d   = await res.json();
    renderPending(d.jobs || []);
  } catch (_) {}
}

pollPending();
setInterval(pollPending, 1000);

// ── Debug drawer ──────────────────────────────────────────────────────────────
const debugDrawer   = document.getElementById('debug-drawer');
const debugBackdrop = document.getElementById('debug-backdrop');
const debugList     = document.getElementById('debug-list');
const debugCount    = document.getElementById('debug-count');
const debugBadge    = document.getElementById('debug-badge');
const btnDebugOpen  = document.getElementById('btn-debug-open');
const btnDebugClose = document.getElementById('btn-debug-close');
const btnDebugClear = document.getElementById('btn-debug-clear');

const DEBUG_KIND_LABELS = { chalking: 'Chalking', sweeper: 'Sweeper', pe_vehicle: 'PE Vehicle' };

let _debugOpen = false;
let _debugItems = [];

function openDebugDrawer()  { _debugOpen = true;  debugDrawer.classList.add('open');  debugBackdrop.classList.add('open'); }
function closeDebugDrawer() { _debugOpen = false; debugDrawer.classList.remove('open'); debugBackdrop.classList.remove('open'); }

btnDebugOpen.addEventListener('click', () => { openDebugDrawer(); fetchDebug(); });
btnDebugClose.addEventListener('click', closeDebugDrawer);
debugBackdrop.addEventListener('click', closeDebugDrawer);
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeDebugDrawer(); });

btnDebugClear.addEventListener('click', async () => {
  try {
    await fetch('/api/debug/rejected', { method: 'DELETE' });
    _debugItems = [];
    renderDebug([]);
  } catch (_) {}
});

function renderDebug(items) {
  const n = items.length;
  debugCount.textContent = n;
  debugBadge.textContent = n;
  debugBadge.classList.toggle('visible', n > 0);

  if (!n) {
    debugList.innerHTML = '<div class="debug-empty">No rejected frames yet</div>';
    return;
  }

  // Only re-render if count changed (avoid flash on poll)
  if (n === _debugItems.length) return;
  _debugItems = items;

  debugList.innerHTML = '';
  for (const item of items) {
    const timeStr = new Date(item.timestamp * 1000).toLocaleTimeString([], {
      hour: '2-digit', minute: '2-digit', second: '2-digit'
    });
    const pct = Math.round((item.confidence ?? 0) * 100);
    const div = document.createElement('div');
    div.className = 'debug-item';
    div.innerHTML = `
      ${item.thumbnail ? `<img src="data:image/jpeg;base64,${item.thumbnail}" alt="" loading="lazy">` : ''}
      <div class="debug-item-meta">
        <div class="debug-item-header">
          <span class="debug-kind ${item.kind}">${DEBUG_KIND_LABELS[item.kind] ?? item.kind}</span>
          <span class="debug-conf">${pct}%</span>
          <span class="debug-time">${timeStr}</span>
        </div>
        <div class="debug-desc">${item.description || 'No description'}</div>
      </div>
    `;
    debugList.appendChild(div);
  }
}

async function fetchDebug() {
  try {
    const res = await fetch('/api/debug/rejected');
    const d   = await res.json();
    renderDebug(d.items || []);
  } catch (_) {}
}

// Poll count every 5s to keep badge fresh; full render only when drawer is open
setInterval(async () => {
  try {
    const res = await fetch('/api/debug/rejected');
    const d   = await res.json();
    const items = d.items || [];
    const n = items.length;
    debugCount.textContent = n;
    debugBadge.textContent = n;
    debugBadge.classList.toggle('visible', n > 0);
    if (_debugOpen) renderDebug(items);
  } catch (_) {}
}, 5000);

fetchDebug();

// ── Event detail modal ────────────────────────────────────────────────────────
const eventModal      = document.getElementById('event-modal');
const eventModalImg   = document.getElementById('event-modal-img');
const eventModalType  = document.getElementById('event-modal-type');
const eventModalConf  = document.getElementById('event-modal-conf');
const eventModalTime  = document.getElementById('event-modal-time');
const eventModalDesc  = document.getElementById('event-modal-desc');
const eventModalClose = document.getElementById('event-modal-close');
const btnAlert        = document.getElementById('btn-alert');
const alertStatus     = document.getElementById('event-modal-alert-status');

let _currentModalEvent = null;

const _TYPE_LABELS = { chalking: 'Chalking', sweeper: 'Sweeper', pe_vehicle: 'PE Vehicle' };

function openEventModal(src, ev) {
  eventModalImg.src = src;
  _currentModalEvent = ev;

  if (ev) {
    const pct = Math.round((ev.confidence ?? 0) * 100);
    eventModalType.textContent  = _TYPE_LABELS[ev.event_type] ?? ev.event_type;
    eventModalType.className    = `event-modal-badge ${ev.event_type}`;
    eventModalConf.textContent  = `${pct}% confidence`;
    eventModalTime.textContent  = new Date(ev.timestamp * 1000).toLocaleString();
    eventModalDesc.textContent  = ev.description || 'No description';
  } else {
    eventModalType.textContent = '';
    eventModalConf.textContent = '';
    eventModalTime.textContent = '';
    eventModalDesc.textContent = '';
  }

  alertStatus.textContent = '';
  alertStatus.className   = 'event-modal-status';
  btnAlert.disabled       = false;
  btnAlert.textContent    = '\u{1F4F1} Send Alert';
  eventModal.classList.remove('hidden');
}

function closeEventModal() {
  eventModal.classList.add('hidden');
  eventModalImg.src      = '';
  _currentModalEvent     = null;
}

eventModalClose.addEventListener('click', closeEventModal);
eventModal.addEventListener('click', (e) => { if (e.target === eventModal) closeEventModal(); });
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeEventModal(); });

btnAlert.addEventListener('click', async () => {
  if (!_currentModalEvent) return;
  btnAlert.disabled    = true;
  btnAlert.textContent = 'Sending…';
  alertStatus.textContent = '';
  alertStatus.className   = 'event-modal-status';
  try {
    const res = await fetch('/api/alert', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(_currentModalEvent),
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.detail || 'Send failed');
    }
    alertStatus.textContent = '✓ Alert sent';
    alertStatus.className   = 'event-modal-status ok';
  } catch (err) {
    alertStatus.textContent = err.message;
    alertStatus.className   = 'event-modal-status err';
    btnAlert.disabled       = false;
  }
  btnAlert.textContent = '\u{1F4F1} Send Alert';
});

// Delegate thumbnail clicks from the event list
eventList.addEventListener('click', (e) => {
  const thumb = e.target.closest('.event-thumb');
  if (thumb) {
    const src = thumb.getAttribute('src');
    openEventModal(src, _eventDataMap.get(src));
  }
});
