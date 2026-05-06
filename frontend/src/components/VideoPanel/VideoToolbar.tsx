import { useAppStore } from '../../store';

const SPEEDS = [0.5, 1, 2, 4, 8, 16];
const SEEKS  = [-10, -5, 5, 10];

function fmtSpeed(s: number) { return s === 0.5 ? '½x' : `${s}x`; }

interface Props {
  zoneEditing:     boolean;
  privEditing:     boolean;
  onEnterZoneEdit: () => void;
  onEnterPrivEdit: () => void;
}

export function VideoToolbar({ zoneEditing, privEditing, onEnterZoneEdit, onEnterPrivEdit }: Props) {
  const stats     = useAppStore((s) => s.stats);
  const isLive    = stats?.is_live ?? true;
  const paused    = stats?.paused ?? false;
  const speed     = stats?.playback_speed ?? 1;
  const motionOn  = stats?.motion_detect_enabled ?? false;
  const privacyOn = stats?.privacy_mode ?? false;
  const demo      = stats?.demo_mode ?? false;

  async function togglePause() {
    try {
      const res = await fetch('/api/pipeline/pause', { method: 'POST' });
      const d   = await res.json();
      useAppStore.getState().setStats({ ...useAppStore.getState().stats!, paused: d.paused });
    } catch { /* ignore */ }
  }

  async function seek(seconds: number) {
    try {
      await fetch('/api/playback/seek', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ seconds }),
      });
    } catch { /* ignore */ }
  }

  async function setSpeed(s: number) {
    try {
      const res = await fetch('/api/playback/speed', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ speed: s }),
      });
      const d = await res.json();
      useAppStore.getState().setStats({ ...useAppStore.getState().stats!, playback_speed: d.speed });
    } catch { /* ignore */ }
  }

  async function toggleMotion() {
    try {
      const res = await fetch('/api/motion/toggle', { method: 'POST' });
      const d   = await res.json();
      useAppStore.getState().setStats({ ...useAppStore.getState().stats!, motion_detect_enabled: d.motion_detect_enabled });
    } catch { /* ignore */ }
  }

  async function togglePrivacy() {
    try {
      const res = await fetch('/api/privacy/toggle', { method: 'POST' });
      const d   = await res.json();
      useAppStore.getState().setStats({ ...useAppStore.getState().stats!, privacy_mode: d.privacy_mode });
    } catch { /* ignore */ }
  }

  return (
    <div className="video-toolbar">
      <button id="btn-pause" className={`btn-pause${paused ? ' paused' : ''}`} onClick={togglePause}>
        {paused ? (demo ? '▶ Run Demo' : '▶ Resume') : '⏸ Pause'}
      </button>

      {!isLive && (
        <div className="speed-controls">
          {SEEKS.map((s) => (
            <button key={s} className="btn-seek" onClick={() => seek(s)}>
              {s > 0 ? `+${s}s` : `${s}s`}
            </button>
          ))}
          <span className="speed-sep" />
          {SPEEDS.map((s) => (
            <button
              key={s}
              className={`btn-speed${speed === s ? ' active' : ''}`}
              onClick={() => setSpeed(s)}
            >
              {fmtSpeed(s)}
            </button>
          ))}
        </div>
      )}

      <div className="toolbar-right">
        <button
          className={`btn-privacy-edit${privEditing ? ' active' : ''}`}
          onClick={privEditing ? undefined : onEnterPrivEdit}
        >
          Edit Regions
        </button>
        {!demo && (
          <>
            <button className={`btn-motion${motionOn ? ' active' : ''}`} onClick={toggleMotion}>
              {motionOn ? 'Motion Detect ON' : 'Motion Detect'}
            </button>
            <button className={`btn-privacy${privacyOn ? ' active' : ''}`} onClick={togglePrivacy}>
              &#128683; Privacy
            </button>
            <button
              id="btn-edit-zone"
              className={`btn-zone${zoneEditing ? ' active' : ''}`}
              onClick={zoneEditing ? undefined : onEnterZoneEdit}
            >
              {zoneEditing ? 'Editing…' : 'Edit Zone'}
            </button>
          </>
        )}
      </div>
    </div>
  );
}
