interface Props { cameraId: number }

type Dir = 'up_left'|'up'|'up_right'|'left'|'right'|'down_left'|'down'|'down_right'|'zoom_in'|'zoom_out';

const DPAD: (Dir | null)[][] = [
  ['up_left',   'up',   'up_right'],
  ['left',       null,  'right'],
  ['down_left', 'down', 'down_right'],
];

const LABELS: Record<Dir, string> = {
  up_left:'↖', up:'↑', up_right:'↗',
  left:'←',             right:'→',
  down_left:'↙', down:'↓', down_right:'↘',
  zoom_in:'+', zoom_out:'−',
};

function send(cameraId: number, action: 'move'|'stop', dir: Dir, speed = 4) {
  fetch(`/api/camera/${cameraId}/ptz/${action}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(action === 'move' ? { direction: dir, speed } : { direction: dir }),
  }).catch(() => {});
}

function PtzBtn({ cameraId, dir }: { cameraId: number; dir: Dir }) {
  return (
    <button
      className="ptz-btn"
      onPointerDown={(e) => { e.currentTarget.setPointerCapture(e.pointerId); send(cameraId, 'move', dir); }}
      onPointerUp={() => send(cameraId, 'stop', dir)}
      onPointerLeave={() => send(cameraId, 'stop', dir)}
      onContextMenu={(e) => e.preventDefault()}
    >
      {LABELS[dir]}
    </button>
  );
}

export function PtzOverlay({ cameraId }: Props) {
  return (
    <div className="ptz-overlay">
      <div className="ptz-dpad">
        {DPAD.map((row, ri) =>
          row.map((dir, ci) =>
            dir
              ? <PtzBtn key={dir} cameraId={cameraId} dir={dir} />
              : <div key={`${ri}-${ci}`} className="ptz-center" />
          )
        )}
      </div>
      <div className="ptz-zoom">
        <PtzBtn cameraId={cameraId} dir="zoom_in"  />
        <PtzBtn cameraId={cameraId} dir="zoom_out" />
      </div>
    </div>
  );
}
