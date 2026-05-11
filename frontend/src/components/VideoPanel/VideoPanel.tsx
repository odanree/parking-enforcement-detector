import { useRef, useState } from 'react';
import { useAppStore } from '../../store';
import { useVideoStream } from '../../hooks/useVideoStream';
import { VideoToolbar } from './VideoToolbar';
import { ZoneOverlay } from './ZoneOverlay';
import { PrivacyOverlay } from './PrivacyOverlay';
import { PtzOverlay } from './PtzOverlay';

interface Props { cameraId: number }

export function VideoPanel({ cameraId }: Props) {
  const stats      = useAppStore((s) => s.stats);
  const wsStatuses = useAppStore((s) => s.wsStatuses);
  const wsStatus   = wsStatuses[cameraId] ?? 'disconnected';

  const feedRef = useRef<HTMLCanvasElement>(null);
  const zoneRef = useRef<HTMLCanvasElement>(null);
  const privRef = useRef<HTMLCanvasElement>(null);

  useVideoStream(feedRef, cameraId);

  const [zoneEditing, setZoneEditing] = useState(false);
  const [privEditing, setPrivEditing] = useState(false);
  const [ptzActive,   setPtzActive]   = useState(false);

  const privacyOn = stats?.privacy_mode ?? false;
  const fps       = Math.round(stats?.fps ?? 0);

  return (
    <section className={`video-panel${zoneEditing || privEditing ? ' editing' : ''}`}>
      <div className="canvas-label">Cam {cameraId}</div>
      <div className="canvas-wrap">
        <canvas ref={feedRef}  className="feed-canvas"    width={1280} height={720} />
        <canvas ref={privRef}  className="privacy-canvas" width={1280} height={720} />
        <canvas ref={zoneRef}  className="zone-canvas"    width={1280} height={720} />

        {wsStatus === 'disconnected' && (
          <div className="no-signal">Waiting for stream&hellip;</div>
        )}
        <div className="fps-badge">{fps} fps</div>
        {ptzActive && <PtzOverlay cameraId={cameraId} />}
      </div>

      <VideoToolbar
        cameraId={cameraId}
        zoneEditing={zoneEditing}
        privEditing={privEditing}
        ptzActive={ptzActive}
        onEnterZoneEdit={() => setZoneEditing(true)}
        onEnterPrivEdit={() => setPrivEditing(true)}
        onTogglePtz={() => setPtzActive((v) => !v)}
      />
      <PrivacyOverlay
        canvasRef={privRef}
        privacyMode={privacyOn}
        editing={privEditing}
        onDone={() => setPrivEditing(false)}
        cameraId={cameraId}
      />
      <ZoneOverlay
        canvasRef={zoneRef}
        editing={zoneEditing}
        onDone={() => setZoneEditing(false)}
        cameraId={cameraId}
      />
    </section>
  );
}
