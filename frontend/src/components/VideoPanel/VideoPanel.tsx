import { useRef, useState, type RefObject } from 'react';
import { useAppStore } from '../../store';
import { VideoToolbar } from './VideoToolbar';
import { ZoneOverlay } from './ZoneOverlay';
import { PrivacyOverlay } from './PrivacyOverlay';

interface Props { feedRef: RefObject<HTMLCanvasElement | null> }

export function VideoPanel({ feedRef }: Props) {
  const stats    = useAppStore((s) => s.stats);
  const wsStatus = useAppStore((s) => s.wsStatus);
  const zoneRef  = useRef<HTMLCanvasElement>(null);
  const privRef  = useRef<HTMLCanvasElement>(null);

  const [zoneEditing, setZoneEditing] = useState(false);
  const [privEditing, setPrivEditing] = useState(false);

  const privacyOn = stats?.privacy_mode ?? false;
  const fps       = Math.round(stats?.fps ?? 0);

  return (
    <section className="video-panel">
      <div className="canvas-wrap">
        <canvas ref={feedRef}  id="feed"            width={1280} height={720} />
        <canvas ref={privRef}  id="privacy-overlay" width={1280} height={720} />
        <canvas ref={zoneRef}  id="zone-overlay"    width={1280} height={720} />

        {wsStatus === 'disconnected' && (
          <div className="no-signal">Waiting for stream&hellip;</div>
        )}
        <div className="fps-badge">{fps} fps</div>
      </div>

      <VideoToolbar
        zoneEditing={zoneEditing}
        privEditing={privEditing}
        onEnterZoneEdit={() => setZoneEditing(true)}
        onEnterPrivEdit={() => setPrivEditing(true)}
      />

      <PrivacyOverlay
        canvasRef={privRef}
        privacyMode={privacyOn}
        editing={privEditing}
        onDone={() => setPrivEditing(false)}
      />

      <ZoneOverlay
        canvasRef={zoneRef}
        editing={zoneEditing}
        onDone={() => setZoneEditing(false)}
      />
    </section>
  );
}
