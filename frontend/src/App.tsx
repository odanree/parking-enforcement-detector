import { useEffect, useState } from 'react';
import { Header }            from './components/Header';
import { VideoPanel }        from './components/VideoPanel/VideoPanel';
import { SidePanel }         from './components/SidePanel/SidePanel';
import { EventModal }        from './components/EventModal';
import { DebugDrawer }       from './components/DebugDrawer';
import { ComparisonDrawer }  from './components/ComparisonDrawer';
import { PipelineKanban }   from './components/PipelineKanban';
import { useAppStore }      from './store';
import { useStats }         from './hooks/useStats';
import { useEvents }        from './hooks/useEvents';
import { useSessions }      from './hooks/useSessions';
import { usePending }       from './hooks/usePending';
import { useDebugRejected } from './hooks/useDebugRejected';

export default function App() {
  useStats();
  useEvents();
  useSessions();
  usePending();
  useDebugRejected();

  const kanbanOpen    = useAppStore((s) => s.kanbanOpen);
  const setKanbanOpen = useAppStore((s) => s.setKanbanOpen);
  const [pipClosed, setPipClosed] = useState<Set<number>>(new Set());

  useEffect(() => {
    if (!kanbanOpen) setPipClosed(new Set());
  }, [kanbanOpen]);

  const cams = [0, 1].filter((id) => !kanbanOpen || !pipClosed.has(id));
  const showPanels = !kanbanOpen || cams.length > 0;

  return (
    <>
      <Header />
      <main>
        {showPanels && (
          <div className={`video-panels${kanbanOpen ? ' pip' : ''}`}>
            {cams.map((camId) => (
              <div key={camId} className="video-panel-pip-wrap">
                <VideoPanel cameraId={camId} />
                {kanbanOpen && (
                  <div className="pip-controls">
                    <button
                      className="pip-btn"
                      title="Back to fullscreen"
                      onClick={() => setKanbanOpen(false)}
                    >⛶</button>
                    <button
                      className="pip-btn"
                      title="Close preview"
                      onClick={() => setPipClosed((s) => new Set([...s, camId]))}
                    >×</button>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
        <SidePanel />
      </main>
      <DebugDrawer />
      <ComparisonDrawer />
      <PipelineKanban />
      <EventModal />
    </>
  );
}
