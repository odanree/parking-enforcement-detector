import { useEffect, useState } from 'react';
import { Header }            from './components/Header';
import { VideoPanel }        from './components/VideoPanel/VideoPanel';
import { SidePanel }         from './components/SidePanel/SidePanel';
import { EventModal }        from './components/EventModal';
import { DebugDrawer }       from './components/DebugDrawer';
import { ComparisonDrawer }  from './components/ComparisonDrawer';
import { PipelineKanban }   from './components/PipelineKanban';
import { DatasetAdmin }     from './components/DatasetAdmin';
import { TimelinePlot }    from './components/TimelinePlot';
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
  const [pipLarge,  setPipLarge]  = useState(false);

  useEffect(() => {
    if (!kanbanOpen) { setPipClosed(new Set()); setPipLarge(false); }
  }, [kanbanOpen]);

  const cams = [0, 1].filter((id) => !kanbanOpen || !pipClosed.has(id));
  const showPanels = !kanbanOpen || cams.length > 0;

  return (
    <>
      <Header />
      <main>
        {showPanels && (
          <div className={`video-panels${kanbanOpen ? ' pip' : ''}${kanbanOpen && pipLarge ? ' pip-large' : ''}`}>
            {cams.map((camId) => (
              <div key={camId} className="video-panel-pip-wrap">
                <VideoPanel cameraId={camId} />
                {kanbanOpen && (
                  <div className="pip-controls">
                    <button
                      className="pip-btn"
                      title={pipLarge ? 'Small preview' : 'Large preview'}
                      onClick={() => setPipLarge(l => !l)}
                    >{pipLarge ? '⊟' : '⊞'}</button>
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
      <DatasetAdmin />
      <TimelinePlot />
      <EventModal />
    </>
  );
}
