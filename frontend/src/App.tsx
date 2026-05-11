import { Header }            from './components/Header';
import { VideoPanel }        from './components/VideoPanel/VideoPanel';
import { SidePanel }         from './components/SidePanel/SidePanel';
import { EventModal }        from './components/EventModal';
import { DebugDrawer }       from './components/DebugDrawer';
import { ComparisonDrawer }  from './components/ComparisonDrawer';
import { PipelineKanban }   from './components/PipelineKanban';
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

  return (
    <>
      <Header />
      <main>
        <div className="video-panels">
          <VideoPanel cameraId={0} />
          <VideoPanel cameraId={1} />
        </div>
        <SidePanel />
      </main>
      <DebugDrawer />
      <ComparisonDrawer />
      <PipelineKanban />
      <EventModal />
    </>
  );
}
