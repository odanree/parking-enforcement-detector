import { Header }      from './components/Header';
import { VideoPanel }  from './components/VideoPanel/VideoPanel';
import { SidePanel }   from './components/SidePanel/SidePanel';
import { EventModal }  from './components/EventModal';
import { DebugDrawer } from './components/DebugDrawer';
import { useStats }         from './hooks/useStats';
import { useEvents }        from './hooks/useEvents';
import { usePending }       from './hooks/usePending';
import { useDebugRejected } from './hooks/useDebugRejected';

export default function App() {
  useStats();
  useEvents();
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
      <EventModal />
    </>
  );
}
