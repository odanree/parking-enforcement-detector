import { useRef } from 'react';
import { Header }      from './components/Header';
import { VideoPanel }  from './components/VideoPanel/VideoPanel';
import { SidePanel }   from './components/SidePanel/SidePanel';
import { EventModal }  from './components/EventModal';
import { DebugDrawer } from './components/DebugDrawer';
import { useStats }         from './hooks/useStats';
import { useEvents }        from './hooks/useEvents';
import { usePending }       from './hooks/usePending';
import { useDebugRejected } from './hooks/useDebugRejected';
import { useVideoStream }   from './hooks/useVideoStream';

export default function App() {
  useStats();
  useEvents();
  usePending();
  useDebugRejected();

  const feedRef = useRef<HTMLCanvasElement>(null);
  useVideoStream(feedRef);

  return (
    <>
      <Header />
      <main>
        <VideoPanel feedRef={feedRef} />
        <SidePanel />
      </main>
      <DebugDrawer />
      <EventModal />
    </>
  );
}
