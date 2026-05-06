import { useEffect, useRef } from 'react';
import { useAppStore } from '../store';
import type { AppEvent } from '../types';

export function useEvents() {
  const setEvents = useAppStore((s) => s.setEvents);
  // ref so mutations don't cause stale closures in the interval callback
  const lastTs = useRef(0);

  useEffect(() => {
    let dead = false;

    async function poll() {
      try {
        const res = await fetch('/api/events');
        const evs: AppEvent[] = await res.json();
        if (dead || !evs.length) return;

        const newest = evs[0].timestamp;
        if (newest <= lastTs.current) return;

        lastTs.current = newest;
        setEvents(evs);
      } catch { /* ignore */ }
    }

    poll();
    const id = setInterval(poll, 2000);
    return () => { dead = true; clearInterval(id); };
  }, [setEvents]);
}
