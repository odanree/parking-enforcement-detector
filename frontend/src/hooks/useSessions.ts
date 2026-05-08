import { useEffect } from 'react';
import { useAppStore } from '../store';
import type { Session } from '../types';

export function useSessions() {
  const setSessions = useAppStore((s) => s.setSessions);

  useEffect(() => {
    let dead = false;

    async function poll() {
      try {
        const res = await fetch('/api/sessions');
        const sessions: Session[] = await res.json();
        if (!dead) setSessions(sessions);
      } catch { /* ignore */ }
    }

    poll();
    const id = setInterval(poll, 5000);
    return () => { dead = true; clearInterval(id); };
  }, [setSessions]);
}
