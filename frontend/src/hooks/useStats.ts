import { useEffect } from 'react';
import { useAppStore } from '../store';

export function useStats() {
  const setStats = useAppStore((s) => s.setStats);

  useEffect(() => {
    let dead = false;
    async function poll() {
      try {
        const res = await fetch('/api/stats');
        if (!dead) setStats(await res.json());
      } catch { /* ignore transient errors */ }
    }
    poll();
    const id = setInterval(poll, 2000);
    return () => { dead = true; clearInterval(id); };
  }, [setStats]);
}
