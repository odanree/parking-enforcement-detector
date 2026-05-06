import { useEffect } from 'react';
import { useAppStore } from '../store';
import type { VlmJob } from '../types';

export function usePending() {
  const setPending = useAppStore((s) => s.setPending);

  useEffect(() => {
    let dead = false;
    async function poll() {
      try {
        const res = await fetch('/api/pending');
        const d   = await res.json();
        if (!dead) setPending((d.jobs as VlmJob[]) ?? []);
      } catch { /* ignore */ }
    }
    poll();
    const id = setInterval(poll, 1000);
    return () => { dead = true; clearInterval(id); };
  }, [setPending]);
}
