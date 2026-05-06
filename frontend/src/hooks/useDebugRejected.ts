import { useEffect } from 'react';
import { useAppStore } from '../store';
import type { DebugItem } from '../types';

export function useDebugRejected() {
  const setDebugItems = useAppStore((s) => s.setDebugItems);

  useEffect(() => {
    let dead = false;
    async function poll() {
      try {
        const res   = await fetch('/api/debug/rejected');
        const d     = await res.json();
        if (!dead) setDebugItems((d.items as DebugItem[]) ?? []);
      } catch { /* ignore */ }
    }
    poll();
    const id = setInterval(poll, 5000);
    return () => { dead = true; clearInterval(id); };
  }, [setDebugItems]);
}
