import { create } from 'zustand';
import type { AppEvent, DebugItem, Stats, VlmJob } from './types';

interface AppStore {
  stats:       Stats | null;
  events:      AppEvent[];
  pending:     VlmJob[];
  debugItems:  DebugItem[];
  debugOpen:   boolean;
  modalEvent:  AppEvent | null;
  wsStatuses:  ('connected' | 'disconnected')[];

  setStats:      (s: Stats) => void;
  setEvents:     (evs: AppEvent[]) => void;
  setPending:    (jobs: VlmJob[]) => void;
  setDebugItems: (items: DebugItem[]) => void;
  setDebugOpen:  (open: boolean) => void;
  openModal:     (ev: AppEvent) => void;
  closeModal:    () => void;
  setWsStatus:   (cam: number, s: 'connected' | 'disconnected') => void;
}

export const useAppStore = create<AppStore>((set) => ({
  stats:      null,
  events:     [],
  pending:    [],
  debugItems: [],
  debugOpen:  false,
  modalEvent: null,
  wsStatuses: ['disconnected', 'disconnected'],

  setStats:      (stats)      => set({ stats }),
  setEvents:     (events)     => set({ events }),
  setPending:    (pending)    => set({ pending }),
  setDebugItems: (debugItems) => set({ debugItems }),
  setDebugOpen:  (debugOpen)  => set({ debugOpen }),
  openModal:     (modalEvent) => set({ modalEvent }),
  closeModal:    ()           => set({ modalEvent: null }),
  setWsStatus:   (cam, s)     => set((state) => {
    const next = [...state.wsStatuses] as ('connected' | 'disconnected')[];
    next[cam] = s;
    return { wsStatuses: next };
  }),
}));
