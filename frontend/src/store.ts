import { create } from 'zustand';
import type { AppEvent, DebugItem, Session, Stats, VlmJob } from './types';

interface AppStore {
  stats:       Stats | null;
  events:      AppEvent[];
  sessions:    Session[];
  pending:     VlmJob[];
  debugItems:  DebugItem[];
  debugOpen:        boolean;
  kanbanOpen:       boolean;
  adminOpen:        boolean;
  timelineOpen:     boolean;
  modalEvent:    AppEvent | null;
  wsStatuses:    ('connected' | 'disconnected')[];
  sessionFilter: string | null;

  setStats:            (s: Stats) => void;
  setEvents:           (evs: AppEvent[]) => void;
  setSessions:         (sessions: Session[]) => void;
  setPending:          (jobs: VlmJob[]) => void;
  setDebugItems:       (items: DebugItem[]) => void;
  setDebugOpen:        (open: boolean) => void;
  setKanbanOpen:       (open: boolean) => void;
  setAdminOpen:        (open: boolean) => void;
  setTimelineOpen:     (open: boolean) => void;
  openModal:           (ev: AppEvent) => void;
  closeModal:          () => void;
  setWsStatus:         (cam: number, s: 'connected' | 'disconnected') => void;
  setSessionFilter:    (id: string | null) => void;
}

export const useAppStore = create<AppStore>((set) => ({
  stats:      null,
  events:     [],
  sessions:   [],
  pending:    [],
  debugItems: [],
  debugOpen:        false,
  kanbanOpen:       false,
  adminOpen:        false,
  timelineOpen:     false,
  modalEvent:    null,
  wsStatuses:    ['disconnected', 'disconnected'],
  sessionFilter: null,

  setStats:            (stats)      => set({ stats }),
  setEvents:           (events)     => set({ events }),
  setSessions:         (sessions)   => set({ sessions }),
  setPending:          (pending)    => set({ pending }),
  setDebugItems:       (debugItems) => set({ debugItems }),
  setDebugOpen:        (debugOpen)  => set({ debugOpen }),
  setKanbanOpen:       (kanbanOpen)     => set({ kanbanOpen }),
  setAdminOpen:        (adminOpen)      => set({ adminOpen }),
  setTimelineOpen:     (timelineOpen)   => set({ timelineOpen }),
  openModal:           (modalEvent) => set({ modalEvent }),
  closeModal:          ()           => set({ modalEvent: null }),
  setSessionFilter:    (sessionFilter) => set({ sessionFilter }),
  setWsStatus:         (cam, s)    => set((state) => {
    const next = [...state.wsStatuses] as ('connected' | 'disconnected')[];
    next[cam] = s;
    return { wsStatuses: next };
  }),
}));
