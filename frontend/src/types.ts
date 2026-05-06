export type EventType = 'chalking' | 'sweeper' | 'pe_vehicle';

export interface AppEvent {
  timestamp: number;
  event_type: EventType;
  confidence: number;
  description: string | null;
  snapshot_url: string | null;
}

export interface Stats {
  pipeline_running: boolean;
  paused: boolean;
  motion_detect_enabled: boolean;
  privacy_mode: boolean;
  sweep_window_active: boolean;
  total_chalking: number;
  total_sweeper: number;
  last_chalking: number | null;
  last_sweeper: number | null;
  uptime_seconds: number;
  playback_speed: number;
  playback_direction: number;
  is_live: boolean;
  fps: number;
}

export interface VlmJob {
  id: string;
  kind: EventType;
  thumbnail: string | null;
  submitted_at: number;
  sample_num: number;
  completed_at?: number;
  detected?: boolean;
}

export interface DebugItem {
  kind: EventType;
  thumbnail: string | null;
  confidence: number;
  description: string | null;
  timestamp: number;
}

export type Polygon = [number, number][];
export type PrivacyRegion = [number, number, number, number];

export const TYPE_LABELS: Record<EventType, string> = {
  chalking:   'Chalking',
  sweeper:    'Sweeper',
  pe_vehicle: 'PE Vehicle',
};
