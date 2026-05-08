export type EventType = 'chalking' | 'sweeper' | 'pe_vehicle';

export type Vote = 'up' | 'down' | 'archive' | null;

export interface AppEvent {
  timestamp: number;
  event_type: EventType;
  confidence: number;
  description: string | null;
  snapshot_url: string | null;
  camera?: number;
  camera_id?: number;
  frames?: string[];
  vote?: Vote;
  track_id?: number | null;
  session_id?: string | null;
}

export interface Stats {
  pipeline_running: boolean;
  paused: boolean;
  motion_detect_enabled: boolean;
  privacy_mode: boolean;
  total_chalking: number;
  last_chalking: number | null;
  uptime_seconds: number;
  playback_speed: number;
  playback_direction: number;
  is_live: boolean;
  is_nvr_playback: boolean;
  fps: number;
  demo_mode: boolean;
  vlm_calls: number;
  vlm_cost_usd: number;
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
  frames?: string[];
  suppressed_by_session?: string | null;
}

export type Polygon = [number, number][];
export type PrivacyRegion = [number, number, number, number];

export interface Session {
  session_id: string;
  camera_id: number;
  started_at: number;
  last_seen: number;
  event_count: number;
  track_ids: number[];
  duration_seconds: number;
}

export const TYPE_LABELS: Record<EventType, string> = {
  chalking:   'Chalking',
  sweeper:    'Sweeper',
  pe_vehicle: 'PE Vehicle',
};
