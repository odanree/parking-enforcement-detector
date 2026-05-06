import { useEffect, useState } from 'react';
import { useAppStore } from '../../store';

export function VlmQueue() {
  const pending = useAppStore((s) => s.pending);
  const [, setTick] = useState(0);

  const hasActive = pending.some((j) => !j.completed_at);

  // 10fps ticker — only runs while jobs are in flight
  useEffect(() => {
    if (!hasActive) return;
    const id = setInterval(() => setTick((t) => t + 1), 100);
    return () => clearInterval(id);
  }, [hasActive]);

  if (!pending.length) return null;

  const KIND_LABELS: Record<string, string> = { chalking: 'Chalking', sweeper: 'Sweeper', pe_vehicle: 'PE Vehicle' };

  return (
    <div className="card processing-card">
      <h2>VLM Processing</h2>
      <ul className="vlm-job-list">
        {pending.map((job) => {
          const elapsed = ((Date.now() / 1000) - job.submitted_at).toFixed(1) + 's';
          const done    = job.completed_at != null;
          const sample  = job.sample_num > 1 ? ` · Sample #${job.sample_num}` : '';

          return (
            <li key={job.id} className="vlm-job">
              <div className="vlm-thumb-wrap">
                {job.thumbnail && (
                  <img src={`data:image/jpeg;base64,${job.thumbnail}`} alt="" />
                )}
                <div
                  className="vlm-spinner"
                  style={done ? { background: job.detected ? 'rgba(0,200,80,0.55)' : 'rgba(80,80,80,0.55)' } : undefined}
                >
                  {done
                    ? <span style={{ fontSize: 16 }}>{job.detected ? '✓' : '✗'}</span>
                    : <div className="vlm-spinner-ring" />}
                </div>
              </div>
              <div className="vlm-job-meta">
                <div className={`vlm-kind ${job.kind}`}>{KIND_LABELS[job.kind] ?? job.kind}{sample}</div>
                <div className="vlm-elapsed">{elapsed}</div>
                {done && (
                  <div style={{ fontSize: 11, marginTop: 2, color: job.detected ? 'var(--green)' : 'var(--muted)' }}>
                    {job.detected ? 'Detected' : 'Not detected'}
                  </div>
                )}
              </div>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
