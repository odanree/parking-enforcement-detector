import { useEffect, useRef, useState, useCallback } from 'react';
import { useAppStore } from '../store';

interface Dot { t: number; cam: number; det: number; people: number; yc: number; }

const YOLO_HIGH = 0.70;
interface Gap { x1: number; x2: number; durSec: number; tStart: number; tEnd: number; }
interface Tooltip { x: number; y: number; text: string; }

const GAP_MIN = 30 * 60;
const DOT_R   = 3;
const LANE_H  = 48;
const AXIS_H  = 24;
const PAD_L   = 8;
const PAD_R   = 16;

function fmt(ts: number): string {
  return new Date(ts * 1000).toLocaleString([], {
    month: 'short', day: 'numeric',
    hour: '2-digit', minute: '2-digit', second: '2-digit',
  });
}

function fmtDur(sec: number): string {
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  if (h > 0) return `${h}h ${m}m`;
  return `${m}m`;
}

function fmtAxis(ts: number, span: number): string {
  const d = new Date(ts * 1000);
  if (span < 3600 * 2)
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  if (span < 86400 * 2)
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  return d.toLocaleDateString([], { month: 'short', day: 'numeric' }) +
         ' ' + d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

export function TimelinePlot() {
  const open    = useAppStore((s) => s.timelineOpen);
  const setOpen = useAppStore((s) => s.setTimelineOpen);

  const [dots,      setDots]      = useState<Dot[]>([]);
  const [loading,   setLoading]   = useState(false);
  const [tooltip,   setTooltip]   = useState<Tooltip | null>(null);
  const svgRef      = useRef<SVGSVGElement>(null);
  const [svgW,      setSvgW]      = useState(800);
  const [analyzing, setAnalyzing] = useState(false);
  const [analysis,  setAnalysis]  = useState<string>('');
  const analysisRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) { setAnalysis(''); return; }
    setLoading(true);
    fetch('/api/timeline')
      .then(r => r.json())
      .then((rows: Dot[]) => { setDots(rows); setLoading(false); })
      .catch(() => setLoading(false));
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const el = svgRef.current?.parentElement;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => setSvgW(entry.contentRect.width));
    ro.observe(el);
    return () => ro.disconnect();
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const h = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false); };
    window.addEventListener('keydown', h);
    return () => window.removeEventListener('keydown', h);
  }, [open, setOpen]);

  const handleMouseMove = useCallback((e: React.MouseEvent<SVGSVGElement>, gaps: Gap[]) => {
    if (!dots.length) return;
    const rect = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    const tMin = dots[0].t;
    const tMax = dots[dots.length - 1].t;
    const span = tMax - tMin || 1;
    const drawW = svgW - PAD_L - PAD_R;

    // check if hovering a gap band first
    for (const g of gaps) {
      if (mx >= g.x1 && mx <= g.x2 && my >= 0 && my <= LANE_H * 2) {
        setTooltip({
          x: (g.x1 + g.x2) / 2,
          y: LANE_H - 10,
          text: `Gap: ${fmtDur(g.durSec)} · ${fmt(g.tStart)} → ${fmt(g.tEnd)}`,
        });
        return;
      }
    }

    // find closest dot within 12px
    let best: Dot | null = null;
    let bestDist = 144;
    for (const d of dots) {
      const cx = PAD_L + ((d.t - tMin) / span) * drawW;
      const cy = d.cam === 0 ? LANE_H * 0.5 : LANE_H * 1.5;
      const dist = (cx - mx) ** 2 + (cy - my) ** 2;
      if (dist < bestDist) { bestDist = dist; best = d; }
    }
    if (best) {
      const cx = PAD_L + ((best.t - tMin) / span) * drawW;
      const cy = best.cam === 0 ? LANE_H * 0.5 : LANE_H * 1.5;
      const outcome = best.det ? 'Chalking'
        : best.yc >= YOLO_HIGH ? `High YOLO ${Math.round(best.yc * 100)}%`
        : best.people ? 'People detected'
        : 'Rejected';
      setTooltip({
        x: cx,
        y: cy - 10,
        text: `Cam ${best.cam} · ${outcome} · ${fmt(best.t)}`,
      });
    } else {
      setTooltip(null);
    }
  }, [dots, svgW]);

  const handleAnalyze = useCallback(async () => {
    if (!dots.length || analyzing) return;
    setAnalysis('');
    setAnalyzing(true);
    try {
      const res = await fetch('/api/timeline/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ rows: dots }),
      });
      if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = '';
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split('\n');
        buf = lines.pop() ?? '';
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const payload = line.slice(6).trim();
          if (payload === '[DONE]') break;
          try {
            const { text } = JSON.parse(payload);
            setAnalysis(prev => prev + text);
            // auto-scroll to bottom of analysis panel
            if (analysisRef.current) {
              analysisRef.current.scrollTop = analysisRef.current.scrollHeight;
            }
          } catch { /* ignore malformed */ }
        }
      }
    } catch (err) {
      setAnalysis(`Error: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setAnalyzing(false);
    }
  }, [dots, analyzing]);

  if (!open) return null;

  const svgH = LANE_H * 2 + AXIS_H;

  const tMin = dots.length ? dots[0].t : 0;
  const tMax = dots.length ? dots[dots.length - 1].t : 1;
  const span = tMax - tMin || 1;
  const drawW = svgW - PAD_L - PAD_R;

  const toX = (t: number) => PAD_L + ((t - tMin) / span) * drawW;

  const gaps: Gap[] = [];
  for (let i = 1; i < dots.length; i++) {
    const durSec = dots[i].t - dots[i - 1].t;
    if (durSec > GAP_MIN) {
      gaps.push({
        x1: toX(dots[i - 1].t),
        x2: toX(dots[i].t),
        durSec,
        tStart: dots[i - 1].t,
        tEnd:   dots[i].t,
      });
    }
  }

  const ticks: number[] = [];
  if (dots.length > 1) {
    const step = span / 5;
    for (let i = 0; i <= 5; i++) ticks.push(tMin + i * step);
  }

  const TOOLTIP_W = 340;

  return (
    <div className="tl-overlay" onClick={(e) => { if (e.target === e.currentTarget) setOpen(false); }}>
      <div className="tl-panel">
        <div className="tl-header">
          <span className="tl-title">Snapshot Timeline</span>
          <span className="tl-meta">
            {loading ? 'Loading…' : `${dots.length} snapshots · ${gaps.length} gap${gaps.length !== 1 ? 's' : ''}`}
            {!loading && dots.length > 0 && ` · ${fmt(tMin)} → ${fmt(tMax)}`}
          </span>
          <button
            className="tl-analyze-btn"
            onClick={handleAnalyze}
            disabled={analyzing || !dots.length}
          >{analyzing ? 'Analyzing…' : 'Analyze'}</button>
          <button className="tl-close" onClick={() => setOpen(false)}>×</button>
        </div>

        <div className="tl-body">
          {loading && <div className="tl-loading">Loading…</div>}
          {!loading && dots.length === 0 && <div className="tl-loading">No data yet.</div>}
          {!loading && dots.length > 0 && (
            <div className="tl-svg-wrap">
              <div className="tl-lane-labels">
                <div style={{ height: LANE_H }}>Cam 0</div>
                <div style={{ height: LANE_H }}>Cam 1</div>
              </div>

              <svg
                ref={svgRef}
                className="tl-svg"
                width="100%"
                height={svgH}
                onMouseMove={(e) => handleMouseMove(e, gaps)}
                onMouseLeave={() => setTooltip(null)}
              >
                {/* lane backgrounds */}
                <rect x={0} y={0}      width={svgW} height={LANE_H} fill="#111" />
                <rect x={0} y={LANE_H} width={svgW} height={LANE_H} fill="#0d0d0d" />
                <line x1={0} y1={LANE_H} x2={svgW} y2={LANE_H} stroke="#1e1e1e" strokeWidth={1} />

                {/* gap bands */}
                {gaps.map((g, i) => {
                  const bw = g.x2 - g.x1;
                  const cx = (g.x1 + g.x2) / 2;
                  const label = fmtDur(g.durSec);
                  // estimate text width: ~7px per char at font-size 11
                  const labelFits = bw > label.length * 7 + 8;
                  return (
                    <g key={i}>
                      <rect x={g.x1} y={0} width={bw} height={LANE_H * 2}
                        fill="rgba(255,145,0,0.10)" />
                      {/* left/right border lines */}
                      <line x1={g.x1} y1={0} x2={g.x1} y2={LANE_H * 2}
                        stroke="rgba(255,145,0,0.35)" strokeWidth={1} strokeDasharray="3 3" />
                      <line x1={g.x2} y1={0} x2={g.x2} y2={LANE_H * 2}
                        stroke="rgba(255,145,0,0.35)" strokeWidth={1} strokeDasharray="3 3" />
                      {/* duration label centered in band */}
                      {labelFits && (
                        <text
                          x={cx}
                          y={LANE_H}
                          textAnchor="middle"
                          dominantBaseline="middle"
                          fontSize={11}
                          fontWeight={600}
                          fill="rgba(255,145,0,0.8)"
                        >{label}</text>
                      )}
                    </g>
                  );
                })}

                {/* dots — red=rejected, orange=high YOLO, blue=VLM people, green=chalking */}
                {dots.map((d, i) => {
                  const highYolo = d.yc >= YOLO_HIGH;
                  const fill = d.det      ? '#00e676'
                             : highYolo   ? '#ff9100'
                             : d.people   ? '#40c4ff'
                             : '#ff1744';
                  const r = d.det ? DOT_R + 1 : DOT_R;
                  return (
                    <circle
                      key={i}
                      cx={toX(d.t)}
                      cy={d.cam === 0 ? LANE_H * 0.5 : LANE_H * 1.5}
                      r={r}
                      fill={fill}
                      opacity={d.det ? 1 : 0.8}
                    />
                  );
                })}

                {/* axis ticks */}
                {ticks.map((t, i) => (
                  <g key={i}>
                    <line x1={toX(t)} y1={LANE_H * 2} x2={toX(t)} y2={LANE_H * 2 + 4}
                      stroke="#333" strokeWidth={1} />
                    <text
                      x={toX(t)}
                      y={LANE_H * 2 + AXIS_H - 4}
                      textAnchor="middle"
                      fontSize={9}
                      fill="#555"
                    >{fmtAxis(t, span)}</text>
                  </g>
                ))}

                {/* tooltip */}
                {tooltip && (() => {
                  const tx = Math.min(Math.max(tooltip.x - 4, 4), svgW - TOOLTIP_W - 4);
                  return (
                    <g>
                      <rect x={tx} y={tooltip.y - 18} width={TOOLTIP_W} height={20}
                        rx={3} fill="rgba(0,0,0,0.88)" stroke="#333" strokeWidth={1} />
                      <text x={tx + 6} y={tooltip.y - 4} fontSize={11} fill="#d4d4d4">
                        {tooltip.text}
                      </text>
                    </g>
                  );
                })()}
              </svg>
            </div>
          )}

          <div className="tl-legend">
            <span className="tl-leg tl-leg-det">● Chalking</span>
            <span className="tl-leg tl-leg-ppl">● People detected</span>
            <span className="tl-leg tl-leg-yolo">● High YOLO ≥70%</span>
            <span className="tl-leg tl-leg-rej">● Rejected</span>
            <span className="tl-leg tl-leg-gap">▬ Gap &gt;30 min</span>
          </div>

          {(analysis || analyzing) && (
            <div className="tl-analysis-panel" ref={analysisRef}>
              {analysis || <span className="tl-analysis-thinking">Thinking…</span>}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
