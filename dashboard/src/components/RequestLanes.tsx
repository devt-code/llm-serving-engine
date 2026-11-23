import { useMemo, useRef, useState } from "react";
import type { EngineModel, ReqState } from "../model";
import type { Theme } from "../theme";

const ROW = 11;
const GAP = 3;
const LEFT = 64;
const MAX_ROWS = 40;

/** Short row label: "you" for the visitor's own request, "chat"/"api" for other API requests. */
function label(id: string, highlight: string) {
  if (id === highlight) return "you";
  if (id.startsWith("chatcmpl-")) return "chat";
  if (id.startsWith("cmpl-")) return "api";
  return id.replace(/^req-|^api-/, "#");
}

/** One row per request on a moving time window: queued (gray), running (blue), preempted (orange).
 *  The highlighted request (the visitor's own prompt) is always shown and marked. */
export function RequestLanes({ model, theme, maxWindowSec = 30, version, highlight = "" }: {
  model: EngineModel; theme: Theme; maxWindowSec?: number; version: number; highlight?: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [hover, setHover] = useState<{ r: ReqState; x: number; y: number } | null>(null);
  const width = 760;
  const now = model.now;
  // The window grows with the run until it reaches maxWindowSec, then scrolls.
  const windowSec = Math.min(maxWindowSec, Math.max(5, now));
  const t0 = Math.max(0, now - windowSec);

  // Pick the rows worth seeing: everything in the batch (running or preempted) first, then the
  // head of the queue, then the most recently finished. Shown in arrival order.
  const rows = useMemo(() => {
    const active: ReqState[] = [], queued: ReqState[] = [], done: ReqState[] = [];
    for (const id of model.order) {
      const r = model.requests.get(id)!;
      const kind = r.segments[r.segments.length - 1]?.kind;
      if (r.finished !== null) { if (r.finished >= t0) done.push(r); }
      else if (kind === "running" || kind === "preempted" || r.preemptions > 0) active.push(r);
      else queued.push(r);
    }
    const pick = active.slice(0, MAX_ROWS);
    pick.push(...queued.slice(0, Math.max(0, Math.min(8, MAX_ROWS - pick.length))));
    pick.push(...done.slice(-Math.max(0, MAX_ROWS - pick.length)));
    pick.push(...queued.slice(8, 8 + Math.max(0, MAX_ROWS - pick.length)));
    const mine = model.requests.get(highlight);
    if (mine && !pick.includes(mine) && (mine.finished === null || mine.finished >= t0)) {
      if (pick.length >= MAX_ROWS) pick.pop();
      pick.push(mine);
    }
    const arrival = new Map(model.order.map((id, i) => [id, i]));
    return pick.sort((a, b) => arrival.get(a.id)! - arrival.get(b.id)!);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [version, t0, highlight]);

  if (!rows.length) return <div className="empty">No requests yet.</div>;

  const plotW = width - LEFT - 8;
  const x = (t: number) => LEFT + ((Math.max(t, t0) - t0) / Math.max(windowSec, 1e-6)) * plotW;
  const height = rows.length * (ROW + GAP) + 22;
  const color = { waiting: theme.waiting, running: theme["series-1"], preempted: theme["series-2"] };
  const ticks = Array.from({ length: 7 }, (_, i) => t0 + (i * windowSec) / 6);

  return (
    <div className="lanes" ref={ref} onMouseLeave={() => setHover(null)}>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Request timeline">
        {ticks.map((t) => (
          <g key={t.toFixed(3)}>
            <line x1={x(t)} x2={x(t)} y1={0} y2={height - 18} stroke={theme.grid} strokeWidth={1} />
            <text x={x(t)} y={height - 4} fontSize={10} fill={theme["text-muted"]} textAnchor="middle">
              {t.toFixed(0)}s
            </text>
          </g>
        ))}
        {rows.map((r, i) => {
          const y = i * (ROW + GAP);
          return (
            <g key={r.id}
               onMouseMove={(ev) => {
                 const box = ref.current!.getBoundingClientRect();
                 setHover({ r, x: ev.clientX - box.left, y: ev.clientY - box.top });
               }}>
              <rect x={0} y={y - 1.5} width={width} height={ROW + 3} rx={3}
                    fill={r.id === highlight ? theme["series-1"] : "transparent"} fillOpacity={0.14} />
              <text x={LEFT - 8} y={y + ROW - 2} fontSize={9.5} textAnchor="end"
                    fill={r.id === highlight ? theme["text-primary"] : theme["text-muted"]}
                    fontWeight={r.id === highlight ? 700 : 400}>
                {label(r.id, highlight)}
              </text>
              {r.segments.map((s, j) => {
                const end = s.end ?? now;
                if (end < t0) return null;
                const x1 = x(s.start), x2 = Math.max(x(end), x1 + 1.5);
                return <rect key={j} x={x1} y={y} width={Math.max(0, x2 - x1 - 1)} height={ROW} rx={2}
                             fill={color[s.kind]} />;
              })}
              {r.firstToken !== null && r.firstToken >= t0 && (
                <line x1={x(r.firstToken)} x2={x(r.firstToken)} y1={y - 1} y2={y + ROW + 1}
                      stroke={theme["text-primary"]} strokeWidth={1.5} />
              )}
            </g>
          );
        })}
      </svg>
      <div className="legend">
        <span><i className="swatch" style={{ background: theme.waiting }} />Queued</span>
        <span><i className="swatch" style={{ background: theme["series-1"] }} />Running</span>
        <span><i className="swatch" style={{ background: theme["series-2"] }} />Preempted (blocks freed, will recompute)</span>
        <span><i className="swatch" style={{ background: theme["text-primary"], width: 2, borderRadius: 0 }} />First token</span>
      </div>
      {hover && <LaneTooltip r={hover.r} x={hover.x} y={hover.y} now={now} />}
    </div>
  );
}

function LaneTooltip({ r, x, y, now }: { r: ReqState; x: number; y: number; now: number }) {
  const state = r.finished !== null ? `finished (${r.reason})` : r.segments[r.segments.length - 1]?.kind;
  const ttft = r.firstToken !== null ? `${((r.firstToken - r.arrived) * 1000).toFixed(0)} ms` : "—";
  const total = ((r.finished ?? now) - r.arrived).toFixed(2);
  return (
    <div className="tooltip" style={{ left: Math.min(x + 14, 560), top: y + 12 }}>
      <div><b>{r.id}</b> · {state}</div>
      <div><span className="k">Prompt</span> {r.promptLen} tokens{r.cachedTokens ? ` (${r.cachedTokens} from prefix cache)` : ""}</div>
      <div><span className="k">Output</span> {r.outputLen || "…"} / {r.maxNew} tokens</div>
      <div><span className="k">Time to first token</span> {ttft}</div>
      <div><span className="k">Elapsed</span> {total} s{r.preemptions ? ` · preempted ${r.preemptions}×` : ""}</div>
    </div>
  );
}
