import { useEffect, useRef, useState } from "react";
import type { BlockState, EngineModel } from "../model";
import type { Theme } from "../theme";

/** Every KV cache block as a square: free, in use, shared by several requests, or cached. */
export function BlockMap({ model, theme, version }: { model: EngineModel; theme: Theme; version: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const [hover, setHover] = useState<{ b: number; x: number; y: number } | null>(null);
  const [width, setWidth] = useState(480);
  const n = model.numBlocks;
  const cols = Math.max(16, Math.ceil(Math.sqrt(n * 2.2)));
  const cell = Math.max(4, Math.floor(width / cols));
  const rows = Math.ceil(n / cols);

  // Re-run once the map exists: on first render it may be the "waiting" placeholder, with no wrapper to measure.
  const hasBlocks = n > 0;
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    setWidth(el.clientWidth);
    const ro = new ResizeObserver(() => setWidth(el.clientWidth));
    ro.observe(el);
    return () => ro.disconnect();
  }, [hasBlocks]);

  useEffect(() => {
    const c = canvasRef.current;
    if (!c || !n) return;
    const dpr = window.devicePixelRatio || 1;
    c.width = cols * cell * dpr;
    c.height = rows * cell * dpr;
    c.style.height = `${rows * cell}px`;
    c.style.width = `${cols * cell}px`;
    const ctx = c.getContext("2d")!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cols * cell, rows * cell);
    const fill: Record<BlockState, string> = {
      free: theme.free, used: theme["series-1"], shared: theme["series-3"], cached: theme["seq-cached"],
    };
    const gap = cell >= 7 ? 1.5 : 1;
    for (let b = 0; b < n; b++) {
      ctx.fillStyle = fill[model.blockState(b)];
      const x = (b % cols) * cell, y = Math.floor(b / cols) * cell;
      ctx.fillRect(x, y, cell - gap, cell - gap);
    }
  }, [version, n, cols, cell, rows, theme, model]);

  if (!n) return <div className="empty">Waiting for the engine to report its KV cache…</div>;

  const counts = model.blockCounts();
  const pct = (k: number) => `${((100 * k) / n).toFixed(0)}%`;

  return (
    <div className="blockmap" ref={wrapRef} onMouseLeave={() => setHover(null)}>
      <canvas
        ref={canvasRef}
        role="img"
        aria-label={`KV cache: ${counts.used + counts.shared} of ${n} blocks in use`}
        onMouseMove={(ev) => {
          const box = canvasRef.current!.getBoundingClientRect();
          const cx = ev.clientX - box.left, cy = ev.clientY - box.top;
          const b = Math.floor(cy / cell) * cols + Math.floor(cx / cell);
          setHover(b >= 0 && b < n ? { b, x: cx, y: cy } : null);
        }}
      />
      <div className="legend">
        <span><i className="swatch" style={{ background: theme["series-1"] }} />In use · {counts.used} ({pct(counts.used)})</span>
        <span><i className="swatch" style={{ background: theme["series-3"] }} />Shared prefix · {counts.shared}</span>
        <span><i className="swatch" style={{ background: theme["seq-cached"] }} />Cached, idle · {counts.cached}</span>
        <span><i className="swatch" style={{ background: theme.free, boxShadow: `inset 0 0 0 1px ${theme.axis}` }} />Free · {counts.free}</span>
      </div>
      {hover && (
        <div className="tooltip" style={{ left: Math.min(hover.x + 14, Math.max(0, width - 220)), top: hover.y + 12 }}>
          <div><b>Block {hover.b}</b> · {label(model.blockState(hover.b))}</div>
          <div><span className="k">Holds</span> {model.blockSize} tokens of K and V for all layers</div>
          {(model.owners.get(hover.b)?.size ?? 0) > 0 && (
            <div><span className="k">Used by</span> {[...model.owners.get(hover.b)!].slice(0, 6).join(", ")}
              {model.owners.get(hover.b)!.size > 6 ? ` +${model.owners.get(hover.b)!.size - 6}` : ""}</div>
          )}
        </div>
      )}
    </div>
  );
}

function label(s: BlockState) {
  return { free: "free", used: "in use", shared: "shared by several requests", cached: "cached (reusable prefix)" }[s];
}
