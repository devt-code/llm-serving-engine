import { percentile, type EngineModel } from "../model";

function Tile({ label, value, unit, note }: { label: string; value: string; unit?: string; note?: string }) {
  return (
    <div className="tile">
      <div className="label">{label}</div>
      <div className="value">{value}{unit && <span className="unit">{unit}</span>}</div>
      {note && <div className="note">{note}</div>}
    </div>
  );
}

export function StatTiles({ model }: { model: EngineModel }) {
  const cur = model.current();
  const counts = model.blockCounts();
  const ttfts = model.ttfts();
  const p50 = percentile(ttfts, 50);
  const hit = model.promptTokens ? model.cachedPromptTokens / model.promptTokens : 0;
  const used = counts.used + counts.shared;
  const ms = (s: number | null) => (s === null ? "—" : s < 1 ? (s * 1000).toFixed(0) : s.toFixed(2));
  return (
    <div className="tiles">
      <Tile label="Output throughput" value={model.throughput().toFixed(0)} unit="tok/s" note="last 3 s" />
      <Tile label="Running" value={String(cur.running)} note="requests in the batch" />
      <Tile label="Waiting" value={String(cur.waiting)} note="queued or preempted" />
      <Tile label="KV cache in use" value={model.numBlocks ? ((100 * used) / model.numBlocks).toFixed(0) : "—"} unit="%"
            note={`${used} / ${model.numBlocks} blocks`} />
      <Tile label="Prefix cache hits" value={(100 * hit).toFixed(0)} unit="%" note="of prompt tokens reused" />
      <Tile label="Time to first token" value={ms(p50)} unit={p50 !== null && p50 < 1 ? "ms" : "s"} note="median" />
      <Tile label="Completed" value={String(cur.finished)} note={`${model.preemptions} preemptions`} />
    </div>
  );
}
