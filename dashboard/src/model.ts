// Rebuilds the engine's state from its event stream (engine/events.py).
// The same model serves live mode (events from the WebSocket) and replay mode (events from a
// recorded JSONL trace): apply events in order, then read the derived state.

export type EngineEvent = { t: number; type: string; [k: string]: any };

export type SegmentKind = "waiting" | "running" | "preempted";
export type Segment = { kind: SegmentKind; start: number; end: number | null };

export type ReqState = {
  id: string;
  arrived: number;
  promptLen: number;
  maxNew: number;
  segments: Segment[];
  firstToken: number | null;
  finished: number | null;
  outputLen: number;
  reason: string | null;
  cachedTokens: number;
  preemptions: number;
};

export type BlockState = "free" | "used" | "shared" | "cached";

export type StepPoint = {
  t: number;
  kind: string;
  numSeqs: number;
  tokens: number; // tokens produced by this step (one per request in it)
  running: number;
  waiting: number;
  kvUsed: number;
  kvCached: number;
  durationMs: number;
};

export class EngineModel {
  numBlocks = 0;
  blockSize = 16;
  kvUnit = "block";
  policy = "";
  device = "";
  kvBytes = 0;
  now = 0;
  requests = new Map<string, ReqState>();
  order: string[] = []; // request ids by arrival
  refs: Int32Array = new Int32Array(0);
  cached: Uint8Array = new Uint8Array(0);
  owners: Map<number, Set<string>> = new Map();
  steps: StepPoint[] = [];
  preemptions = 0;
  promptTokens = 0;
  cachedPromptTokens = 0;
  events = 0;

  apply(e: EngineEvent): void {
    this.events++;
    this.now = Math.max(this.now, e.t);
    const r = e.id !== undefined ? this.requests.get(e.id) : undefined;
    switch (e.type) {
      case "engine_started": {
        this.policy = e.policy ?? "";
        this.device = e.device ?? "";
        this.kvBytes = e.kv_cache_bytes ?? 0;
        this.kvUnit = e.kv_unit ?? "block";
        this.blockSize = e.block_size ?? 16;
        this.resizeBlocks(e.kv_total ?? 0);
        break;
      }
      case "request_arrived": {
        const req: ReqState = {
          id: e.id, arrived: e.t, promptLen: e.prompt_len ?? 0, maxNew: e.max_new_tokens ?? 0,
          segments: [{ kind: "waiting", start: e.t, end: null }], firstToken: null, finished: null,
          outputLen: 0, reason: null, cachedTokens: 0, preemptions: 0,
        };
        this.requests.set(e.id, req);
        this.order.push(e.id);
        this.promptTokens += req.promptLen;
        break;
      }
      case "request_started":
      case "request_resumed": {
        if (!r) break;
        this.close(r, e.t);
        r.segments.push({ kind: "running", start: e.t, end: null });
        if (e.type === "request_started") {
          r.cachedTokens = e.cached_tokens ?? 0;
          this.cachedPromptTokens += Math.min(r.cachedTokens, r.promptLen);
        }
        break;
      }
      case "request_preempted": {
        if (!r) break;
        this.close(r, e.t);
        r.segments.push({ kind: "preempted", start: e.t, end: null });
        r.preemptions++;
        this.preemptions++;
        break;
      }
      case "first_token": {
        if (r && r.firstToken === null) r.firstToken = e.t;
        break;
      }
      case "request_finished": {
        if (!r) break;
        this.close(r, e.t);
        r.finished = e.t;
        r.outputLen = e.output_len ?? 0;
        r.reason = e.reason ?? null;
        break;
      }
      case "blocks_allocated":
      case "prefix_cache_hit": {
        for (const b of e.blocks as number[]) this.take(b, e.id);
        break;
      }
      case "blocks_freed": {
        const cached = new Set<number>((e.cached as number[]) ?? []);
        for (const b of e.blocks as number[]) {
          if (b >= this.refs.length) continue;
          this.refs[b] = Math.max(0, this.refs[b] - 1);
          this.owners.get(b)?.delete(e.id);
          if (this.refs[b] === 0) this.cached[b] = cached.has(b) ? 1 : 0;
        }
        break;
      }
      case "block_evicted": {
        if (e.block < this.cached.length) this.cached[e.block] = 0;
        break;
      }
      case "step": {
        if (this.numBlocks === 0 && e.kv_total) this.resizeBlocks(e.kv_total);
        this.steps.push({
          t: e.t, kind: e.kind, numSeqs: e.num_seqs, tokens: e.num_seqs,
          running: (e.running as string[] | undefined)?.length ?? 0, waiting: e.waiting ?? 0,
          kvUsed: e.kv_used ?? 0, kvCached: e.kv_cached ?? 0, durationMs: e.duration_ms ?? 0,
        });
        break;
      }
    }
  }

  private resizeBlocks(n: number) {
    this.numBlocks = n;
    this.refs = new Int32Array(n);
    this.cached = new Uint8Array(n);
    this.owners = new Map();
  }

  private take(b: number, id: string) {
    if (b >= this.refs.length) return;
    this.refs[b]++;
    this.cached[b] = 0;
    let s = this.owners.get(b);
    if (!s) this.owners.set(b, (s = new Set()));
    s.add(id);
  }

  private close(r: ReqState, t: number) {
    const last = r.segments[r.segments.length - 1];
    if (last && last.end === null) last.end = t;
  }

  // ------------------------------------------------------------ derived views

  blockState(b: number): BlockState {
    if (this.refs[b] > 1) return "shared";
    if (this.refs[b] === 1) return "used";
    return this.cached[b] ? "cached" : "free";
  }

  blockCounts() {
    const c = { free: 0, used: 0, shared: 0, cached: 0 };
    for (let b = 0; b < this.numBlocks; b++) c[this.blockState(b)]++;
    return c;
  }

  /** Output tokens per second over the last `window` seconds. */
  throughput(window = 3): number {
    let tokens = 0;
    const from = this.now - window;
    for (let i = this.steps.length - 1; i >= 0 && this.steps[i].t >= from; i--) tokens += this.steps[i].tokens;
    const span = Math.min(window, Math.max(this.now - (this.steps[0]?.t ?? this.now), 1e-6));
    return tokens / span;
  }

  current() {
    let running = 0, waiting = 0, finished = 0;
    for (const r of this.requests.values()) {
      if (r.finished !== null) finished++;
      else if (r.segments[r.segments.length - 1]?.kind === "running") running++;
      else waiting++;
    }
    return { running, waiting, finished };
  }

  ttfts(): number[] {
    const out: number[] = [];
    for (const r of this.requests.values()) if (r.firstToken !== null) out.push(r.firstToken - r.arrived);
    return out.sort((a, b) => a - b);
  }

  /** Time series binned to `bin` seconds, for the charts. */
  series(bin = 0.5, windowSec = 60) {
    // Only completed bins: the newest, still-filling bin would plot as a false drop.
    const end = Math.floor(this.now / bin) * bin;
    const start = Math.max(0, end - windowSec);
    const n = Math.max(1, Math.round((end - start) / bin));
    const rows = Array.from({ length: n }, (_, i) => ({
      t: +(start + (i + 1) * bin).toFixed(2), tokPerS: 0, running: 0, waiting: 0, kvUsed: 0, kvCached: 0, _n: 0,
    }));
    for (const s of this.steps) {
      if (s.t < start) continue;
      const i = Math.min(n - 1, Math.floor((s.t - start) / bin));
      const row = rows[i];
      row.tokPerS += s.tokens / bin;
      row.running += s.running;
      row.waiting += s.waiting;
      row.kvUsed += s.kvUsed;
      row.kvCached += s.kvCached;
      row._n++;
    }
    // Carry the last known state through empty bins (idle engine) so lines don't drop to 0 falsely.
    let prev = { running: 0, waiting: 0, kvUsed: 0, kvCached: 0 };
    for (const row of rows) {
      if (row._n) {
        row.running = Math.round(row.running / row._n);
        row.waiting = Math.round(row.waiting / row._n);
        row.kvUsed = Math.round(row.kvUsed / row._n);
        row.kvCached = Math.round(row.kvCached / row._n);
        prev = { running: row.running, waiting: row.waiting, kvUsed: row.kvUsed, kvCached: row.kvCached };
      } else {
        Object.assign(row, prev);
      }
      row.tokPerS = Math.round(row.tokPerS);
    }
    return rows;
  }
}

export function percentile(sorted: number[], p: number): number | null {
  if (!sorted.length) return null;
  return sorted[Math.min(sorted.length - 1, Math.round((p / 100) * (sorted.length - 1)))];
}

export function parseJsonl(text: string): EngineEvent[] {
  const out: EngineEvent[] = [];
  for (const line of text.split("\n")) {
    const s = line.trim();
    if (!s) continue;
    try {
      out.push(JSON.parse(s));
    } catch {
      /* skip a truncated last line */
    }
  }
  return out;
}
