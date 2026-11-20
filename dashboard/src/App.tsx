import { useEffect, useState } from "react";
import { BlockMap } from "./components/BlockMap";
import { KvChart, QueueChart, ThroughputChart } from "./components/Charts";
import { RequestLanes } from "./components/RequestLanes";
import { StatTiles } from "./components/StatTiles";
import type { EngineModel } from "./model";
import { useLive, useLoadGenerator, useReplay, type TraceInfo } from "./sources";
import { useTheme } from "./theme";

type Mode = "replay" | "live";
const REPO = "https://github.com/devt-code/llm-serving-engine";

export default function App() {
  const params = new URLSearchParams(location.search);
  const [mode, setMode] = useState<Mode>(params.has("live") ? "live" : "replay");
  const [server, setServer] = useState(params.get("server") ?? "http://localhost:8000");
  const theme = useTheme();
  const live = useLive(server, mode === "live");
  const replay = useReplay();
  const [traces, setTraces] = useState<TraceInfo[]>([]);
  const [trace, setTrace] = useState("");

  useEffect(() => {
    fetch("traces/index.json")
      .then((r) => (r.ok ? r.json() : []))
      .then((list: TraceInfo[]) => {
        setTraces(list);
        if (list.length) setTrace(list[0].file);
      })
      .catch(() => setTraces([]));
  }, []);

  useEffect(() => {
    if (mode === "replay" && trace) void replay.loadUrl(`traces/${trace}`);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [trace, mode]);

  const model = mode === "live" ? live.model : replay.model;
  const version = mode === "live" ? live.version : replay.version;
  if (import.meta.env.DEV) (window as unknown as { __model: EngineModel }).__model = model; // for debugging in devtools
  const info = traces.find((t) => t.file === trace);

  return (
    <div className="app">
      <header className="header">
        <div>
          <h1>LLM Serving Engine</h1>
          <div className="sub">
            Continuous batching, a paged KV cache and prefix caching, running Qwen3-0.6B. Built from scratch in PyTorch.
          </div>
        </div>
        <div className="spacer" />
        <div className="tabs" role="tablist">
          <button role="tab" aria-selected={mode === "replay"} onClick={() => setMode("replay")}>Replay a recorded run</button>
          <button role="tab" aria-selected={mode === "live"} onClick={() => setMode("live")}>Live engine</button>
        </div>
      </header>

      {mode === "replay" ? <ReplayControls replay={replay} traces={traces} trace={trace} setTrace={setTrace} />
                         : <LiveControls server={server} setServer={setServer} live={live} />}
      {mode === "replay" && info && <p className="hint" style={{ marginTop: -6, marginBottom: 12, color: "var(--text-secondary)" }}>{info.description}</p>}
      {mode === "replay" && replay.error && <p className="hint" style={{ color: "var(--critical)" }}>{replay.error}</p>}

      <Panels model={model} version={version} theme={theme} />

      <div className="footer">
        Every panel is rebuilt from the engine's event stream: request arrivals, scheduler steps, KV block
        allocations, preemptions and prefix-cache hits. <a href={REPO}>Source on GitHub</a>.
      </div>
    </div>
  );
}

function Panels({ model, version, theme }: { model: EngineModel; version: number; theme: ReturnType<typeof useTheme> }) {
  const data = model.series(0.5, 60);
  return (
    <>
      <StatTiles model={model} />
      <div className="grid-2">
        <section className="card">
          <h2>Requests in the batch</h2>
          <p className="hint">Each row is a request. With continuous batching, requests join and leave at every step.</p>
          <RequestLanes model={model} theme={theme} version={version} />
        </section>
        <section className="card">
          <h2>KV cache blocks</h2>
          <p className="hint">Each square is a {model.blockSize}-token block of GPU memory, handed out as requests grow.</p>
          <BlockMap model={model} theme={theme} version={version} />
        </section>
      </div>
      <div className="grid-3">
        <section className="card">
          <h2>Output tokens per second</h2>
          <p className="hint">Last 60 s</p>
          <ThroughputChart data={data} theme={theme} />
        </section>
        <section className="card">
          <h2>Running vs waiting requests</h2>
          <p className="hint">Last 60 s</p>
          <QueueChart data={data} theme={theme} />
        </section>
        <section className="card">
          <h2>KV cache blocks over time</h2>
          <p className="hint">Of {model.numBlocks || "—"} blocks</p>
          <KvChart data={data} total={model.numBlocks} theme={theme} />
        </section>
      </div>
    </>
  );
}

function ReplayControls({ replay, traces, trace, setTrace }: {
  replay: ReturnType<typeof useReplay>; traces: TraceInfo[]; trace: string; setTrace: (s: string) => void;
}) {
  return (
    <div className="controls">
      {traces.length > 0 && (
        <label>Recording
          <select value={trace} onChange={(e) => setTrace(e.target.value)}>
            {traces.map((t) => <option key={t.file} value={t.file}>{t.title}</option>)}
          </select>
        </label>
      )}
      <button className="btn primary" onClick={() => (replay.time >= replay.duration ? replay.restart() : replay.setPlaying(!replay.playing))}
              disabled={!replay.duration}>
        {replay.playing ? "Pause" : replay.time >= replay.duration && replay.duration ? "Replay" : "Play"}
      </button>
      <label>Speed
        <select value={replay.speed} onChange={(e) => replay.setSpeed(Number(e.target.value))}>
          {[0.5, 1, 2, 4, 8].map((s) => <option key={s} value={s}>{s}×</option>)}
        </select>
      </label>
      <div className="scrub">
        <input type="range" min={0} max={replay.duration || 1} step={0.01} value={replay.time}
               aria-label="Playback position"
               onChange={(e) => { replay.setPlaying(false); replay.seek(Number(e.target.value)); }} />
        <span className="mono" style={{ color: "var(--text-secondary)", fontSize: 13 }}>
          {replay.time.toFixed(1)} / {replay.duration.toFixed(1)} s
        </span>
      </div>
      <label className="btn" style={{ cursor: "pointer" }}>
        Open trace…
        <input type="file" accept=".jsonl,.json,.txt" hidden
               onChange={(e) => e.target.files?.[0] && replay.loadFile(e.target.files[0])} />
      </label>
    </div>
  );
}

function LiveControls({ server, setServer, live }: {
  server: string; setServer: (s: string) => void; live: ReturnType<typeof useLive>;
}) {
  const [draft, setDraft] = useState(server);
  const load = useLoadGenerator(server);
  const [maxTokens, setMaxTokens] = useState(96);
  const [shared, setShared] = useState(true);
  load.opts.current = { maxTokens, sharedPrefix: shared };
  const statusClass = live.status === "open" ? "ok" : live.status === "error" || live.status === "closed" ? "bad" : "";
  const statusText = {
    idle: "Not connected", connecting: "Connecting…", open: `Connected${live.servedModel ? ` · ${live.servedModel}` : ""}`,
    closed: "Disconnected", error: "Can't reach the server. Is `python -m server.api` running?",
  }[live.status];
  // A page served from the public internet can't open connections to localhost (browsers block
  // public sites from reaching the local network). Live mode needs the dashboard served locally.
  const hosted = !["localhost", "127.0.0.1", "[::1]"].includes(location.hostname);
  const failed = live.status === "error" || live.status === "closed";
  return (
    <>
    {hosted && live.status !== "open" && (
      <div className="card notice">
        <h2>Live mode runs on your own computer</h2>
        <p>
          This page is hosted on the web, and browsers don't let a website connect to programs on your machine. To watch
          the engine live, run the engine and this dashboard locally (Python 3.11 and Node 18+, about 4 GB):
        </p>
        <pre>{`git clone ${REPO}.git && cd llm-serving-engine
python3.11 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-0.6B', local_dir='models/qwen3-0.6b')"
python -m server.api                      # terminal 1
cd dashboard && npm install && npx vite   # terminal 2`}</pre>
        <p>Then open <a href="http://localhost:5173/?live">http://localhost:5173/?live</a>. Or switch to <b>Replay a recorded run</b> above: it shows a real recorded run, no setup needed.</p>
      </div>
    )}
    <div className="controls" style={hosted && failed ? { opacity: 0.55 } : undefined}>
      <label>Server
        <input type="text" value={draft} onChange={(e) => setDraft(e.target.value)}
               onKeyDown={(e) => e.key === "Enter" && setServer(draft)} onBlur={() => setServer(draft)} />
      </label>
      <span className={`status-dot ${statusClass}`}>{statusText}</span>
      <button className="btn primary" onClick={() => load.burst(50)} disabled={live.status !== "open"}>Send 50 requests</button>
      <label>Arrival rate
        <input type="range" min={0} max={20} step={1} value={load.rate} onChange={(e) => load.setRate(Number(e.target.value))}
               disabled={live.status !== "open"} />
        <span className="mono" style={{ minWidth: 52 }}>{load.rate} req/s</span>
      </label>
      <label>Max tokens
        <input type="range" min={16} max={256} step={16} value={maxTokens} onChange={(e) => setMaxTokens(Number(e.target.value))} />
        <span className="mono" style={{ minWidth: 28 }}>{maxTokens}</span>
      </label>
      <label><input type="checkbox" checked={shared} onChange={(e) => setShared(e.target.checked)} /> Shared system prompt</label>
      <span className="mono" style={{ color: "var(--text-muted)", fontSize: 12 }}>
        {load.sent} sent{load.errors ? ` · ${load.errors} errors` : ""}
      </span>
    </div>
    </>
  );
}
