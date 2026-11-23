import { useCallback, useEffect, useRef, useState } from "react";
import { EngineModel, parseJsonl, type EngineEvent } from "./model";

// ------------------------------------------------------------------ live: WebSocket + load generator

export type LiveStatus = "idle" | "connecting" | "open" | "closed" | "error";

export function useLive(baseUrl: string, enabled: boolean) {
  const modelRef = useRef(new EngineModel());
  const [version, setVersion] = useState(0);
  const [status, setStatus] = useState<LiveStatus>("idle");
  const [model, setModel] = useState("");
  const [device, setDevice] = useState("");
  const [closeReason, setCloseReason] = useState("");

  useEffect(() => {
    if (!enabled) return;
    modelRef.current = new EngineModel();
    setStatus("connecting");
    setCloseReason("");
    const ws = new WebSocket(baseUrl.replace(/^http/, "ws").replace(/\/$/, "") + "/v1/events");
    let dirty = false;
    ws.onopen = () => setStatus("open");
    ws.onerror = () => setStatus("error");
    ws.onclose = (ev) => {
      setCloseReason(ev.reason); // "idle": the server dropped an idle viewer so it can shut down
      setStatus((s) => (s === "error" ? s : "closed"));
    };
    ws.onmessage = (msg) => {
      const e = JSON.parse(msg.data) as EngineEvent;
      if (e.type === "hello") {
        setModel(e.model);
        setDevice(e.device ?? "");
        const m = modelRef.current;
        // Joined mid-run: size the block map from the server's current stats.
        if (!m.numBlocks && e.stats?.kv_total) m.apply({ t: 0, type: "engine_started", kv_total: e.stats.kv_total });
        return;
      }
      modelRef.current.apply(e);
      dirty = true;
    };
    // Re-render at most ~10 times a second, and keep the clock moving while idle.
    const start = performance.now();
    let base: number | null = null;
    const timer = setInterval(() => {
      const m = modelRef.current;
      if (m.events > 0) {
        if (base === null) base = m.now - (performance.now() - start) / 1000;
        m.now = Math.max(m.now, base + (performance.now() - start) / 1000);
      }
      if (dirty || m.events > 0) setVersion((v) => v + 1);
      dirty = false;
    }, 100);
    return () => {
      clearInterval(timer);
      ws.close();
    };
  }, [baseUrl, enabled]);

  return { model: modelRef.current, version, status, servedModel: model, device, closeReason };
}

export type LoadOptions = { maxTokens: number; sharedPrefix: boolean };

/** Drives load through POST /v1/demo/load, so the server starts the requests itself. (A browser
 *  allows only ~6 open connections per server, so requests sent from the page would cap the batch at 6.) */
export function useLoadGenerator(baseUrl: string) {
  const [sent, setSent] = useState(0);
  const [errors, setErrors] = useState(0);
  const [rate, setRate] = useState(0);
  const opts = useRef<LoadOptions>({ maxTokens: 96, sharedPrefix: true });

  const start = useCallback(async (count: number) => {
    const { maxTokens, sharedPrefix } = opts.current;
    try {
      const res = await fetch(baseUrl.replace(/\/$/, "") + "/v1/demo/load", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ count, max_tokens: maxTokens, shared_prefix: sharedPrefix }),
      });
      if (!res.ok) throw new Error(String(res.status));
      setSent((n) => n + count);
    } catch {
      setErrors((n) => n + 1);
    }
  }, [baseUrl]);

  const burst = useCallback((n: number) => void start(n), [start]);

  // Poisson arrivals at `rate` requests per second.
  useEffect(() => {
    if (rate <= 0) return;
    let timer: number;
    const next = () => {
      timer = window.setTimeout(() => { void start(1); next(); }, (-Math.log(1 - Math.random()) / rate) * 1000);
    };
    next();
    return () => clearTimeout(timer);
  }, [rate, start]);

  return { burst, rate, setRate, sent, errors, opts };
}

// ------------------------------------------------------------------ replay: recorded JSONL trace

export type TraceInfo = { file: string; title: string; description: string };

export function useReplay() {
  const events = useRef<EngineEvent[]>([]);
  const modelRef = useRef(new EngineModel());
  const cursor = useRef(0);
  const [version, setVersion] = useState(0);
  const [time, setTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const timeRef = useRef(0);

  const rebuildTo = useCallback((t: number) => {
    const evs = events.current;
    let m = modelRef.current;
    if (t < m.now || cursor.current === 0) {
      m = modelRef.current = new EngineModel();
      cursor.current = 0;
    }
    while (cursor.current < evs.length && evs[cursor.current].t <= t) m.apply(evs[cursor.current++]);
    m.now = t;
    timeRef.current = t;
    setTime(t);
    setVersion((v) => v + 1);
  }, []);

  const load = useCallback((text: string) => {
    const evs = parseJsonl(text).sort((a, b) => a.t - b.t);
    if (!evs.length) throw new Error("no events in trace");
    events.current = evs;
    modelRef.current = new EngineModel();
    cursor.current = 0;
    setDuration(evs[evs.length - 1].t);
    rebuildTo(0);
    setPlaying(true);
  }, [rebuildTo]);

  const loadUrl = useCallback(async (url: string) => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      load(await res.text());
    } catch (e) {
      setError(`Could not load trace: ${(e as Error).message}`);
    } finally {
      setLoading(false);
    }
  }, [load]);

  const loadFile = useCallback(async (file: File) => {
    try {
      load(await file.text());
      setError(null);
    } catch (e) {
      setError(`Could not read ${file.name}: ${(e as Error).message}`);
    }
  }, [load]);

  useEffect(() => {
    if (!playing) return;
    let last = performance.now();
    let frame = 0, raf = 0;
    const tick = (now: number) => {
      const dt = ((now - last) / 1000) * speed;
      last = now;
      const t = Math.min(duration, timeRef.current + dt);
      // Apply every frame, re-render every 3rd (~20 fps) to keep React work small.
      if (++frame % 3 === 0 || t >= duration) rebuildTo(t);
      else {
        const evs = events.current, m = modelRef.current;
        while (cursor.current < evs.length && evs[cursor.current].t <= t) m.apply(evs[cursor.current++]);
        m.now = t;
        timeRef.current = t;
      }
      if (t >= duration) setPlaying(false);
      else raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, speed, duration, rebuildTo]);

  return {
    model: modelRef.current, version, time, duration, playing, speed, loading, error,
    setPlaying, setSpeed, seek: rebuildTo, loadUrl, loadFile,
    restart: () => { rebuildTo(0); setPlaying(true); },
  };
}
