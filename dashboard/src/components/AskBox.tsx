import { useEffect, useRef, useState } from "react";

const MAX_TOKENS = 160;
// Short plain-text answers: on a CPU every token takes a while, and the reply is shown as plain text.
const SYSTEM = "You are a helpful assistant. Answer in two or three plain sentences, without Markdown.";
// Questions a 0.6B model answers well (checked by hand); deeper technical ones get vague answers.
const EXAMPLES = [
  "Why is batching requests faster for a language model server?",
  "Explain what a cache is, in simple terms.",
  "Write a haiku about GPUs.",
];

type State = "idle" | "waiting" | "streaming" | "done" | "error";

/** A prompt box that streams a reply from /v1/chat/completions. The request runs in the same engine
 *  as everyone else's, and onRequest reports its id so the request timeline can mark it. */
export function AskBox({ server, model, connected, onRequest }: {
  server: string; model: string; connected: boolean; onRequest: (id: string) => void;
}) {
  const [prompt, setPrompt] = useState("");
  const [reply, setReply] = useState("");
  const [state, setState] = useState<State>("idle");
  const [stats, setStats] = useState("");
  const abort = useRef<AbortController | null>(null);
  useEffect(() => () => abort.current?.abort(), []);
  const busy = state === "waiting" || state === "streaming";

  async function ask(question: string) {
    question = question.trim();
    if (!question || busy) return;
    const ctl = new AbortController();
    abort.current = ctl;
    setReply("");
    setStats("");
    setState("waiting");
    const start = performance.now();
    let first = 0, tokens = 0, finish = "", text = "", named = false;
    const summary = () => {
      const secs = (performance.now() - first) / 1000;
      const rate = tokens > 1 && secs > 0.05 ? ` at ${(tokens / secs).toFixed(1)} tokens/s` : "";
      return first ? `First token after ${((first - start) / 1000).toFixed(2)} s · ${tokens} tokens${rate}` : "";
    };
    try {
      const res = await fetch(server.replace(/\/$/, "") + "/v1/chat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: ctl.signal,
        body: JSON.stringify({
          model, stream: true, max_tokens: MAX_TOKENS, stream_options: { include_usage: true },
          temperature: 0.7, top_p: 0.8, // Qwen3's recommended settings without thinking
          messages: [{ role: "system", content: SYSTEM }, { role: "user", content: question }],
        }),
      });
      if (!res.ok || !res.body) {
        const body = await res.json().catch(() => null);
        throw new Error(body?.error?.message ?? `the server answered ${res.status}`);
      }
      const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
      let buf = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += value;
        let end;
        while ((end = buf.indexOf("\n\n")) >= 0) {  // server-sent events: "data: {...}\n\n"
          const data = buf.slice(0, end).replace(/^data: /, "");
          buf = buf.slice(end + 2);
          if (data === "[DONE]") continue;
          const chunk = JSON.parse(data);
          if (!named && chunk.id) {
            named = true;
            onRequest(chunk.id);
          }
          if (chunk.usage) tokens = chunk.usage.completion_tokens;
          const choice = chunk.choices?.[0];
          if (choice?.finish_reason) finish = choice.finish_reason;
          if (choice?.delta?.content) {
            if (!first) {
              first = performance.now();
              setState("streaming");
            }
            if (!chunk.usage) tokens += 1;  // one token per chunk, until the usage chunk gives the count
            text += choice.delta.content;
            setReply(text);
          }
        }
      }
      setStats(summary() + (finish === "length" ? ` · stopped at ${MAX_TOKENS} tokens` : ""));
      setState("done");
    } catch (e) {
      if (ctl.signal.aborted) {
        setStats(text ? `${summary()} · stopped` : "");
        setState(text ? "done" : "idle");
        return;
      }
      setStats(`Couldn't get a reply: ${(e as Error).message}`);
      setState("error");
    }
  }

  return (
    <section className="card ask">
      <h2>Ask the model</h2>
      <p className="hint">
        Your prompt joins the same batch as everyone else's. Watch for the row marked <b>you</b> in the request timeline below.
      </p>
      <form onSubmit={(e) => { e.preventDefault(); void ask(prompt); }}>
        <input type="text" value={prompt} maxLength={400} aria-label="Prompt" disabled={!connected}
               placeholder={`Ask anything, e.g. "${EXAMPLES[0]}"`} onChange={(e) => setPrompt(e.target.value)} />
        {busy
          ? <button type="button" className="btn" onClick={() => abort.current?.abort()}>Stop</button>
          : <button type="submit" className="btn primary" disabled={!connected || !prompt.trim()}>Send</button>}
      </form>
      {state === "idle" && (
        <div className="examples">
          Try:
          {EXAMPLES.map((ex) => (
            <button key={ex} type="button" className="chip" disabled={!connected}
                    onClick={() => { setPrompt(ex); void ask(ex); }}>{ex}</button>
          ))}
        </div>
      )}
      {(reply || state === "waiting") && (
        <div className={`reply${reply ? "" : " waiting"}`} aria-live="polite">
          {reply || "Waiting for the first token…"}
          {state === "streaming" && <span className="caret" />}
        </div>
      )}
      {stats && <div className={`ask-stats${state === "error" ? " error" : ""}`}>{stats}</div>}
    </section>
  );
}
