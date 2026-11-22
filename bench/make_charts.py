"""Render README charts from bench/results/*.json into docs/ (light and dark versions).

  python bench/make_charts.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "bench" / "results"
OUT = ROOT / "docs"

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  axis="#c3c2b7", s1="#2a78d6", s2="#eb6834", dim="#b7d3f6"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 axis="#383835", s1="#3987e5", s2="#d95926", dim="#184f95"),
}


def style(ax, t):
    ax.set_facecolor(t["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.tick_params(colors=t["muted"], length=0, labelsize=9)
    ax.grid(axis="x", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


def hbar(name, title, subtitle, labels, values, fmt, t, highlight=None, xlabel="", accent=()):
    """highlight: indices drawn in the main color (others dimmed); accent: indices in the second color."""
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(labels) + 1.35), dpi=160)
    fig.patch.set_facecolor(t["surface"])
    style(ax, t)
    y = range(len(labels))[::-1]
    colors = [t["s2"] if i in accent else t["s1"] if (highlight is None or i in highlight) else t["dim"]
              for i in range(len(labels))]
    bars = ax.barh(list(y), values, color=colors, height=0.62)
    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height() / 2, fmt(v),
                va="center", ha="left", fontsize=9, color=t["ink"])
    ax.set_yticks(list(y), labels, fontsize=9.5, color=t["ink2"])
    ax.set_xlim(0, max(values) * 1.16)
    if xlabel:
        ax.set_xlabel(xlabel, color=t["muted"], fontsize=9)
    fig.text(0.012, 0.97, title, fontsize=12, fontweight="bold", color=t["ink"], va="top")
    fig.text(0.012, 0.97 - 0.32 / (0.42 * len(labels) + 1.35), subtitle, fontsize=9, color=t["ink2"], va="top")
    fig.subplots_adjust(left=0.33, right=0.97, top=1 - 0.78 / (0.42 * len(labels) + 1.35), bottom=0.3 / (0.42 * len(labels) + 1.35) + 0.06)
    return fig


def save(fig, name, mode):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}-{mode}.png", facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    made = []
    ladder = RES / "ladder.json"
    if ladder.exists():
        d = json.loads(ladder.read_text())
        rows = d["results"]
        labels = [r["label"] for r in rows]
        where = d["gpu"] if d["device"].startswith("cuda") else "Apple M3 Pro GPU (MPS)" if d["device"] == "mps" else d["device"]
        sub = (f"{d['workload']['num_requests']} requests, {d['workload']['output_tokens']:,} output tokens, "
               f"{d['dtype']} on {where}. Same KV memory for every config of ours.")
        for mode, t in THEMES.items():
            save(hbar("ladder", "Output throughput, one optimization at a time", sub, labels,
                      [r["output_tok_per_s"] for r in rows], lambda v: f"{v:,.0f} tok/s", t,
                      highlight={len(rows) - 1} if rows[-1]["config"] != "vllm" else {len(rows) - 2, len(rows) - 1}),
                 "ladder-throughput", mode)
        ttft_rows = [r for r in rows if "ttft_p50_s" in r]
        if ttft_rows:
            for mode, t in THEMES.items():
                save(hbar("ttft", "Median time to first token", "Same workload; all requests arrive at once.",
                          [r["label"] for r in ttft_rows], [r["ttft_p50_s"] for r in ttft_rows],
                          lambda v: f"{v:.2f} s", t, highlight={len(ttft_rows) - 1}),
                     "ladder-ttft", mode)
        made += ["ladder-throughput", "ladder-ttft"]

    gpu = RES / "ladder_gpu.json"
    if gpu.exists():
        d = json.loads(gpu.read_text())
        rows = d["results"]
        names = {"vllm": "vLLM, default memory", "vllm_kv": "vLLM, same KV memory as ours"}
        ours = [i for i, r in enumerate(rows) if r["config"] in ("static", "continuous", "paged", "prefix")]
        best = max(ours, key=lambda i: rows[i]["output_tok_per_s"])
        vllm = {i for i, r in enumerate(rows) if r["config"] in names}
        same = next((r for r in rows if r["config"] == "vllm_kv"), None)
        version = next((r["vllm_version"] for r in rows if "vllm_version" in r), None)
        title = (f"On an {d['gpu'].removeprefix('NVIDIA ')}, our best is "
                 f"{100 * rows[best]['output_tok_per_s'] / same['output_tok_per_s']:.0f}% of vLLM at equal memory"
                 if same else f"Our engine and vLLM on an {d['gpu']}")
        sub = (f"{d['workload']['num_requests']} requests, {d['workload']['output_tokens']:,} output tokens, {d['dtype']}"
               + (f", vLLM {version}" if version else "") + f". Our configs get {d['workload']['kv_budget_gb']} GB of KV memory.")
        for mode, t in THEMES.items():
            save(hbar("gpu", title, sub, [names.get(r["config"], r["label"]) for r in rows],
                      [r["output_tok_per_s"] for r in rows], lambda v: f"{v:,.0f} tok/s", t,
                      highlight={best}, accent=vllm), "gpu-vs-vllm", mode)
        made.append("gpu-vs-vllm")

    paged = RES / "paged.json"
    if paged.exists():
        d = json.loads(paged.read_text())
        pick = [r for r in d["results"] if r["budget_slots"] == 32 and (r["kv_cache"] == "slot" or r["max_num_seqs"] in (64, 256))]
        labels = [("Slot cache" if r["kv_cache"] == "slot" else f"Paged, cap {r['max_num_seqs']}") for r in pick]
        for mode, t in THEMES.items():
            save(hbar("paged", "Requests running at once, same 1.17 GB of KV memory",
                      "Mean over decode steps. Paged blocks are handed out as requests grow.",
                      labels, [r["mean_running_decode"] for r in pick], lambda v: f"{v:.0f}", t,
                      highlight={len(pick) - 1}), "paged-concurrency", mode)
        made.append("paged-concurrency")

    prefix = RES / "prefix.json"
    if prefix.exists():
        d = json.loads(prefix.read_text())
        pick = [r for r in d["results"] if r["arrivals"].startswith("poisson")]
        labels = [f"Prefix caching {'on' if r['prefix_caching'] else 'off'}" for r in pick]
        for mode, t in THEMES.items():
            save(hbar("prefix", "Time to first token with a shared system prompt",
                      f"Median, {d['workload']['num_requests']} requests arriving at {d['workload']['poisson_rate']:g}/s, "
                      f"{d['workload']['prompt_tokens']}-token prompts.",
                      labels, [r["ttft_p50_ms"] for r in pick], lambda v: f"{v:,.0f} ms", t,
                      highlight={i for i, r in enumerate(pick) if r["prefix_caching"]}), "prefix-ttft", mode)
        made.append("prefix-ttft")
    print("charts:", ", ".join(made), "->", OUT)


if __name__ == "__main__":
    main()
