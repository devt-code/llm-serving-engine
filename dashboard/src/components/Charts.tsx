import { Area, AreaChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { Theme } from "../theme";

type Row = { t: number; tokPerS: number; running: number; waiting: number; kvUsed: number; kvCached: number };

function axes(theme: Theme) {
  return {
    grid: <CartesianGrid stroke={theme.grid} strokeDasharray="0" vertical={false} />,
    x: <XAxis dataKey="t" type="number" domain={["dataMin", "dataMax"]} tickFormatter={(v) => `${Math.round(v)}s`}
              stroke={theme.axis} tickLine={false} minTickGap={40} />,
    tooltip: (fmt: (v: number) => string) => (
      <Tooltip
        isAnimationActive={false}
        labelFormatter={(v) => `t = ${Number(v).toFixed(1)} s`}
        formatter={(v: number, name: string) => [fmt(v), name]}
        contentStyle={{ background: theme.surface, border: `1px solid ${theme.grid}`, borderRadius: 8, fontSize: 12 }}
        labelStyle={{ color: theme["text-secondary"] }}
        cursor={{ stroke: theme.axis, strokeWidth: 1 }}
      />
    ),
  };
}

function Legend({ items }: { items: [string, string][] }) {
  return (
    <div className="legend">
      {items.map(([c, name]) => (
        <span key={name}><i className="swatch" style={{ background: c, height: 3, borderRadius: 2 }} />{name}</span>
      ))}
    </div>
  );
}

export function ThroughputChart({ data, theme }: { data: Row[]; theme: Theme }) {
  const a = axes(theme);
  return (
    <ResponsiveContainer width="100%" height={170}>
      <AreaChart data={data} margin={{ top: 6, right: 8, bottom: 0, left: -12 }}>
        {a.grid}
        {a.x}
        <YAxis stroke={theme.axis} tickLine={false} axisLine={false} width={48} />
        {a.tooltip((v) => `${v} tok/s`)}
        <Area type="monotone" dataKey="tokPerS" name="Output tokens/s" stroke={theme["series-1"]} strokeWidth={2}
              fill={theme["series-1"]} fillOpacity={0.14} isAnimationActive={false} dot={false} />
      </AreaChart>
    </ResponsiveContainer>
  );
}

export function QueueChart({ data, theme }: { data: Row[]; theme: Theme }) {
  const a = axes(theme);
  return (
    <>
      <ResponsiveContainer width="100%" height={150}>
        <LineChart data={data} margin={{ top: 6, right: 8, bottom: 0, left: -12 }}>
          {a.grid}
          {a.x}
          <YAxis stroke={theme.axis} tickLine={false} axisLine={false} width={48} allowDecimals={false} />
          {a.tooltip((v) => `${v} requests`)}
          <Line type="stepAfter" dataKey="running" name="Running" stroke={theme["series-1"]} strokeWidth={2} dot={false} isAnimationActive={false} />
          <Line type="stepAfter" dataKey="waiting" name="Waiting" stroke={theme["series-2"]} strokeWidth={2} dot={false} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
      <Legend items={[[theme["series-1"], "Running"], [theme["series-2"], "Waiting"]]} />
    </>
  );
}

export function KvChart({ data, total, theme }: { data: Row[]; total: number; theme: Theme }) {
  const a = axes(theme);
  return (
    <>
      <ResponsiveContainer width="100%" height={150}>
        <AreaChart data={data} margin={{ top: 6, right: 8, bottom: 0, left: -12 }}>
          {a.grid}
          {a.x}
          <YAxis stroke={theme.axis} tickLine={false} axisLine={false} width={48} domain={[0, Math.max(total, 1)]} />
          {a.tooltip((v) => `${v} blocks`)}
          <Area type="stepAfter" dataKey="kvUsed" name="In use" stackId="kv" stroke={theme["series-1"]} strokeWidth={2}
                fill={theme["series-1"]} fillOpacity={0.35} isAnimationActive={false} dot={false} />
          <Area type="stepAfter" dataKey="kvCached" name="Cached, idle" stackId="kv" stroke={theme["seq-cached"]} strokeWidth={2}
                fill={theme["seq-cached"]} fillOpacity={0.5} isAnimationActive={false} dot={false} />
        </AreaChart>
      </ResponsiveContainer>
      <Legend items={[[theme["series-1"], "In use"], [theme["seq-cached"], "Cached, idle"]]} />
    </>
  );
}
