import { useEffect, useState } from "react";

// Chart libraries take colors as props, and CSS variables don't work in SVG presentation
// attributes everywhere. So read the current token values and re-read when the theme changes.
const TOKENS = [
  "surface", "text-primary", "text-secondary", "text-muted", "grid", "axis",
  "series-1", "series-2", "series-3", "seq-cached", "waiting", "free",
] as const;
export type Theme = Record<(typeof TOKENS)[number], string>;

function read(): Theme {
  const s = getComputedStyle(document.documentElement);
  return Object.fromEntries(TOKENS.map((k) => [k, s.getPropertyValue(`--${k}`).trim()])) as Theme;
}

export function useTheme(): Theme {
  const [theme, setTheme] = useState<Theme>(read);
  useEffect(() => {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const update = () => setTheme(read());
    mq.addEventListener("change", update);
    const obs = new MutationObserver(update);
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    return () => {
      mq.removeEventListener("change", update);
      obs.disconnect();
    };
  }, []);
  return theme;
}
