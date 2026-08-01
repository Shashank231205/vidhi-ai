"use client";

import { useEffect, useRef } from "react";
import { ActivityIndicator } from "@/components/ActivityIndicator";
import { Label } from "@/components/ui";
import type { NodeStatus, TraceEvent } from "@/lib/api";

/**
 * The live reasoning trace.
 *
 * A run takes tens of seconds. Without this the user cannot tell a slow run
 * from a hung one — but the more important job is showing self-correction: a
 * `retrying` row is the critic rejecting weak retrieval, or the verifier
 * rejecting an ungrounded citation. That is the product's actual claim, so it
 * is presented as a record rather than hidden behind a spinner.
 */

const NODE_LABELS: Record<string, string> = {
  parse: "PARSE",
  retrieve: "RETRIEVE",
  critic: "CRITIC",
  expand: "EXPAND",
  analyze: "ANALYZE",
  stance: "STANCE",
  classify: "CLASSIFY",
  verify: "VERIFY",
  synthesise: "SYNTHESISE",
  clause: "CLAUSE",
  emit: "COMPLETE",
};

const GLYPH: Record<NodeStatus, string> = {
  started: "○",
  completed: "✓",
  retrying: "●",
  failed: "✕",
  skipped: "–",
};

const COLOUR: Record<NodeStatus, string> = {
  started: "var(--muted)",
  completed: "var(--color-verified)",
  retrying: "var(--color-risk-medium)",
  failed: "var(--color-risk-high)",
  skipped: "var(--muted)",
};

function Row({ event, live }: { event: TraceEvent; live?: boolean }) {
  const retrying = event.status === "retrying";
  // The last `started` row is the step currently running. Marking it keeps the
  // eye on the thing in flight rather than on the end of a static list.
  const inFlight = live && event.status === "started";

  return (
    <li
      className="animate-in flex items-baseline gap-2.5 rounded px-2 py-1.5 font-mono text-xs"
      style={
        retrying || inFlight
          ? { background: "var(--surface-sunken)" }
          : undefined
      }
    >
      <span
        aria-hidden
        className={`shrink-0 ${inFlight ? "animate-pulse-dot" : ""}`}
        style={{ color: COLOUR[event.status] }}
      >
        {GLYPH[event.status]}
      </span>

      <span className="shrink-0 tracking-wider" style={{ color: COLOUR[event.status] }}>
        {NODE_LABELS[event.node] ?? event.node.toUpperCase()}
      </span>

      {event.attempt > 1 && (
        <span className="shrink-0" style={{ color: "var(--color-risk-medium)" }}>
          #{event.attempt}
        </span>
      )}

      <span className="min-w-0 flex-1 text-muted">· {event.detail}</span>

      <span className="shrink-0 tabular-nums text-muted opacity-60">
        {(event.elapsed_ms / 1000).toFixed(1)}s
      </span>
    </li>
  );
}

export function AgentTrace({
  events,
  running,
  title = "Live agent trace",
}: {
  events: TraceEvent[];
  running: boolean;
  title?: string;
}) {
  const endRef = useRef<HTMLDivElement>(null);

  // Follow the tail while running, then stop — so a finished trace can be
  // scrolled back through without being yanked to the bottom.
  useEffect(() => {
    if (running) endRef.current?.scrollIntoView({ block: "nearest" });
  }, [events.length, running]);

  if (!events.length && !running) return null;

  const retries = events.filter((event) => event.status === "retrying").length;
  const completed = events.filter((event) => event.status === "completed").length;
  const progress = running
    ? Math.min(95, Math.round((completed / Math.max(events.length, 1)) * 100))
    : 100;

  return (
    <section
      className="rounded-[var(--radius-card)] border"
      style={{ background: "var(--surface)" }}
      aria-label="Agent reasoning trace"
    >
      <header className="flex items-start justify-between gap-4 border-b px-4 py-3">
        <div>
          <Label>{title}</Label>
          <h2 className="mt-1 font-serif text-lg leading-tight">
            {running ? "Verification loop in progress" : "Trace complete"}
          </h2>
        </div>

        <span
          className="shrink-0 rounded px-2 py-1 font-mono text-2xs tabular-nums"
          style={{
            background: running ? "var(--color-gold-600)" : "var(--surface-sunken)",
            color: running ? "white" : "var(--muted)",
          }}
        >
          {progress}%
        </span>
      </header>

      {/* aria-live so a screen reader hears progress rather than silence. */}
      <ol
        className={`max-h-80 space-y-0.5 overflow-y-auto ${events.length ? "p-2" : ""}`}
        aria-live="polite"
        aria-relevant="additions"
      >
        {events.map((event, index) => (
          <Row
            key={`${event.node}-${index}`}
            event={event}
            live={running && index === events.length - 1}
          />
        ))}
        <div ref={endRef} />
      </ol>

      <ActivityIndicator events={events} running={running} />

      {retries > 0 && (
        <footer className="border-t px-4 py-2.5">
          <span
            className="font-mono text-2xs tracking-wider uppercase"
            style={{ color: "var(--color-risk-medium)" }}
            title="The agent rejected its own intermediate result and retried"
          >
            {retries} self-correction{retries === 1 ? "" : "s"}
          </span>
        </footer>
      )}
    </section>
  );
}
