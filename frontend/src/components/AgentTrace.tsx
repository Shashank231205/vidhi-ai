"use client";

import { useEffect, useRef } from "react";
import type { NodeStatus, TraceEvent } from "@/lib/api";

/**
 * The live reasoning trace.
 *
 * An audit runs for tens of seconds. Without this the user sees a spinner and
 * cannot tell a slow run from a hung one. More importantly, it is where the
 * agent's self-correction becomes visible: a `retrying` event is the critic
 * rejecting weak retrieval or the verifier rejecting an ungrounded citation,
 * and that is the part worth watching.
 */

const NODE_LABELS: Record<string, string> = {
  parse: "Parsing",
  retrieve: "Retrieving law",
  critic: "Assessing context",
  expand: "Following citations",
  analyze: "Analysing",
  stance: "Assessing stance",
  classify: "Scoring risk",
  verify: "Verifying citations",
  synthesise: "Drafting memo",
  clause: "Clause complete",
  emit: "Complete",
};

const STATUS_COLOUR: Record<NodeStatus, string> = {
  started: "var(--muted)",
  completed: "var(--color-supports)",
  retrying: "var(--color-risk-medium)",
  failed: "var(--color-risk-high)",
  skipped: "var(--muted)",
};

function TraceRow({ event }: { event: TraceEvent }) {
  const isRetry = event.status === "retrying" || event.attempt > 1;

  return (
    <li className="animate-in flex items-baseline gap-3 py-1.5 text-sm">
      <span
        aria-hidden
        className={`mt-1.5 size-1.5 shrink-0 rounded-full ${
          event.status === "started" ? "animate-pulse-dot" : ""
        }`}
        style={{ background: STATUS_COLOUR[event.status] }}
      />
      <span className="w-36 shrink-0 text-xs font-medium">
        {NODE_LABELS[event.node] ?? event.node}
        {isRetry && (
          <span
            className="ml-1.5 font-normal"
            style={{ color: "var(--color-risk-medium)" }}
          >
            #{event.attempt}
          </span>
        )}
      </span>
      <span className="min-w-0 flex-1 text-muted">{event.detail}</span>
      <span className="shrink-0 font-mono text-2xs text-muted tabular-nums">
        {(event.elapsed_ms / 1000).toFixed(1)}s
      </span>
    </li>
  );
}

export function AgentTrace({
  events,
  running,
}: {
  events: TraceEvent[];
  running: boolean;
}) {
  const endRef = useRef<HTMLDivElement>(null);

  // Follow the tail while running, but stop once finished so the user can
  // scroll back through what happened without being yanked to the bottom.
  useEffect(() => {
    if (running) endRef.current?.scrollIntoView({ block: "nearest" });
  }, [events.length, running]);

  if (!events.length && !running) return null;

  const retries = events.filter((event) => event.status === "retrying").length;

  return (
    <section
      className="surface overflow-hidden"
      aria-label="Agent reasoning trace"
    >
      <header className="flex items-center justify-between border-b px-4 py-2.5">
        <h2 className="text-xs font-semibold tracking-wide uppercase text-muted">
          Reasoning trace
        </h2>
        {retries > 0 && (
          <span
            className="text-2xs font-medium"
            style={{ color: "var(--color-risk-medium)" }}
            title="The agent rejected its own intermediate results and retried"
          >
            {retries} self-correction{retries === 1 ? "" : "s"}
          </span>
        )}
      </header>

      {/* aria-live so a screen reader hears progress instead of silence. */}
      <ol
        className="max-h-72 overflow-y-auto px-4 py-2"
        aria-live="polite"
        aria-relevant="additions"
      >
        {events.map((event, index) => (
          <TraceRow key={`${event.node}-${index}`} event={event} />
        ))}
        {running && !events.length && (
          <li className="py-2 text-sm text-muted">Starting…</li>
        )}
        <div ref={endRef} />
      </ol>
    </section>
  );
}
