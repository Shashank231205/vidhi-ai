"use client";

import { useEffect, useState } from "react";
import type { TraceEvent } from "@/lib/api";

/**
 * Shows that work is happening between trace events.
 *
 * The trace only updates when a node starts or finishes, and a single LLM call
 * can take 20+ seconds — longer when the free tier's token budget is exhausted
 * and the router is waiting out a rate limit. A static list through that gap is
 * indistinguishable from a hung request.
 *
 * So this names the current step, counts the seconds it has been running, and
 * says plainly when a long wait is the provider's throughput rather than a
 * stall. An unexplained pause reads as broken; an explained one reads as
 * honest.
 */

const ACTIVITY: Record<string, string> = {
  parse: "Splitting the document into clauses",
  retrieve: "Searching statutes",
  critic: "Judging whether the retrieved law fits",
  expand: "Following the citation graph",
  analyze: "Reasoning about the clause",
  stance: "Assessing how the precedent cuts",
  classify: "Scoring severity",
  verify: "Checking every quote against its source",
  synthesise: "Drafting the memo",
  clause: "Finishing the clause",
  emit: "Assembling the result",
};

/** Past this, the pause needs an explanation rather than a bare spinner. */
const SLOW_AFTER_S = 12;

export function ActivityIndicator({
  events,
  running,
}: {
  events: TraceEvent[];
  running: boolean;
}) {
  if (!running) return null;

  const latest = events.at(-1);

  // Keyed on the event count so each new event remounts the ticker. A remount
  // is a cleaner reset than clearing state from an effect, and it guarantees
  // the counter never shows the previous step's elapsed time.
  return (
    <Ticker
      key={events.length}
      activity={latest ? (ACTIVITY[latest.node] ?? "Working") : "Starting the audit"}
    />
  );
}

function Ticker({ activity }: { activity: string }) {
  const [seconds, setSeconds] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => setSeconds((value) => value + 1), 1000);
    return () => clearInterval(timer);
  }, []);

  // A long silence on the free tier is almost always the provider's rate
  // limit. Saying so is more useful than a spinner implying we are stuck.
  const slow = seconds >= SLOW_AFTER_S;

  return (
    <div
      className="flex items-center gap-3 border-t px-4 py-3"
      aria-live="polite"
      aria-atomic="true"
    >
      {/* Three staggered dots — motion that continues when nothing else does. */}
      <span className="flex shrink-0 gap-1" aria-hidden>
        {[0, 1, 2].map((index) => (
          <span
            key={index}
            className="animate-pulse-dot size-1.5 rounded-full"
            style={{
              background: slow ? "var(--color-risk-medium)" : "var(--accent)",
              animationDelay: `${index * 0.18}s`,
            }}
          />
        ))}
      </span>

      <span className="min-w-0 flex-1 text-sm">
        <span className="text-muted">{activity}</span>
        {slow && (
          <span className="block text-xs text-muted">
            Still going — the free LLM tier limits throughput, so a busy run
            waits for its token budget to refill.
          </span>
        )}
      </span>

      <span className="shrink-0 font-mono text-2xs tabular-nums text-muted">
        {seconds}s
      </span>
    </div>
  );
}
