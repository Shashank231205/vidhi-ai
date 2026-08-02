"use client";

import { useEffect, useState } from "react";
import type { TraceEvent } from "@/lib/api";

/**
 * Shows that work is happening between trace events.
 *
 * The trace only updates when a node starts or finishes, and a single LLM call
 * can run for seconds. A static list through that gap is indistinguishable from
 * a hung request.
 *
 * So this names the current step, says what that step is for, counts the
 * seconds, and escalates its explanation the longer the wait runs. An
 * unexplained pause reads as broken; an explained one reads as honest.
 */

type Activity = {
  /** What the agent is doing, in the user's terms rather than the node's. */
  label: string;
  /** Why that step exists. Turns dead time into something worth reading. */
  detail: string;
};

const ACTIVITY: Record<string, Activity> = {
  parse: {
    label: "Splitting the document into clauses",
    detail: "Each clause is audited on its own, so findings point at specific text.",
  },
  retrieve: {
    label: "Searching statutes",
    detail: "Two searches at once — meaning and exact wording — then merged by rank.",
  },
  critic: {
    label: "Judging whether the retrieved law fits",
    detail: "Contract wording rarely matches statutory wording, so weak matches get re-searched.",
  },
  expand: {
    label: "Following the citation graph",
    detail: "Pulling the authorities this judgment itself relies on.",
  },
  analyze: {
    label: "Reasoning about the clause",
    detail: "Working out what the retrieved provisions mean for this specific wording.",
  },
  stance: {
    label: "Assessing how the precedent cuts",
    detail: "Deciding whether each authority supports or undermines the position.",
  },
  classify: {
    label: "Scoring severity",
    detail: "Rating exposure against clauses labelled by attorneys.",
  },
  verify: {
    label: "Checking every quote against its source",
    detail: "Each quote must appear in a provision actually retrieved — unverifiable claims are dropped.",
  },
  synthesise: {
    label: "Drafting the memo",
    detail: "Assembling the verified findings into a readable answer.",
  },
  clause: {
    label: "Finishing the clause",
    detail: "Collecting the verified findings for this clause.",
  },
  emit: {
    label: "Assembling the result",
    detail: "Final pass over everything that survived verification.",
  },
};

const FALLBACK: Activity = {
  label: "Working",
  detail: "The agent is between steps.",
};

const STARTING: Activity = {
  label: "Starting",
  detail: "Warming up the retrieval pipeline.",
};

/**
 * Escalating explanations for a wait.
 *
 * Each threshold answers the question the previous one leaves open. Silence at
 * 15s reads as slow; silence at 45s reads as broken. Saying nothing new while
 * time passes is what makes a wait feel indefinite.
 *
 * The free instance sleeps after 15 minutes idle and takes ~1 minute to wake,
 * so the first request of a session is genuinely slow for a reason worth
 * naming rather than leaving the user to guess.
 */
const WAIT_STAGES: { after: number; note: string }[] = [
  {
    after: 15,
    note: "Taking a moment — this step involves a full model call.",
  },
  {
    after: 35,
    note: "Still going. Six models are pooled; a busy one is skipped after 10 seconds.",
  },
  {
    after: 70,
    note: "Longer than usual. If the server was idle it is waking up, which takes about a minute.",
  },
  {
    after: 150,
    note: "This is slower than expected. The run has not failed — output appears as soon as a model responds.",
  },
];

function stageFor(seconds: number): string | null {
  let note: string | null = null;
  for (const stage of WAIT_STAGES) {
    if (seconds >= stage.after) note = stage.note;
  }
  return note;
}

export function ActivityIndicator({
  events,
  running,
}: {
  events: TraceEvent[];
  running: boolean;
}) {
  if (!running) return null;

  const latest = events.at(-1);
  const activity = latest ? (ACTIVITY[latest.node] ?? FALLBACK) : STARTING;

  // Keyed on the event count so each new event remounts the ticker. A remount
  // is a cleaner reset than clearing state from an effect, and it guarantees
  // the counter never shows the previous step's elapsed time.
  return <Ticker key={events.length} activity={activity} step={events.length} />;
}

function Ticker({ activity, step }: { activity: Activity; step: number }) {
  const [seconds, setSeconds] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => setSeconds((value) => value + 1), 1000);
    return () => clearInterval(timer);
  }, []);

  const note = stageFor(seconds);

  return (
    <div className="border-t px-4 py-3" aria-live="polite" aria-atomic="true">
      <div className="flex items-center gap-3">
        {/* Three staggered dots — motion that continues when nothing else does. */}
        <span className="flex shrink-0 gap-1" aria-hidden>
          {[0, 1, 2].map((index) => (
            <span
              key={index}
              className="animate-pulse-dot size-1.5 rounded-full"
              style={{
                background: note ? "var(--color-risk-medium)" : "var(--accent)",
                animationDelay: `${index * 0.18}s`,
              }}
            />
          ))}
        </span>

        <span className="min-w-0 flex-1 truncate text-sm">{activity.label}</span>

        {/* Step count gives a sense of progress the clock alone cannot: the
            seconds reset every step, so a rising step number is the signal
            that the run is advancing rather than stuck. */}
        {step > 0 && (
          <span className="shrink-0 font-mono text-2xs tabular-nums text-muted">
            step {step}
          </span>
        )}
        <span className="shrink-0 font-mono text-2xs tabular-nums text-muted">
          {seconds}s
        </span>
      </div>

      {/* Indented to the text column so the two lines read as one block. */}
      <p className="mt-1 pl-[1.875rem] text-xs text-muted">{activity.detail}</p>
      {note && <p className="mt-1 pl-[1.875rem] text-xs text-muted">{note}</p>}
    </div>
  );
}
