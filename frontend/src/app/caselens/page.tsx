"use client";

import { useCallback, useState } from "react";
import { AgentTrace } from "@/components/AgentTrace";
import {
  Button,
  Citation,
  EmptyState,
  ErrorState,
  Quote,
  StanceBadge,
} from "@/components/ui";
import type { AssessedCase, ResearchResult, Stance } from "@/lib/api";
import { useAgentStream } from "@/lib/useAgentStream";

const SAMPLE = `Our client, a software vendor, entered a services agreement containing a clause barring it from working with any competitor of the customer for three years after termination, anywhere in India. The customer now seeks to enforce it.

We argue the restraint is void and unenforceable.`;

const STANCE_ORDER: Record<Stance, number> = {
  supports: 0,
  undermines: 1,
  neutral: 2,
};

function CaseCard({ item }: { item: AssessedCase }) {
  return (
    <article className="surface animate-in p-5">
      <header className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h3 className="font-medium">{item.case_title}</h3>
          <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted">
            {item.cited_by_count > 0 && (
              <span title="How many ingested judgments rely on this one">
                cited by {item.cited_by_count}
              </span>
            )}
            {/* Worth surfacing: this case was not found by searching the facts
                — it was reached through the citation graph, which is the whole
                argument for keeping one. */}
            {item.via_citation_graph && (
              <span style={{ color: "var(--color-brand-600)" }}>
                via citation graph
              </span>
            )}
          </div>
        </div>
        <StanceBadge stance={item.stance} confidence={item.confidence} />
      </header>

      <p className="mt-3 text-sm font-medium">{item.holding}</p>
      <p className="mt-2 text-sm text-muted">{item.reasoning}</p>

      <div className="mt-4 space-y-1.5">
        <Citation>{item.citation}</Citation>
        <Quote>“{item.quote}”</Quote>
      </div>
    </article>
  );
}

function Memo({ result }: { result: ResearchResult }) {
  if (!result.memo) return null;
  const { summary, supporting_argument, risks, gaps } = result.memo;

  return (
    <section className="surface p-5">
      <h2 className="text-xs font-semibold tracking-wide uppercase text-muted">
        Research memo
      </h2>
      <p className="mt-3 text-sm">{summary}</p>

      <div className="mt-4 space-y-4 text-sm">
        <div>
          <h3 className="text-xs font-semibold" style={{ color: "var(--color-supports)" }}>
            Supporting argument
          </h3>
          <p className="mt-1">{supporting_argument}</p>
        </div>
        <div>
          <h3
            className="text-xs font-semibold"
            style={{ color: "var(--color-undermines)" }}
          >
            What the other side will rely on
          </h3>
          <p className="mt-1">{risks}</p>
        </div>
        {/* Shown prominently: a memo that hides its gaps is the one that gets
            relied on and then fails. */}
        {gaps && (
          <div>
            <h3 className="text-xs font-semibold text-muted">
              Not settled by these authorities
            </h3>
            <p className="mt-1 text-muted">{gaps}</p>
          </div>
        )}
      </div>
    </section>
  );
}

export default function CaseLensPage() {
  const [facts, setFacts] = useState("");
  const { events, result, running, error, start, cancel } =
    useAgentStream<ResearchResult>("/caselens/research/stream");

  const run = useCallback(() => {
    if (facts.trim().length < 20) return;
    void start({ facts, limit: 6, expand: true });
  }, [facts, start]);

  const sorted = result
    ? [...result.cases].sort(
        (a, b) =>
          STANCE_ORDER[a.stance] - STANCE_ORDER[b.stance] ||
          b.confidence - a.confidence,
      )
    : [];

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,26rem)_minmax(0,1fr)]">
      <section className="space-y-4">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">CaseLens</h1>
          <p className="mt-1 text-sm text-muted">
            Describe the facts and the position you are arguing. Retrieved
            judgments are assessed for whether they help or hurt it.
          </p>
        </div>

        <label className="block">
          <span className="text-xs font-medium">Fact pattern and position</span>
          <textarea
            value={facts}
            onChange={(event) => setFacts(event.target.value)}
            rows={14}
            placeholder="Set out the facts, then the proposition you want to establish…"
            className="legal-text mt-1.5 w-full resize-y rounded-lg border bg-[var(--surface)] px-3 py-2"
          />
        </label>

        <div className="flex flex-wrap items-center gap-2">
          <Button onClick={run} disabled={running || facts.trim().length < 20}>
            {running ? "Researching…" : "Find precedents"}
          </Button>
          {running && (
            <Button variant="secondary" onClick={cancel}>
              Cancel
            </Button>
          )}
          <Button
            variant="ghost"
            onClick={() => setFacts(SAMPLE)}
            disabled={running}
          >
            Use sample
          </Button>
        </div>
      </section>

      <section className="space-y-4">
        <AgentTrace events={events} running={running} />

        {error && <ErrorState message={error} onRetry={run} />}

        {result && (
          <div className="surface flex flex-wrap items-center gap-x-6 gap-y-2 px-5 py-4 text-sm">
            <span>
              <strong>{result.cases.length}</strong> authorit
              {result.cases.length === 1 ? "y" : "ies"}
            </span>
            <span className="flex gap-3 text-xs">
              {(["supports", "undermines", "neutral"] as const).map(
                (stance) =>
                  result.stance_summary[stance] > 0 && (
                    <span
                      key={stance}
                      style={{
                        color:
                          stance === "neutral"
                            ? "var(--muted)"
                            : `var(--color-${stance})`,
                      }}
                    >
                      {result.stance_summary[stance]} {stance}
                    </span>
                  ),
              )}
            </span>
            {result.discarded > 0 && (
              <span className="text-xs text-muted">
                {result.discarded} discarded as unverifiable
              </span>
            )}
            <span className="ml-auto font-mono text-2xs text-muted tabular-nums">
              {(result.elapsed_ms / 1000).toFixed(1)}s
            </span>
          </div>
        )}

        {result && <Memo result={result} />}

        {result && sorted.length === 0 && (
          <div className="surface">
            <EmptyState
              title="No judgments matched"
              description="Nothing in the ingested corpus addressed these facts. The judgment corpus is a sample, not the full body of Indian case law."
            />
          </div>
        )}

        <div className="space-y-3">
          {sorted.map((item) => (
            <CaseCard key={item.chunk_id} item={item} />
          ))}
        </div>

        {!result && !running && !error && (
          <div className="surface">
            <EmptyState
              title="No research yet"
              description="Describe a fact pattern to retrieve and assess precedents. The reasoning trace appears here as the agent works."
            />
          </div>
        )}
      </section>
    </div>
  );
}
