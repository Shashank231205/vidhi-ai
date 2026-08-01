"use client";

import { useCallback, useState } from "react";
import { AgentTrace } from "@/components/AgentTrace";
import { SourceDrawer } from "@/components/SourceDrawer";
import {
  Button,
  EmptyState,
  ErrorState,
  Label,
  StanceBadge,
} from "@/components/ui";
import type { AssessedCase, ResearchResult, Stance } from "@/lib/api";
import { useAgentStream } from "@/lib/useAgentStream";

const SAMPLE_FACTS = `A supplier terminated a software implementation agreement after the customer withheld milestone payments, alleging material delay and defective performance.`;

const SAMPLE_POSITION = `The termination was contractually justified and damages are not recoverable without proof of loss.`;

const STANCE_ORDER: Record<Stance, number> = {
  supports: 0,
  undermines: 1,
  neutral: 2,
};

function CaseCard({
  item,
  onOpenSource,
}: {
  item: AssessedCase;
  onOpenSource: () => void;
}) {
  return (
    <article
      className="animate-in rounded-[var(--radius-card)] border p-5"
      style={{ background: "var(--surface)" }}
    >
      <header className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h3 className="font-serif text-lg leading-tight">{item.case_title}</h3>
          <div className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-2xs tracking-wider uppercase text-muted">
            <span>{item.citation}</span>
            {item.cited_by_count > 0 && (
              <span title="How many ingested judgments rely on this one">
                · cited by {item.cited_by_count}
              </span>
            )}
            {/* Worth surfacing: this case was not found by searching the facts,
                it was reached through the citation graph — which is the whole
                argument for keeping one. */}
            {item.via_citation_graph && (
              <span style={{ color: "var(--accent)" }}>· via citation graph</span>
            )}
          </div>
        </div>
        <StanceBadge stance={item.stance} />
      </header>

      <p className="mt-3.5 text-sm">{item.holding}</p>
      <p className="mt-2 text-sm text-muted">{item.reasoning}</p>

      <blockquote
        className="legal-text mt-4 border-l-2 pl-4"
        style={{ borderColor: "var(--accent)" }}
      >
        “{item.quote}”
      </blockquote>

      <footer className="mt-4 flex items-center justify-between border-t pt-3.5">
        <span className="font-mono text-2xs tracking-wider uppercase text-muted">
          Confidence {item.confidence.toFixed(2)}
        </span>
        <button
          onClick={onOpenSource}
          className="font-mono text-2xs tracking-wider uppercase transition-opacity hover:opacity-70"
          style={{ color: "var(--accent)" }}
        >
          Open source →
        </button>
      </footer>
    </article>
  );
}

function Memo({ result }: { result: ResearchResult }) {
  if (!result.memo) return null;
  const { summary, supporting_argument, risks, gaps } = result.memo;

  return (
    <section
      className="rounded-[var(--radius-card)] border p-5"
      style={{ background: "var(--surface-sunken)" }}
    >
      <Label>Research memo</Label>
      <p className="mt-3 text-sm">{summary}</p>

      <div className="mt-5 space-y-4 border-t pt-4 text-sm">
        <div>
          <h3
            className="font-mono text-2xs tracking-wider uppercase"
            style={{ color: "var(--color-supports)" }}
          >
            Supporting argument
          </h3>
          <p className="mt-1.5">{supporting_argument}</p>
        </div>
        <div>
          <h3
            className="font-mono text-2xs tracking-wider uppercase"
            style={{ color: "var(--color-undermines)" }}
          >
            What the other side will rely on
          </h3>
          <p className="mt-1.5">{risks}</p>
        </div>
        {/* Shown prominently: a memo that hides its gaps is the one that gets
            relied on and then fails. */}
        {gaps && (
          <div>
            <h3 className="label-mono">Not settled by these authorities</h3>
            <p className="mt-1.5 text-muted">{gaps}</p>
          </div>
        )}
      </div>
    </section>
  );
}

export default function CaseLensPage() {
  const [facts, setFacts] = useState("");
  const [position, setPosition] = useState("");
  const [openSource, setOpenSource] = useState<{ id: string; quote: string } | null>(
    null,
  );

  const { events, result, running, error, start, cancel } =
    useAgentStream<ResearchResult>("/caselens/research/stream");

  const run = useCallback(() => {
    if (facts.trim().length < 20) return;
    // Facts and position are separate inputs for the user but one prompt for
    // the model: stance only means anything relative to a stated position.
    const combined = position.trim()
      ? `${facts.trim()}\n\nPosition argued: ${position.trim()}`
      : facts.trim();
    void start({ facts: combined, limit: 6, expand: true });
  }, [facts, position, start]);

  const cases = result
    ? [...result.cases].sort(
        (a, b) =>
          STANCE_ORDER[a.stance] - STANCE_ORDER[b.stance] ||
          b.confidence - a.confidence,
      )
    : [];

  const started = running || result !== null || error !== null;

  return (
    <>
      <div className="grid gap-8 lg:grid-cols-[minmax(0,24rem)_minmax(0,1fr)]">
        <section className="space-y-5">
          <div>
            <Label>Frame the question</Label>
            <h1 className="mt-3 text-2xl leading-tight">
              Start with the disputed fact pattern.
            </h1>
            <p className="mt-2 text-sm text-muted">
              Set out what happened and the position you need authorities to
              test.
            </p>
          </div>

          <label className="block">
            <Label>Fact pattern</Label>
            <textarea
              value={facts}
              onChange={(event) => setFacts(event.target.value)}
              rows={7}
              placeholder="What happened…"
              className="legal-text mt-1.5 w-full resize-y rounded border px-3.5 py-3"
              style={{ background: "var(--surface)" }}
            />
          </label>

          <label className="block">
            <Label>Argued position</Label>
            <textarea
              value={position}
              onChange={(event) => setPosition(event.target.value)}
              rows={4}
              placeholder="The proposition you want to establish…"
              className="legal-text mt-1.5 w-full resize-y rounded border px-3.5 py-3"
              style={{ background: "var(--surface)" }}
            />
          </label>

          <div className="flex flex-wrap items-center gap-2">
            <Button onClick={run} disabled={running || facts.trim().length < 20}>
              {running ? "Researching…" : "Research verified precedents"}
            </Button>
            {running && (
              <Button variant="secondary" onClick={cancel}>
                Cancel
              </Button>
            )}
            <Button
              variant="ghost"
              onClick={() => {
                setFacts(SAMPLE_FACTS);
                setPosition(SAMPLE_POSITION);
              }}
              disabled={running}
            >
              Use sample
            </Button>
          </div>

          {!started && (
            <div className="border-t pt-5">
              <Label>A research record, not a chat transcript</Label>
              <ol className="mt-3 space-y-3 text-sm">
                {[
                  ["Retrieve", "Search the issue across the ingested corpus."],
                  ["Critic", "Test relevance and distinguish adverse authority."],
                  ["Verify", "Match every proposition to its source text."],
                ].map(([name, detail], index) => (
                  <li key={name} className="flex gap-3">
                    <span
                      className="font-mono text-2xs"
                      style={{ color: "var(--accent)" }}
                    >
                      0{index + 1}
                    </span>
                    <span>
                      <strong className="font-medium">{name}</strong>
                      <span className="block text-muted">{detail}</span>
                    </span>
                  </li>
                ))}
              </ol>
            </div>
          )}
        </section>

        <section className="space-y-5">
          <AgentTrace events={events} running={running} title="Research trace" />

          {error && (
            <ErrorState
              title="We could not complete the research"
              message={error}
              onRetry={run}
            />
          )}

          {result && (
            <div
              className="flex flex-wrap items-center gap-x-6 gap-y-2 rounded-[var(--radius-card)] border px-5 py-4"
              style={{ background: "var(--surface-sunken)" }}
            >
              <span className="font-serif text-lg">
                {cases.length} verified authorit{cases.length === 1 ? "y" : "ies"}
              </span>
              <span className="flex gap-3 font-mono text-2xs tracking-wider uppercase">
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
                <span className="font-mono text-2xs text-muted">
                  {result.discarded} discarded as unverifiable
                </span>
              )}
              <span className="ml-auto font-mono text-2xs tabular-nums text-muted">
                {(result.elapsed_ms / 1000).toFixed(1)}s
              </span>
            </div>
          )}

          {result && <Memo result={result} />}

          {cases.length > 0 && <Label>Authorities worth reading first</Label>}

          <div className="space-y-4">
            {cases.map((item) => (
              <CaseCard
                key={item.chunk_id}
                item={item}
                onOpenSource={() =>
                  setOpenSource({ id: item.chunk_id, quote: item.quote })
                }
              />
            ))}
          </div>

          {result && cases.length === 0 && (
            <div className="rounded-[var(--radius-card)] border">
              <EmptyState
                title="No judgments matched"
                description="Nothing in the ingested corpus addressed these facts. The judgment corpus is a sample, not the full body of Indian case law."
              />
            </div>
          )}

          {!started && (
            <div className="rounded-[var(--radius-card)] border">
              <EmptyState
                title="No research yet"
                description="Describe a fact pattern to retrieve and assess precedents. The reasoning trace appears here as the agent works."
              />
            </div>
          )}
        </section>
      </div>

      <SourceDrawer
        chunkId={openSource?.id ?? null}
        quote={openSource?.quote}
        onClose={() => setOpenSource(null)}
      />
    </>
  );
}
