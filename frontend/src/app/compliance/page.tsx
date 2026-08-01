"use client";

import { useCallback, useRef, useState } from "react";
import { AgentTrace } from "@/components/AgentTrace";
import { SourceDrawer } from "@/components/SourceDrawer";
import {
  Button,
  EmptyState,
  ErrorState,
  Label,
  RiskBadge,
  VerifiedMark,
} from "@/components/ui";
import { ApiError, api, type AuditResult, type Finding, type Risk } from "@/lib/api";
import { useAgentStream } from "@/lib/useAgentStream";

const SAMPLE = `3. Data Collection. The Company may collect, store and process any and all personal data of the User, including sensitive personal data, for any purpose it deems fit, without providing notice to the User and without obtaining separate consent.

4. Retention & Deletion. Upon termination, the Company may retain all User personal data for as long as reasonably required for its internal business purposes.

5. Liability. The User agrees to indemnify the Company against all claims without limitation, and this obligation shall survive termination.`;

const RISK_ORDER: Record<Risk, number> = { high: 0, medium: 1, low: 2 };

function FindingCard({
  finding,
  onOpenSource,
}: {
  finding: Finding;
  onOpenSource: () => void;
}) {
  return (
    <article
      className="animate-in rounded-[var(--radius-card)] border p-5"
      style={{ background: "var(--surface)" }}
    >
      <header className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          {finding.clause_label && <Label>{finding.clause_label}</Label>}
          <h3 className="mt-1.5 font-serif text-xl leading-tight">{finding.issue}</h3>
        </div>
        <RiskBadge risk={finding.risk} source={finding.risk_source} />
      </header>

      <p className="mt-3 text-sm text-muted">{finding.explanation}</p>

      <div
        className="mt-5 rounded-[var(--radius-card)] border p-4"
        style={{ background: "var(--surface-sunken)" }}
      >
        <div className="flex items-start justify-between gap-3">
          <Label>{finding.citation}</Label>
          <VerifiedMark />
        </div>

        <blockquote
          className="legal-text mt-3 border-l-2 pl-4"
          style={{ borderColor: "var(--accent)" }}
        >
          “{finding.quote}”
        </blockquote>

        {/* The whole guarantee in one control: read the provision this was
            checked against, rather than taking the summary on trust. */}
        <button
          onClick={onOpenSource}
          className="mt-4 flex w-full items-center justify-between rounded border px-3 py-2.5 font-mono text-2xs tracking-wider uppercase transition-colors hover:border-[var(--border-strong)]"
          style={{ background: "var(--surface)" }}
        >
          Open verified source
          <span aria-hidden>→</span>
        </button>
      </div>

      <div className="mt-5">
        <Label>Recommended redline</Label>
        <p className="mt-1.5 text-sm">{finding.suggested_fix}</p>
      </div>
    </article>
  );
}

export default function CompliancePage() {
  const [text, setText] = useState("");
  const [title, setTitle] = useState("Pasted contract");
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [openSource, setOpenSource] = useState<{ id: string; quote: string } | null>(
    null,
  );
  const fileInput = useRef<HTMLInputElement>(null);

  const { events, result, running, error, start, cancel } =
    useAgentStream<AuditResult>("/compliance/audit/stream");

  const runAudit = useCallback(() => {
    if (text.trim().length < 20) return;
    setUploadError(null);
    // Capped for the demo: a full contract is hundreds of LLM calls and meets
    // the provider's rate limit long before it finishes.
    void start({ text, title, max_clauses: 8 });
  }, [start, text, title]);

  const onFile = async (file: File) => {
    setUploading(true);
    setUploadError(null);
    try {
      const uploaded = await api.upload(file, "contract");
      const search = await api.search(uploaded.title, 1);
      setTitle(uploaded.title);
      setText(
        search.hits[0]?.content ??
          "The document was indexed but its text could not be loaded. Paste the clauses to audit them.",
      );
    } catch (caught) {
      setUploadError(
        caught instanceof ApiError ? caught.message : "The upload failed.",
      );
    } finally {
      setUploading(false);
    }
  };

  const findings = result
    ? [...result.findings].sort((a, b) => RISK_ORDER[a.risk] - RISK_ORDER[b.risk])
    : [];

  const started = running || result !== null || error !== null;

  return (
    <>
      <div className="grid gap-8 lg:grid-cols-[minmax(0,24rem)_minmax(0,1fr)]">
        {/* Intake */}
        <section className="space-y-5">
          <div>
            <Label>Start a contract audit</Label>
            <h1 className="mt-3 text-2xl leading-tight">
              Bring the agreement into evidence.
            </h1>
            <p className="mt-2 text-sm text-muted">
              VidhiAI maps clauses to statutory text, then exposes the verified
              source behind every finding.
            </p>
          </div>

          <label className="block">
            <Label>Document title</Label>
            <input
              value={title}
              onChange={(event) => setTitle(event.target.value)}
              className="mt-1.5 w-full rounded border px-3 py-2 text-sm"
              style={{ background: "var(--surface)" }}
            />
          </label>

          <label className="block">
            <Label>Contract text</Label>
            <textarea
              value={text}
              onChange={(event) => setText(event.target.value)}
              rows={13}
              placeholder="Paste the clauses to review…"
              className="legal-text mt-1.5 w-full resize-y rounded border px-3.5 py-3"
              style={{ background: "var(--surface)" }}
            />
          </label>

          <div className="flex flex-wrap items-center gap-2">
            <Button onClick={runAudit} disabled={running || text.trim().length < 20}>
              {running ? "Auditing…" : "Run audit"}
            </Button>
            {running && (
              <Button variant="secondary" onClick={cancel}>
                Cancel
              </Button>
            )}
            <Button
              variant="secondary"
              onClick={() => fileInput.current?.click()}
              disabled={uploading || running}
            >
              {uploading ? "Reading…" : "Upload PDF"}
            </Button>
            <Button
              variant="ghost"
              onClick={() => {
                setText(SAMPLE);
                setTitle("Mutual NDA · Aranya");
              }}
              disabled={running}
            >
              Use sample
            </Button>
            <input
              ref={fileInput}
              type="file"
              accept="application/pdf"
              className="sr-only"
              onChange={(event) => {
                const file = event.target.files?.[0];
                if (file) void onFile(file);
                event.target.value = "";
              }}
            />
          </div>

          {uploadError && <ErrorState message={uploadError} />}

          {!started && (
            <div className="border-t pt-5">
              <Label>How this audit holds up</Label>
              <ol className="mt-3 space-y-2.5 text-sm">
                {[
                  "Identify obligations and sensitive-data clauses.",
                  "Run a visible Retrieve → Critic → Analyze → Verify trace.",
                  "Open each statutory source before relying on it.",
                ].map((step, index) => (
                  <li key={step} className="flex gap-3">
                    <span
                      className="font-mono text-2xs"
                      style={{ color: "var(--accent)" }}
                    >
                      0{index + 1}
                    </span>
                    <span className="text-muted">{step}</span>
                  </li>
                ))}
              </ol>
            </div>
          )}
        </section>

        {/* Trace and findings */}
        <section className="space-y-5">
          <AgentTrace events={events} running={running} />

          {error && (
            <ErrorState
              title="We could not complete the audit"
              message={error}
              onRetry={runAudit}
            />
          )}

          {result && (
            <div
              className="flex flex-wrap items-center gap-x-6 gap-y-2 rounded-[var(--radius-card)] border px-5 py-4"
              style={{ background: "var(--surface-sunken)" }}
            >
              <span className="font-serif text-lg">
                {findings.length === 0
                  ? "No material findings"
                  : `${findings.length} material finding${findings.length === 1 ? "" : "s"}`}
              </span>

              <span className="flex gap-3 font-mono text-2xs tracking-wider uppercase">
                {(["high", "medium", "low"] as const).map(
                  (risk) =>
                    result.risk_summary[risk] > 0 && (
                      <span key={risk} style={{ color: `var(--color-risk-${risk})` }}>
                        {result.risk_summary[risk]} {risk}
                      </span>
                    ),
                )}
              </span>

              {/* Never hidden: a discarded finding is one the verifier could
                  not ground, and concealing it would undo the guarantee. */}
              {result.discarded_findings > 0 && (
                <span
                  className="font-mono text-2xs text-muted"
                  title="Citations that could not be matched to the retrieved text"
                >
                  {result.discarded_findings} discarded as unverifiable
                </span>
              )}

              <span className="ml-auto font-mono text-2xs tabular-nums text-muted">
                {(result.elapsed_ms / 1000).toFixed(1)}s ·{" "}
                {result.clauses_reviewed} clauses
              </span>
            </div>
          )}

          {result && findings.length === 0 && (
            <div className="rounded-[var(--radius-card)] border">
              <EmptyState
                title="No conflicts found"
                description="No clause conflicted with the statutes retrieved for it. That reflects the Acts currently ingested, not a clean bill of health."
              />
            </div>
          )}

          <div className="space-y-4">
            {findings.map((finding, index) => (
              <FindingCard
                key={`${finding.chunk_id}-${index}`}
                finding={finding}
                onOpenSource={() =>
                  setOpenSource({ id: finding.chunk_id, quote: finding.quote })
                }
              />
            ))}
          </div>

          {!started && (
            <div className="rounded-[var(--radius-card)] border">
              <EmptyState
                title="No audit yet"
                description="Paste a contract and run an audit. The reasoning trace appears here as the agent works, including when it corrects itself."
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
