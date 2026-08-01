"use client";

import { useCallback, useRef, useState } from "react";
import { AgentTrace } from "@/components/AgentTrace";
import {
  Button,
  Citation,
  EmptyState,
  ErrorState,
  Quote,
  RiskBadge,
} from "@/components/ui";
import { ApiError, api, type AuditResult, type Finding, type Risk } from "@/lib/api";
import { useAgentStream } from "@/lib/useAgentStream";

const SAMPLE = `3. Data Collection. The Company may collect, store and process any and all personal data of the User, including sensitive personal data, for any purpose it deems fit, without providing notice to the User and without obtaining separate consent.

4. Data Retention. The Company shall retain all User personal data indefinitely, including after the User terminates this Agreement or withdraws consent.

5. Liability. The User agrees to indemnify the Company against all claims without limitation, and this obligation shall survive termination.`;

const RISK_ORDER: Record<Risk, number> = { high: 0, medium: 1, low: 2 };

function FindingCard({ finding }: { finding: Finding }) {
  return (
    <article className="surface animate-in p-5">
      <header className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <h3 className="font-medium">{finding.issue}</h3>
          {finding.clause_label && (
            <p className="mt-0.5 text-xs text-muted">{finding.clause_label}</p>
          )}
        </div>
        <RiskBadge risk={finding.risk} source={finding.risk_source} />
      </header>

      <p className="mt-3 text-sm">{finding.explanation}</p>

      <div className="mt-4 space-y-1.5">
        <Citation>{finding.citation}</Citation>
        <Quote>“{finding.quote}”</Quote>
      </div>

      <div className="mt-4 rounded-lg p-3" style={{ background: "var(--surface-sunken)" }}>
        <p className="text-2xs font-semibold tracking-wide uppercase text-muted">
          Suggested fix
        </p>
        <p className="mt-1 text-sm">{finding.suggested_fix}</p>
      </div>
    </article>
  );
}

function Summary({ result }: { result: AuditResult }) {
  const counts = result.risk_summary;
  return (
    <div className="surface flex flex-wrap items-center gap-x-6 gap-y-2 px-5 py-4 text-sm">
      <span>
        <strong>{result.findings.length}</strong> finding
        {result.findings.length === 1 ? "" : "s"} across{" "}
        <strong>{result.clauses_reviewed}</strong> clause
        {result.clauses_reviewed === 1 ? "" : "s"}
      </span>
      <span className="flex gap-3 text-xs">
        {(["high", "medium", "low"] as const).map(
          (risk) =>
            counts[risk] > 0 && (
              <span key={risk} style={{ color: `var(--color-risk-${risk})` }}>
                {counts[risk]} {risk}
              </span>
            ),
        )}
      </span>
      {/* Never hidden: a discarded finding is one the verifier could not
          ground, and concealing that would undermine the guarantee. */}
      {result.discarded_findings > 0 && (
        <span className="text-xs text-muted" title="Citations that could not be verified against the retrieved text">
          {result.discarded_findings} discarded as unverifiable
        </span>
      )}
      <span className="ml-auto font-mono text-2xs text-muted tabular-nums">
        {(result.elapsed_ms / 1000).toFixed(1)}s
      </span>
    </div>
  );
}

export default function CompliancePage() {
  const [text, setText] = useState("");
  const [title, setTitle] = useState("Pasted contract");
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  const { events, result, running, error, start, cancel } =
    useAgentStream<AuditResult>("/compliance/audit/stream");

  const runAudit = useCallback(() => {
    if (text.trim().length < 20) return;
    setUploadError(null);
    // Capped for the demo: a long contract is hundreds of LLM calls and will
    // meet the provider's rate limit long before it finishes.
    void start({ text, title, max_clauses: 8 });
  }, [start, text, title]);

  const onFile = async (file: File) => {
    setUploading(true);
    setUploadError(null);
    try {
      // Uploading extracts and indexes the PDF; the audit then runs over the
      // extracted text like any pasted contract.
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

  const sorted = result
    ? [...result.findings].sort(
        (a, b) => RISK_ORDER[a.risk] - RISK_ORDER[b.risk],
      )
    : [];

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,26rem)_minmax(0,1fr)]">
      <section className="space-y-4">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">ComplianceGuard</h1>
          <p className="mt-1 text-sm text-muted">
            Paste a contract or upload a PDF. Each clause is audited against the
            ingested Indian statutes.
          </p>
        </div>

        <label className="block">
          <span className="text-xs font-medium">Document title</span>
          <input
            value={title}
            onChange={(event) => setTitle(event.target.value)}
            className="mt-1.5 w-full rounded-lg border bg-[var(--surface)] px-3 py-2 text-sm"
          />
        </label>

        <label className="block">
          <span className="text-xs font-medium">Contract text</span>
          <textarea
            value={text}
            onChange={(event) => setText(event.target.value)}
            rows={14}
            placeholder="Paste the clauses to review…"
            className="legal-text mt-1.5 w-full resize-y rounded-lg border bg-[var(--surface)] px-3 py-2"
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
              setTitle("Sample vendor agreement");
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
      </section>

      <section className="space-y-4">
        <AgentTrace events={events} running={running} />

        {error && <ErrorState message={error} onRetry={runAudit} />}

        {result && <Summary result={result} />}

        {result && sorted.length === 0 && (
          <div className="surface">
            <EmptyState
              title="No compliance issues found"
              description="No clause conflicted with the statutes retrieved for it. That is not a clean bill of health — it reflects the Acts currently ingested."
            />
          </div>
        )}

        <div className="space-y-3">
          {sorted.map((finding, index) => (
            <FindingCard key={`${finding.chunk_id}-${index}`} finding={finding} />
          ))}
        </div>

        {!result && !running && !error && (
          <div className="surface">
            <EmptyState
              title="No audit yet"
              description="Paste a contract and run an audit. The reasoning trace appears here as the agent works, including when it corrects itself."
            />
          </div>
        )}
      </section>
    </div>
  );
}
