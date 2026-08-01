"use client";

import { useEffect, useState } from "react";
import { Button, Label, Skeleton, VerifiedMark } from "@/components/ui";
import { ApiError, api, type ChunkDetail } from "@/lib/api";

/**
 * Opens the exact source text a citation was verified against.
 *
 * This is the difference between claiming groundedness and showing it. Every
 * finding carries the chunk id its quote was checked against; this fetches
 * that chunk so a reviewer can read the provision themselves rather than
 * trusting the summary of it.
 */
export function SourceDrawer({
  chunkId,
  quote,
  onClose,
}: {
  chunkId: string | null;
  quote?: string;
  onClose: () => void;
}) {
  if (!chunkId) return null;
  // Keyed on the id: opening a different source remounts and resets state,
  // rather than clearing it from an effect after the stale content has already
  // rendered for a frame.
  return <Drawer key={chunkId} chunkId={chunkId} quote={quote} onClose={onClose} />;
}

function Drawer({
  chunkId,
  quote,
  onClose,
}: {
  chunkId: string;
  quote?: string;
  onClose: () => void;
}) {
  const [chunk, setChunk] = useState<ChunkDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Guards against a slow response landing after this drawer has closed.
    let cancelled = false;

    api
      .chunk(chunkId)
      .then((result) => {
        if (!cancelled) setChunk(result);
      })
      .catch((caught: unknown) => {
        if (cancelled) return;
        setError(
          caught instanceof ApiError ? caught.message : "Could not load the source.",
        );
      });

    return () => {
      cancelled = true;
    };
  }, [chunkId]);

  // Escape closes, which is the expected affordance for an overlay.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div
      className="fixed inset-0 z-30 flex justify-end"
      role="dialog"
      aria-modal="true"
      aria-label="Verified source"
    >
      <button
        className="absolute inset-0 bg-black/25"
        onClick={onClose}
        aria-label="Close source"
        tabIndex={-1}
      />

      <aside
        className="animate-in relative flex h-full w-full max-w-xl flex-col overflow-y-auto border-l shadow-xl"
        style={{ background: "var(--surface)" }}
      >
        <header className="sticky top-0 flex items-center justify-between gap-4 border-b px-6 py-4"
          style={{ background: "var(--surface)" }}
        >
          <Label>Verified source</Label>
          <VerifiedMark label="hash matched" />
        </header>

        <div className="flex-1 px-6 py-5">
          {error && (
            <p className="text-sm" style={{ color: "var(--color-risk-high)" }}>
              {error}
            </p>
          )}

          {!chunk && !error && (
            <div className="space-y-3">
              <Skeleton className="h-6 w-2/3" />
              <Skeleton className="h-28 w-full" />
              <Skeleton className="h-16 w-full" />
            </div>
          )}

          {chunk && (
            <>
              <h2 className="font-serif text-2xl leading-tight">
                {chunk.document_title}
                {chunk.label && (
                  <span className="text-muted"> · {chunk.label}</span>
                )}
              </h2>

              <div
                className="mt-5 rounded-[var(--radius-card)] border p-5"
                style={{ background: "var(--surface-sunken)" }}
              >
                <p className="legal-text whitespace-pre-wrap">{chunk.content}</p>
              </div>

              {/* The verified quote is shown alongside the full provision, so
                  the reader can see exactly which sentence was relied on. */}
              {quote && (
                <div className="mt-5">
                  <Label>Quoted in the finding</Label>
                  <blockquote
                    className="legal-text mt-2 border-l-2 pl-4"
                    style={{ borderColor: "var(--accent)" }}
                  >
                    “{quote}”
                  </blockquote>
                </div>
              )}

              <dl className="mt-6 grid grid-cols-2 gap-x-6 gap-y-4 border-t pt-5">
                <div>
                  <Label>Source</Label>
                  <dd className="mt-1 font-mono text-xs">{chunk.source_ref}</dd>
                </div>
                <div>
                  <Label>Chunk</Label>
                  <dd className="mt-1 font-mono text-xs">
                    #{chunk.ordinal} · {chunk.content_hash}
                  </dd>
                </div>
                {typeof chunk.meta?.year === "number" && (
                  <div>
                    <Label>Year</Label>
                    <dd className="mt-1 font-mono text-xs">{chunk.meta.year}</dd>
                  </div>
                )}
                {chunk.source_url && (
                  <div>
                    <Label>Official text</Label>
                    <dd className="mt-1">
                      <a
                        href={chunk.source_url}
                        target="_blank"
                        rel="noreferrer noopener"
                        className="font-mono text-xs underline"
                        style={{ color: "var(--accent)" }}
                      >
                        Open PDF →
                      </a>
                    </dd>
                  </div>
                )}
              </dl>
            </>
          )}
        </div>

        <footer className="sticky bottom-0 border-t px-6 py-4"
          style={{ background: "var(--surface)" }}
        >
          <Button variant="secondary" onClick={onClose}>
            Return to finding
          </Button>
        </footer>
      </aside>
    </div>
  );
}
