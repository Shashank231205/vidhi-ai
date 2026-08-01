/**
 * Backend client.
 *
 * Requests go to this origin under `/api`, which next.config.ts proxies to the
 * API process. The backend is a separate service — Vercel cannot host it, with
 * ~3GB of models against a 250MB function limit — but the browser never needs
 * to know that, which keeps development to a single port and avoids CORS
 * entirely.
 */

export const API_BASE = "/api";

/** Risk levels, matching the backend's RiskLevel enum. */
export type Risk = "high" | "medium" | "low";

/** How a precedent bears on the user's position. */
export type Stance = "supports" | "undermines" | "neutral";

export type NodeStatus =
  | "started"
  | "completed"
  | "retrying"
  | "failed"
  | "skipped";

/**
 * One observable step in an agent run.
 *
 * `attempt > 1` is the visible evidence of self-correction: the critic sent
 * retrieval back, or the verifier rejected a citation.
 */
export interface TraceEvent {
  run_id: string;
  node: string;
  status: NodeStatus;
  detail: string;
  attempt: number;
  elapsed_ms: number;
  data: Record<string, unknown>;
}

export interface Finding {
  clause_label: string | null;
  clause_text: string;
  issue: string;
  explanation: string;
  risk: Risk;
  /** Which path set the risk level — "llm" or "classifier". */
  risk_source: string;
  risk_confidence: number | null;
  citation: string;
  quote: string;
  suggested_fix: string;
  chunk_id: string;
}

export interface AuditResult {
  run_id: string;
  document_title: string;
  clauses_reviewed: number;
  findings: Finding[];
  /** Findings the verifier could not ground. Shown, never hidden. */
  discarded_findings: number;
  risk_summary: Record<string, number>;
  elapsed_ms: number;
}

export interface AssessedCase {
  document_id: string;
  chunk_id: string;
  citation: string;
  case_title: string;
  stance: Stance;
  confidence: number;
  reasoning: string;
  holding: string;
  quote: string;
  cited_by_count: number;
  via_citation_graph: boolean;
}

export interface ResearchMemo {
  summary: string;
  supporting_argument: string;
  risks: string;
  gaps: string | null;
}

export interface ResearchResult {
  run_id: string;
  cases: AssessedCase[];
  memo: ResearchMemo | null;
  stance_summary: Record<string, number>;
  discarded: number;
  elapsed_ms: number;
}

export interface SearchHit {
  chunk_id: string;
  document_id: string;
  document_title: string;
  label: string | null;
  citation: string;
  content: string;
  score: number;
  matched_by: string[];
}

/** The exact source text behind a citation, with its provenance. */
export interface ChunkDetail {
  chunk_id: string;
  document_id: string;
  document_title: string;
  source_ref: string;
  source_url: string | null;
  label: string | null;
  content: string;
  ordinal: number;
  content_hash: string;
  meta: Record<string, unknown>;
}

export interface DocumentSummary {
  id: string;
  kind: string;
  title: string;
  source_ref: string;
  source_url: string | null;
  chunk_count: number;
  meta: Record<string, unknown>;
}

export interface HealthReport {
  status: "ok" | "degraded";
  environment: string;
  version: string;
  components: Record<string, { configured: boolean; detail: string }>;
}

/** An error carrying the backend's user-facing message, where it gave one. */
export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: {
        ...(init?.body instanceof FormData
          ? {}
          : { "Content-Type": "application/json" }),
        ...init?.headers,
      },
    });
  } catch {
    // A network failure here almost always means the backend is asleep —
    // HuggingFace Spaces suspend after inactivity — so say that rather than
    // surfacing "Failed to fetch".
    throw new ApiError(
      "Could not reach the API. If it is deployed on a free tier it may be waking up; retry in a moment.",
      0,
    );
  }

  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      if (typeof body?.detail === "string") detail = body.detail;
    } catch {
      // Body was not JSON; the status-derived message stands.
    }
    throw new ApiError(detail, response.status);
  }

  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export const api = {
  health: () => request<HealthReport>("/health"),

  ready: () => request<{ ready: boolean; database: boolean }>("/ready"),

  /** Resolve a citation to the chunk its quote was verified against. */
  chunk: (chunkId: string) =>
    request<ChunkDetail>(`/documents/chunks/${encodeURIComponent(chunkId)}`),

  documents: (kind?: string) =>
    request<DocumentSummary[]>(
      `/documents${kind ? `?kind=${encodeURIComponent(kind)}` : ""}`,
    ),

  search: (query: string, limit = 8) =>
    request<{ query: string; hits: SearchHit[]; took_ms: number }>(
      `/documents/search?q=${encodeURIComponent(query)}&limit=${limit}`,
    ),

  upload: (file: File, kind = "contract") => {
    const form = new FormData();
    form.append("file", file);
    return request<{
      document_id: string;
      title: string;
      pages: number;
      characters: number;
      chunks: number;
      embedded: number;
      already_present: boolean;
    }>(`/documents/upload?kind=${kind}`, { method: "POST", body: form });
  },

  audit: (text: string, title: string, maxClauses?: number) =>
    request<AuditResult>("/compliance/audit", {
      method: "POST",
      body: JSON.stringify({ text, title, max_clauses: maxClauses }),
    }),

  research: (facts: string, limit = 6, expand = true) =>
    request<ResearchResult>("/caselens/research", {
      method: "POST",
      body: JSON.stringify({ facts, limit, expand }),
    }),
};
