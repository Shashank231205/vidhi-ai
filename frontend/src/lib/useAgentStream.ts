"use client";

import { useCallback, useRef, useState } from "react";
import { API_BASE, type TraceEvent } from "./api";

/**
 * Consumes an agent's SSE trace.
 *
 * `EventSource` cannot be used: it only issues GET requests, and both agent
 * endpoints POST a body. So this reads the response stream directly and parses
 * the SSE framing, which also gives us abort control the client can trigger on
 * unmount or cancel.
 */

export interface StreamState<T> {
  events: TraceEvent[];
  result: T | null;
  running: boolean;
  error: string | null;
}

interface Options<T> {
  onResult?: (result: T) => void;
}

/** Split an SSE frame into its `event:` and `data:` parts. */
function parseFrame(frame: string): { event: string; data: string } | null {
  let event = "message";
  const data: string[] = [];

  for (const line of frame.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data.push(line.slice(5).trim());
  }

  return data.length ? { event, data: data.join("\n") } : null;
}

export function useAgentStream<T>(path: string, options: Options<T> = {}) {
  const [state, setState] = useState<StreamState<T>>({
    events: [],
    result: null,
    running: false,
    error: null,
  });

  const controller = useRef<AbortController | null>(null);
  // Held in a ref so `start` never has to be re-created when the callback
  // identity changes, which would restart the stream.
  const onResult = useRef(options.onResult);
  onResult.current = options.onResult;

  const cancel = useCallback(() => {
    controller.current?.abort();
    controller.current = null;
    setState((previous) => ({ ...previous, running: false }));
  }, []);

  const start = useCallback(
    async (body: unknown) => {
      // A second run must not interleave with the first.
      controller.current?.abort();
      const abort = new AbortController();
      controller.current = abort;

      setState({ events: [], result: null, running: true, error: null });

      try {
        const response = await fetch(`${API_BASE}${path}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
          signal: abort.signal,
        });

        if (!response.ok || !response.body) {
          let detail = `Request failed (${response.status})`;
          try {
            const parsed = await response.json();
            if (typeof parsed?.detail === "string") detail = parsed.detail;
          } catch {
            // Non-JSON body; keep the status message.
          }
          throw new Error(detail);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });

          // Frames are separated by a blank line. The last segment may be a
          // partial frame, so it stays in the buffer until the next read.
          const frames = buffer.split("\n\n");
          buffer = frames.pop() ?? "";

          for (const raw of frames) {
            const frame = parseFrame(raw);
            if (!frame) continue;

            if (frame.event === "trace") {
              const traceEvent = JSON.parse(frame.data) as TraceEvent;
              setState((previous) => ({
                ...previous,
                events: [...previous.events, traceEvent],
              }));
            } else if (frame.event === "result") {
              const result = JSON.parse(frame.data) as T;
              setState((previous) => ({ ...previous, result }));
              onResult.current?.(result);
            } else if (frame.event === "done") {
              const finished = JSON.parse(frame.data) as {
                status: string;
                error?: string | null;
              };
              setState((previous) => ({
                ...previous,
                running: false,
                error: finished.error ?? previous.error,
              }));
              return;
            }
          }
        }

        setState((previous) => ({ ...previous, running: false }));
      } catch (error) {
        // An abort is a deliberate cancel, not a failure to report.
        if (error instanceof DOMException && error.name === "AbortError") return;

        setState((previous) => ({
          ...previous,
          running: false,
          error:
            error instanceof Error
              ? error.message
              : "The request failed. Please retry.",
        }));
      }
    },
    [path],
  );

  return { ...state, start, cancel };
}
