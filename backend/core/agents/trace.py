"""Typed agent trace events.

Every node transition in every agent graph emits one of these. They are streamed
to the client over SSE and rendered as a live reasoning trace, so the schema is a
public contract with the frontend — additive changes only.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import BaseModel, Field


class NodeStatus(StrEnum):
    STARTED = "started"
    COMPLETED = "completed"
    #: A bounded self-correction fired — the graph is looping back deliberately.
    RETRYING = "retrying"
    #: Terminal failure for this node after retries were exhausted.
    FAILED = "failed"
    SKIPPED = "skipped"


class TraceEvent(BaseModel):
    """A single observable step in an agent run."""

    run_id: str
    node: str = Field(description="Graph node name, e.g. 'retrieve', 'critic'.")
    status: NodeStatus
    #: Human-readable, shown verbatim in the UI: "retrieving DPDP §8".
    detail: str
    #: 1-based; >1 means a critic or verifier loop sent us back here.
    attempt: int = 1
    elapsed_ms: int = 0
    #: Node-specific payload (chunk ids, scores, counts). Kept loose on purpose;
    #: the UI reads known keys and ignores the rest.
    data: dict[str, Any] = Field(default_factory=dict)

    def sse(self) -> dict[str, str]:
        """Render for sse-starlette's EventSourceResponse."""
        return {"event": "trace", "data": self.model_dump_json()}


class RunFinished(BaseModel):
    """Terminal event. Exactly one per run, always sent — including on failure."""

    run_id: str
    status: Literal["ok", "error"]
    total_ms: int
    #: Populated when status == "error"; safe to surface to the user.
    error: str | None = None

    def sse(self) -> dict[str, str]:
        return {"event": "done", "data": self.model_dump_json()}


class TraceEmitter:
    """Collects trace events for one agent run.

    Buffers into an unbounded queue so graph nodes never block on a slow client.
    """

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self._started_at = time.monotonic()
        self._events: list[TraceEvent] = []
        self._subscribers: list[Any] = []

    def _elapsed_ms(self) -> int:
        return int((time.monotonic() - self._started_at) * 1000)

    def emit(
        self,
        node: str,
        status: NodeStatus,
        detail: str,
        *,
        attempt: int = 1,
        **data: Any,
    ) -> TraceEvent:
        event = TraceEvent(
            run_id=self.run_id,
            node=node,
            status=status,
            detail=detail,
            attempt=attempt,
            elapsed_ms=self._elapsed_ms(),
            data=data,
        )
        self._events.append(event)
        for queue in self._subscribers:
            queue.put_nowait(event)
        return event

    def subscribe(self, queue: Any) -> None:
        self._subscribers.append(queue)

    @property
    def events(self) -> list[TraceEvent]:
        """Full ordered history — persisted to the audit log after a run."""
        return list(self._events)

    def finished(self, error: str | None = None) -> RunFinished:
        return RunFinished(
            run_id=self.run_id,
            status="error" if error else "ok",
            total_ms=self._elapsed_ms(),
            error=error,
        )


class node_span:  # noqa: N801 — used as a context manager, reads better lowercase
    """Emits started/completed (or failed) around a graph node.

    ```python
    with node_span(emitter, "retrieve", "searching DPDP Act", attempt=2) as span:
        hits = await retriever.search(q)
        span.data["hits"] = len(hits)
    ```
    """

    def __init__(
        self,
        emitter: TraceEmitter,
        node: str,
        detail: str,
        *,
        attempt: int = 1,
    ) -> None:
        self.emitter = emitter
        self.node = node
        self.detail = detail
        self.attempt = attempt
        self.data: dict[str, Any] = {}

    def __enter__(self) -> Self:
        self.emitter.emit(self.node, NodeStatus.STARTED, self.detail, attempt=self.attempt)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> Literal[False]:
        if exc_type is not None:
            self.emitter.emit(
                self.node,
                NodeStatus.FAILED,
                str(exc) or exc_type.__name__,
                attempt=self.attempt,
                **self.data,
            )
        else:
            self.emitter.emit(
                self.node,
                NodeStatus.COMPLETED,
                self.detail,
                attempt=self.attempt,
                **self.data,
            )
        return False


async def stream_trace(
    emitter: TraceEmitter,
    queue: Any,
) -> AsyncIterator[dict[str, str]]:
    """Yield SSE frames until the run's terminal event arrives."""
    while True:
        item = await queue.get()
        if isinstance(item, RunFinished):
            yield item.sse()
            return
        yield item.sse()
