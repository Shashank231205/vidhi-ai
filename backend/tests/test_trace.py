"""Trace events are a public contract with the frontend — test them as such."""

import asyncio

import pytest

from core.agents.trace import (
    NodeStatus,
    RunFinished,
    TraceEmitter,
    TraceEvent,
    node_span,
    stream_trace,
)


def test_span_emits_started_then_completed() -> None:
    emitter = TraceEmitter("run-1")

    with node_span(emitter, "retrieve", "searching DPDP Act") as span:
        span.data["hits"] = 5

    assert [e.status for e in emitter.events] == [
        NodeStatus.STARTED,
        NodeStatus.COMPLETED,
    ]
    assert emitter.events[-1].data["hits"] == 5


def test_span_marks_failed_and_propagates() -> None:
    emitter = TraceEmitter("run-2")

    with (
        pytest.raises(ValueError, match="upstream down"),
        node_span(emitter, "retrieve", "searching"),
    ):
        raise ValueError("upstream down")

    failed = emitter.events[-1]
    assert failed.status is NodeStatus.FAILED
    assert failed.detail == "upstream down"


def test_attempt_number_records_self_correction() -> None:
    """The critic loop is only observable if attempt is carried through."""
    emitter = TraceEmitter("run-3")

    for attempt in (1, 2, 3):
        with node_span(emitter, "retrieve", "searching", attempt=attempt):
            pass

    completed = [e for e in emitter.events if e.status is NodeStatus.COMPLETED]
    assert [e.attempt for e in completed] == [1, 2, 3]


def test_elapsed_is_monotonic() -> None:
    emitter = TraceEmitter("run-4")
    for _ in range(3):
        emitter.emit("n", NodeStatus.COMPLETED, "d")

    elapsed = [e.elapsed_ms for e in emitter.events]
    assert elapsed == sorted(elapsed)


def test_finished_reports_error() -> None:
    emitter = TraceEmitter("run-5")
    ok = emitter.finished()
    assert ok.status == "ok" and ok.error is None

    bad = emitter.finished("no providers configured")
    assert bad.status == "error"
    assert bad.error == "no providers configured"


def test_sse_frames_are_named() -> None:
    emitter = TraceEmitter("run-6")
    event = emitter.emit("retrieve", NodeStatus.STARTED, "searching")

    assert event.sse()["event"] == "trace"
    assert "retrieve" in event.sse()["data"]
    assert emitter.finished().sse()["event"] == "done"


async def test_subscriber_receives_events_live() -> None:
    emitter = TraceEmitter("run-7")
    queue: asyncio.Queue[TraceEvent | RunFinished] = asyncio.Queue()
    emitter.subscribe(queue)

    emitter.emit("retrieve", NodeStatus.STARTED, "searching")
    received = queue.get_nowait()
    assert isinstance(received, TraceEvent)
    assert received.node == "retrieve"


async def test_stream_terminates_on_done() -> None:
    emitter = TraceEmitter("run-8")
    queue: asyncio.Queue[TraceEvent | RunFinished] = asyncio.Queue()
    emitter.subscribe(queue)

    emitter.emit("retrieve", NodeStatus.COMPLETED, "found 3")
    await queue.put(emitter.finished())

    frames = [frame async for frame in stream_trace(emitter, queue)]
    assert [f["event"] for f in frames] == ["trace", "done"]


async def test_emit_never_blocks_on_slow_consumer() -> None:
    """Graph nodes must not stall behind a client that isn't reading."""
    emitter = TraceEmitter("run-9")
    queue: asyncio.Queue[TraceEvent | RunFinished] = asyncio.Queue()
    emitter.subscribe(queue)

    for _ in range(500):
        emitter.emit("n", NodeStatus.COMPLETED, "d")

    assert queue.qsize() == 500


def test_run_finished_roundtrips() -> None:
    finished = RunFinished(run_id="r", status="ok", total_ms=120)
    assert RunFinished.model_validate_json(finished.model_dump_json()) == finished
