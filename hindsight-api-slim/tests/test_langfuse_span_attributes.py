"""Spans carry what Langfuse needs to render IO, price a generation and count tokens.

Langfuse stores ``langfuse.observation.usage_details`` as-is, so the buckets must
already be exclusive; it prices by ``langfuse.observation.model.name``; and an
operation span without ``langfuse.observation.input``/``output`` shows ``undefined``.
"""

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest

from hindsight_api.engine.providers.openai_compatible_llm import (
    _capture_routed_model,
    _served_model,
    _start_routed_model_capture,
)
from hindsight_api.tracing import LLMSpanRecorder, set_span_io, summarize_recall_for_span


def _record(**kwargs) -> tuple[dict[str, object], MagicMock]:
    span = MagicMock()
    tracer = MagicMock()
    tracer.start_as_current_span.return_value.__enter__.return_value = span
    call = {
        "provider": "openai",
        "model": "dynamic/hindsight",
        "scope": "reflect_tool_call",
        "messages": [{"role": "user", "content": "q"}],
        "response_content": "a",
        "input_tokens": 100,
        "output_tokens": 20,
        "duration": 0.1,
    }
    call.update(kwargs)
    LLMSpanRecorder(tracer).record_llm_call(**call)
    return {c.args[0]: c.args[1] for c in span.set_attribute.call_args_list}, span


def test_routed_model_names_the_generation():
    attrs, _ = _record(response_model="deepseek-v4-flash")

    assert attrs["gen_ai.request.model"] == "dynamic/hindsight"
    assert attrs["gen_ai.response.model"] == "deepseek-v4-flash"
    assert attrs["langfuse.observation.model.name"] == "deepseek-v4-flash"


def test_configured_model_is_the_fallback_name():
    attrs, _ = _record()

    assert attrs["langfuse.observation.model.name"] == "dynamic/hindsight"


def test_usage_buckets_do_not_overlap():
    attrs, _ = _record(input_tokens=43036, output_tokens=613, cached_tokens=43008, thoughts_tokens=4821)

    usage = json.loads(attrs["langfuse.observation.usage_details"])
    assert usage == {
        "input": 28,
        "input_cached_tokens": 43008,
        "output": 613,
        "output_reasoning_tokens": 4821,
    }
    assert sum(usage.values()) == 43036 + 613 + 4821


def test_usage_without_cache_or_reasoning_has_only_base_buckets():
    attrs, _ = _record()

    assert json.loads(attrs["langfuse.observation.usage_details"]) == {"input": 100, "output": 20}


def test_tool_calls_reach_the_output_message():
    tool_calls = [{"id": "t1", "name": "recall", "arguments": {"query": "trip"}}]
    _, span = _record(response_content=None, tool_calls=tool_calls)

    event_attrs = span.add_event.call_args_list[0].kwargs["attributes"]
    output = json.loads(event_attrs["gen_ai.output.messages"])
    assert output == [{"role": "assistant", "content": None, "tool_calls": tool_calls}]


def test_span_io_serializes_dicts_with_datetimes():
    span = MagicMock()
    when = datetime(2026, 10, 30, tzinfo=UTC)

    set_span_io(span, input={"query": "旅行", "at": when}, output="done")

    attrs = {c.args[0]: c.args[1] for c in span.set_attribute.call_args_list}
    assert json.loads(attrs["langfuse.observation.input"]) == {"query": "旅行", "at": str(when)}
    assert attrs["langfuse.observation.output"] == "done"


def test_span_io_ignores_missing_span():
    set_span_io(None, input={"q": 1})


def test_recall_summary_keeps_top_facts_truncated():
    facts = [SimpleNamespace(id=i, fact_type="world", text="x" * 500) for i in range(25)]

    summary = summarize_recall_for_span(SimpleNamespace(results=facts))

    assert summary["count"] == 25
    assert len(summary["facts"]) == 20
    assert summary["facts"][0] == {"id": "0", "fact_type": "world", "text": "x" * 300}


@pytest.mark.asyncio
async def test_gateway_header_wins_over_body_model():
    slot = _start_routed_model_capture()
    await _capture_routed_model(httpx.Response(200, headers={"cf-aig-model": "deepseek-v4-flash"}))

    assert _served_model(SimpleNamespace(model="dynamic/hindsight"), slot) == "deepseek-v4-flash"


@pytest.mark.asyncio
async def test_body_model_when_no_gateway_header():
    slot = _start_routed_model_capture()
    await _capture_routed_model(httpx.Response(200))

    assert _served_model(SimpleNamespace(model="gpt-4o-mini-2024-07-18"), slot) == "gpt-4o-mini-2024-07-18"
    assert _served_model(SimpleNamespace(), slot) is None
