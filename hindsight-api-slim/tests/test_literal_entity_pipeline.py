"""确认每条素材的精确匹配要求在分块、合并和解析之间不丢失。"""

from unittest.mock import AsyncMock

import pytest

from hindsight_api.api.http import MemoryItem, UpdateMemoryRequest
from hindsight_api.engine.entity_resolver import EntityResolver
from hindsight_api.engine.retain.entity_processing import _prepare_facts_for_entity_processing
from hindsight_api.engine.retain.link_utils import _prepare_entities_for_resolution
from hindsight_api.engine.retain.orchestrator import _build_contents, _build_delta_contents
from hindsight_api.engine.retain.types import EntityRef, ProcessedFact, UserEntities


def test_supplied_names_override_extraction_but_other_names_still_resolve():
    fact = ProcessedFact(
        fact_text="P003 和 P009 见到了 Alice。",
        fact_type="world",
        embedding=[],
        occurred_start=None,
        occurred_end=None,
        mentioned_at=None,
        context="",
        metadata={},
        entities=[EntityRef(name="P003"), EntityRef(name="Alice")],
    )
    supplied = UserEntities([{"text": "P003"}, {"text": "P009"}], resolve=False)
    texts, dates, entities = _prepare_facts_for_entity_processing([fact], {0: supplied})
    flat, _, _ = _prepare_entities_for_resolution(["unit"], texts, dates, entities)
    assert {item["text"]: item["resolve"] for item in flat} == {
        "P003": False,
        "P009": False,
        "Alice": True,
    }


def test_content_and_delta_keep_each_items_resolution_choice():
    contents = _build_contents(
        [
            {"content": "P003", "entities": [{"text": "P003"}], "resolve_entities": False},
            {"content": "Alice", "entities": [{"text": "Alice"}]},
        ],
        None,
    )
    assert [item.resolve_entities for item in contents] == [False, True]
    delta, _ = _build_delta_contents(contents, {0: "P003 的新照片"}, [0])
    assert delta[0].resolve_entities is False


@pytest.mark.parametrize("model", [MemoryItem, UpdateMemoryRequest])
def test_http_flag_is_optional_and_preserves_false(model):
    fields = {"content": "照片"} if model is MemoryItem else {"entities": ["P003"]}
    assert model(**fields).resolve_entities is True
    assert model(**fields, resolve_entities=False).resolve_entities is False


@pytest.mark.asyncio
async def test_literal_batch_skips_fuzzy_database_queries():
    resolver = EntityResolver(pool=None, entity_lookup="trigram")
    resolver._resolve_from_candidates = AsyncMock(return_value=[])
    conn = AsyncMock()
    await resolver.resolve_entities_batch("bank", [{"text": "P003", "resolve": False}], "", None, conn=conn)
    conn.fetchval.assert_not_called()
    assert resolver._resolve_from_candidates.call_args.kwargs["all_candidates"] == {}
