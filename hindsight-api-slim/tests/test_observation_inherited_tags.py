"""``observation_inherited_tag_prefixes``: observations inherit chosen tags from their sources.

A bank that consolidates one observation scope per user (``observation_scopes`` set to
``[["user:1"]]``) still wants to filter observations by the person or place their facts
are about. The configured prefixes make an observation carry, on top of its scope tags,
exactly the union of its live sources' tags under those prefixes — re-derived when
consolidation writes it and when a source's tags change, so a removed tag leaves too.
"""

import uuid

import pytest

from hindsight_api import RequestContext
from hindsight_api.config import _get_raw_config
from hindsight_api.engine.memories import get_memories
from hindsight_api.engine.memories.base import StoredMemory, inherited_observation_tags
from hindsight_api.engine.memory_engine import MemoryEngine, fq_table


def test_inherited_tags_replace_only_the_prefixed_part():
    observation = StoredMemory(unit_id="o1", text="", fact_type="observation", source_memory_ids=["f1", "f2"])
    tags_by_id = {
        "o1": ["user:1", "place:paris", "status:active"],
        "f1": ["user:1", "place:rome", "day:2025-01-01"],
        "f2": ["user:1", "person:p2"],
    }
    changed = inherited_observation_tags([observation], tags_by_id, ["place:", "person:"])
    # place:paris is gone (no source carries it); day: is not configured; status:active is kept.
    assert set(changed["o1"]) == {"user:1", "status:active", "place:rome", "person:p2"}


def test_unchanged_observations_are_not_reported():
    observation = StoredMemory(unit_id="o1", text="", fact_type="observation", source_memory_ids=["f1"])
    tags_by_id = {"o1": ["user:1", "place:rome"], "f1": ["user:1", "place:rome"]}
    assert inherited_observation_tags([observation], tags_by_id, ["place:"]) == {}


@pytest.fixture(autouse=True)
def enable_observations():
    config = _get_raw_config()
    original = config.enable_observations
    config.enable_observations = True
    yield
    config.enable_observations = original


async def _observation_tags(memory: MemoryEngine, bank_id: str) -> list[set[str]]:
    pool = await memory._get_pool()
    async with pool.acquire() as conn:
        page = await get_memories().scan_memories(
            conn=conn, fq_table=fq_table, bank_id=bank_id, fact_types=["observation"], limit=1000
        )
    return [set(o.tags or []) for o in page.memories]


async def _retain(memory, bank_id, tags, request_context):
    await memory.retain_batch_async(
        bank_id=bank_id,
        contents=[
            {
                "content": "Alice and Bob watched the sunset at the harbour.",
                "document_id": "doc-1",
                "tags": tags,
                "observation_scopes": [["user:1"]],
            }
        ],
        request_context=request_context,
    )


@pytest.mark.asyncio
@pytest.mark.timeout(600)
async def test_observations_follow_their_sources_prefixed_tags(memory: MemoryEngine, request_context: RequestContext):
    bank_id = f"test_obs_inherit_{uuid.uuid4().hex[:8]}"
    try:
        await memory._ensure_bank_exists(bank_id, request_context)
        await memory._config_resolver.update_bank_config(bank_id, {"observation_inherited_tag_prefixes": ["place:"]})

        await _retain(memory, bank_id, ["user:1", "place:harbour", "day:2025-01-01"], request_context)
        await memory.run_consolidation(bank_id=bank_id, request_context=request_context)
        observations = await _observation_tags(memory, bank_id)
        assert observations, "Setup: consolidation should have produced observations"
        # The scope plus the inherited place; the day is not configured to follow.
        assert all(tags == {"user:1", "place:harbour"} for tags in observations), observations

        # A tags PATCH keeps the scope (so the observations stand) and moves the place.
        await memory.update_document(
            "doc-1", bank_id, tags=["user:1", "place:pier", "day:2025-01-01"], request_context=request_context
        )
        assert all(tags == {"user:1", "place:pier"} for tags in await _observation_tags(memory, bank_id))

        # A re-retain of the same body takes the relabel path; the place follows again.
        await _retain(memory, bank_id, ["user:1", "place:beach"], request_context)
        assert all(tags == {"user:1", "place:beach"} for tags in await _observation_tags(memory, bank_id))
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


@pytest.mark.asyncio
@pytest.mark.timeout(600)
async def test_without_prefixes_observations_keep_their_scope_only(
    memory: MemoryEngine, request_context: RequestContext
):
    bank_id = f"test_obs_no_inherit_{uuid.uuid4().hex[:8]}"
    try:
        await memory._ensure_bank_exists(bank_id, request_context)
        await _retain(memory, bank_id, ["user:1", "place:harbour"], request_context)
        await memory.run_consolidation(bank_id=bank_id, request_context=request_context)
        observations = await _observation_tags(memory, bank_id)
        assert observations and all(tags == {"user:1"} for tags in observations), observations
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
