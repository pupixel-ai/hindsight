"""``store: metadata`` label groups keep extracted values in fact metadata, not entities.

A per-fact attribute such as "the photo this fact came from" must not become an entity:
entities link facts in the graph and are appended to the embedded text, so every fact
from one source would be pulled together by an id. The values go to ``metadata[key]``
as a JSON array instead, and — like label tags (#4068) — a re-retain survivor that was
not re-extracted keeps them rather than losing them to the document's metadata.
"""

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from hindsight_api.engine.providers.mock_llm import MockLLM
from hindsight_api.engine.retain.entity_labels import LabelGroup, label_metadata_keys
from hindsight_api.engine.retain.fact_extraction import _move_metadata_labels


@pytest.fixture(scope="session")
def db_url():
    """Isolated pg0 instance so this module never touches the dev database."""
    return "pg0://hslabelmeta:55471"


_LABELS = [
    {
        "key": "source_id",
        "type": "multi-text",
        "store": "metadata",
        "description": "IDs of the sections that support this fact.",
    }
]


def test_metadata_groups_cannot_be_tags_or_maps():
    with pytest.raises(ValidationError):
        LabelGroup(key="k", type="multi-text", store="metadata", tag=True)
    with pytest.raises(ValidationError):
        LabelGroup(key="k", type="map", store="metadata")
    assert label_metadata_keys(_LABELS) == {"source_id"}
    assert label_metadata_keys([{"key": "category", "type": "text"}]) == set()


def test_metadata_labels_leave_entities_and_keep_other_metadata():
    fact = SimpleNamespace(
        entities=["Alice", "Source_ID:101", "source_id:102", "source_id:101"],
        metadata={"document_id": "doc-1"},
    )
    _move_metadata_labels([fact], SimpleNamespace(entity_labels=_LABELS))
    assert fact.entities == ["Alice"]
    assert fact.metadata == {"document_id": "doc-1", "source_id": json.dumps(["101", "102"])}


def test_facts_without_metadata_labels_are_untouched():
    fact = SimpleNamespace(entities=["Alice"], metadata={"document_id": "doc-1"})
    _move_metadata_labels([fact], SimpleNamespace(entity_labels=_LABELS))
    assert fact.entities == ["Alice"]
    assert fact.metadata == {"document_id": "doc-1"}


class _Source:
    """The source id the mock LLM attaches to every fact it extracts."""

    value = "101"


@pytest.fixture(autouse=True)
def labelled_extraction(monkeypatch):
    def _facts(messages):
        user_text = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
        return {
            "facts": [
                {
                    "what": f"Fact about {user_text[-40:]}",
                    "when": "N/A",
                    "where": "N/A",
                    "who": "N/A",
                    "why": "N/A",
                    "fact_kind": "conversation",
                    "fact_type": "world",
                    "entities": [f"source_id:{_Source.value}"],
                }
            ]
        }

    _Source.value = "101"
    monkeypatch.setattr(MockLLM, "_build_mock_facts", staticmethod(_facts))


async def _units(memory, bank_id, document_id, request_context):
    res = await memory.list_memory_units(bank_id, document_id=document_id, limit=200, request_context=request_context)
    return {u["id"]: u for u in res["items"]}


async def _retain(memory, bank_id, doc, content, metadata, request_context):
    await memory.retain_batch_async(
        bank_id=bank_id,
        contents=[{"content": content, "document_id": doc, "metadata": metadata}],
        request_context=request_context,
    )


# Two chunks under a 60-char chunk size, so a re-retain that rewrites only the second
# leaves the first one's units in place as survivors.
_A = "Alpha " * 20
_B = "Bravo " * 20
_C = "Charlie " * 20


@pytest.mark.asyncio
@pytest.mark.timeout(600)
async def test_survivors_keep_metadata_labels_and_fresh_units_take_new_ones(memory, request_context):
    bank_id = f"test_label_meta_{datetime.now(timezone.utc).timestamp()}"
    doc = "doc-meta"
    try:
        await memory._ensure_bank_exists(bank_id, request_context)
        await memory._config_resolver.update_bank_config(
            bank_id, {"entity_labels": _LABELS, "entities_allow_free_form": False, "retain_chunk_size": 60}
        )

        await _retain(memory, bank_id, doc, _A + "\n\n" + _B, {"src": "v1"}, request_context)
        before = await _units(memory, bank_id, doc, request_context)
        assert before
        for unit in before.values():
            assert unit["metadata"] == {"src": "v1", "source_id": '["101"]'}, unit["metadata"]
            assert not any(e.lower().startswith("source_id:") for e in unit.get("entities") or [])

        _Source.value = "202"
        await _retain(memory, bank_id, doc, _A + "\n\n" + _C, {"src": "v2"}, request_context)
        after = await _units(memory, bank_id, doc, request_context)

        survivors = {uid: u for uid, u in after.items() if uid in before}
        fresh = {uid: u for uid, u in after.items() if uid not in before}
        assert survivors and fresh, (survivors, fresh)
        # Document metadata is replaced everywhere; the label value stays per unit.
        for uid, unit in survivors.items():
            assert unit["metadata"] == {"src": "v2", "source_id": '["101"]'}, f"survivor {uid}: {unit['metadata']}"
        for uid, unit in fresh.items():
            assert unit["metadata"] == {"src": "v2", "source_id": '["202"]'}, f"fresh {uid}: {unit['metadata']}"
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)
