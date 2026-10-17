"""官方 #3576 的精确实体回归，适配 0.8.5 runtime。"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from hindsight_api.engine.entity_resolver import EntityResolver
from hindsight_api.engine.retain.types import ResolvedEntity


@pytest.mark.asyncio
async def test_unresolved_names_beat_a_winning_fuzzy_candidate():
    """A caller-authored name must not be re-resolved to a similar entity (#3479).

    "Dr Wall" here is the shape that broke curation: a typo entity that outscores the
    0.6 threshold on name similarity (0.41) plus co-occurrence with the other name in
    the same request (0.3), so resolution silently swaps it in for the corrected
    "Dr. Waller". With resolve=False on that mention the scoring is skipped entirely and
    the literal name is created/reused instead.
    """
    from types import SimpleNamespace

    now = datetime.now(timezone.utc)
    candidates = {"Dr. Waller": [("typo-entity-id", "Dr Wall", {}, now, 40)]}
    cooccurrence = {"typo-entity-id": {"careorg"}}
    entities_data = [
        {
            "text": "Dr. Waller",
            "nearby_entities": [{"text": "Dr. Waller"}, {"text": "CareOrg"}],
            "event_date": now,
        }
    ]

    # Fuzzy matching (retain's behaviour) picks the typo entity — the bug, pinned here so
    # the assertion below is measuring a real difference.
    fuzzy_resolver = EntityResolver(
        pool=SimpleNamespace(
            ops=SimpleNamespace(bulk_insert_entities=AsyncMock(), fetch_missing_entity_ids=AsyncMock())
        ),
        entity_lookup="full",
    )
    fuzzy = await fuzzy_resolver._resolve_from_candidates(
        conn=AsyncMock(),
        bank_id="bank-1",
        entities_data=entities_data,
        unit_event_date=now,
        all_candidates=candidates,
        cooccurrence_map=cooccurrence,
    )
    assert fuzzy[0].canonical_name == "Dr Wall", "precondition: fuzzy scoring picks the typo entity"

    ops = SimpleNamespace(
        bulk_insert_entities=AsyncMock(return_value={"dr. waller": "correct-entity-id"}),
        fetch_missing_entity_ids=AsyncMock(return_value=[]),
    )
    exact_resolver = EntityResolver(pool=SimpleNamespace(ops=ops), entity_lookup="full")
    exact = await exact_resolver._resolve_from_candidates(
        conn=AsyncMock(),
        bank_id="bank-1",
        entities_data=[{**entities_data[0], "resolve": False}],
        unit_event_date=now,
        all_candidates=candidates,
        cooccurrence_map=cooccurrence,
    )
    assert exact == [ResolvedEntity(entity_id="correct-entity-id", canonical_name="Dr. Waller")]


@pytest.mark.asyncio
async def test_per_mention_resolve_flag_only_spares_the_names_that_opt_out():
    """Retain mixes caller-supplied and extracted names in one batch (#3479).

    The flag is per mention precisely so a caller can have their own names taken literally
    without turning off resolution for the extractor's — which would silently fill the bank
    with near-duplicate entities. Here the caller's "Dr. Waller" must be created as written
    while the extractor's "Dr Waler" still resolves onto the existing "Dr Wall".
    """
    from types import SimpleNamespace

    now = datetime.now(timezone.utc)
    typo = ("typo-entity-id", "Dr Wall", {}, now, 40)
    ops = SimpleNamespace(
        bulk_insert_entities=AsyncMock(return_value={"dr. waller": "literal-id"}),
        fetch_missing_entity_ids=AsyncMock(return_value=[]),
    )
    resolver = EntityResolver(pool=SimpleNamespace(ops=ops), entity_lookup="full")

    resolved = await resolver._resolve_from_candidates(
        conn=AsyncMock(),
        bank_id="bank-1",
        entities_data=[
            {"text": "Dr. Waller", "nearby_entities": [], "event_date": now, "resolve": False},
            {"text": "Dr Waler", "nearby_entities": [], "event_date": now},
        ],
        unit_event_date=now,
        all_candidates={"Dr. Waller": [typo], "Dr Waler": [typo]},
        cooccurrence_map={},
    )

    assert resolved[0] == ResolvedEntity(entity_id="literal-id", canonical_name="Dr. Waller"), (
        "the opted-out name is created as written"
    )
    assert resolved[1].canonical_name == "Dr Wall", "the extracted name still resolves onto the existing entity"


@pytest.mark.asyncio
async def test_intrabatch_clustering_skips_names_that_opted_out():
    """A literal name must not be folded into a same-batch variant, or pull one into itself."""
    from types import SimpleNamespace

    ops = SimpleNamespace(
        bulk_insert_entities=AsyncMock(return_value={"alice": "alice-id", "alice smith": "alice-smith-id"}),
        fetch_missing_entity_ids=AsyncMock(return_value=[]),
    )
    resolver = EntityResolver(pool=SimpleNamespace(ops=ops), entity_lookup="full")

    resolved = await resolver._resolve_from_candidates(
        conn=AsyncMock(),
        bank_id="bank-1",
        entities_data=[
            {"text": "Alice", "nearby_entities": [], "resolve": False},
            {"text": "Alice Smith", "nearby_entities": [], "resolve": False},
        ],
        unit_event_date=None,
        all_candidates={},
        cooccurrence_map={},
    )

    assert [e.canonical_name for e in resolved] == ["Alice", "Alice Smith"]
    assert [e.entity_id for e in resolved] == ["alice-id", "alice-smith-id"]
