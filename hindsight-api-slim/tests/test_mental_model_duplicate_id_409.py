"""A duplicate mental-model id is a conflict, not an unhandled 500.

Clients that sweep a list of external ids and create a model for each one re-POST
ids they already created. Before this was handled, every such call propagated the
raw ``mental_models_pkey`` violation out of the engine and surfaced as a 500.
"""

import asyncio
import uuid

import asyncpg
import pytest
import pytest_asyncio

from hindsight_api.engine.cross_encoder import RRFPassthroughCrossEncoder
from hindsight_api.engine.embeddings import Embeddings
from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.query_analyzer import DateparserQueryAnalyzer
from hindsight_api.engine.task_backend import SyncTaskBackend
from hindsight_api.extensions import OperationValidationError


class _OtherMentalModelUniqueViolation(asyncpg.UniqueViolationError):
    constraint_name = "mental_models_future_unique"


class _StubEmbeddings(Embeddings):
    @property
    def provider_name(self) -> str:
        return "stub"

    @property
    def dimension(self) -> int:
        return 384

    async def initialize(self) -> None:
        return None

    async def encode(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * self.dimension for _ in texts]


@pytest_asyncio.fixture
async def conflict_memory(pg0_db_url):
    engine = MemoryEngine(
        db_url=pg0_db_url,
        memory_llm_provider="none",
        memory_llm_model="none",
        embeddings=_StubEmbeddings(),
        cross_encoder=RRFPassthroughCrossEncoder(),
        query_analyzer=DateparserQueryAnalyzer(),
        pool_min_size=1,
        pool_max_size=4,
        run_migrations=False,
        task_backend=SyncTaskBackend(),
    )
    await engine.initialize()
    yield engine
    await engine.close()


async def test_duplicate_mental_model_id_raises_conflict(memory: MemoryEngine, request_context):
    bank_id = f"test-mm-dup-{uuid.uuid4().hex[:8]}"
    mental_model_id = "account-0013a00001aATOhAAO"
    create = dict(
        bank_id=bank_id,
        name="Account standing",
        source_query="what is the standing of this account",
        content="",
        mental_model_id=mental_model_id,
        request_context=request_context,
    )
    # Outside the try: the bank only exists once this has returned, and cleaning up
    # a bank that was never created would mask whatever made the first create fail.
    first = await memory.create_mental_model(**create)
    assert first["id"] == mental_model_id
    try:
        with pytest.raises(OperationValidationError) as excinfo:
            await memory.create_mental_model(**create)
        assert excinfo.value.status_code == 409
        assert "already exists" in excinfo.value.reason
    finally:
        await memory.delete_bank(bank_id, request_context=request_context)


async def test_concurrent_duplicate_mental_model_id_creates_once(conflict_memory: MemoryEngine, request_context):
    bank_id = f"test-mm-race-{uuid.uuid4().hex[:8]}"
    mental_model_id = "travel"
    create = dict(
        bank_id=bank_id,
        name="Travel",
        source_query="where has this user travelled",
        content="",
        mental_model_id=mental_model_id,
        request_context=request_context,
    )

    async def attempt_create():
        try:
            return await conflict_memory.create_mental_model(**create)
        except OperationValidationError as exc:
            return exc

    try:
        outcomes = await asyncio.gather(attempt_create(), attempt_create())

        created = [outcome for outcome in outcomes if isinstance(outcome, dict)]
        conflicts = [outcome for outcome in outcomes if isinstance(outcome, OperationValidationError)]
        assert [model["id"] for model in created] == [mental_model_id]
        assert len(conflicts) == 1
        assert conflicts[0].status_code == 409

        page = await conflict_memory.list_mental_models(bank_id=bank_id, request_context=request_context)
        assert [model["id"] for model in page.items] == [mental_model_id]
    finally:
        await conflict_memory.delete_bank(bank_id, request_context=request_context)


async def test_knowledge_page_duplicate_mental_model_id_raises_conflict(conflict_memory, request_context):
    bank_id = f"test-page-mm-dup-{uuid.uuid4().hex[:8]}"
    existing = await conflict_memory.create_mental_model(
        bank_id,
        "Existing MM",
        "what exists",
        "seed",
        request_context=request_context,
    )

    try:
        with pytest.raises(OperationValidationError) as excinfo:
            await conflict_memory.create_knowledge_page(
                bank_id,
                "New page",
                "what is new",
                "seed",
                mental_model_id=existing["id"],
                request_context=request_context,
            )

        assert excinfo.value.status_code == 409
        assert "already exists" in excinfo.value.reason
        nodes = await conflict_memory.list_knowledge_nodes(bank_id, request_context=request_context)
        assert nodes == []
    finally:
        await conflict_memory.delete_bank(bank_id, request_context=request_context)


async def test_non_primary_mental_model_unique_violation_is_not_mislabeled(
    conflict_memory, request_context, monkeypatch
):
    async def fail_insert(*args, **kwargs):
        raise _OtherMentalModelUniqueViolation("different unique constraint")

    monkeypatch.setattr(conflict_memory, "_insert_pinned_mental_model", fail_insert)

    with pytest.raises(_OtherMentalModelUniqueViolation):
        await conflict_memory.create_mental_model(
            f"test-mm-other-unique-{uuid.uuid4().hex[:8]}",
            "Other constraint",
            "query",
            "content",
            request_context=request_context,
        )
