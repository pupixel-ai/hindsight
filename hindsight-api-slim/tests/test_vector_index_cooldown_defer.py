"""验证索引缺口在冷却内持久延期，并在没有下一次写入时最终覆盖。"""

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from hindsight_api import RequestContext
from hindsight_api.engine import memory_engine as engine_module
from hindsight_api.engine import vector_index_health
from hindsight_api.engine.db.postgresql import PostgreSQLBackend
from hindsight_api.engine.memory_engine import MemoryEngine
from hindsight_api.engine.retain import bank_utils
from hindsight_api.engine.task_backend import SyncTaskBackend, WorkerTaskBackend
from hindsight_api.engine.vector_index_health import CoverageTrigger, plan_bank_vector_indexes
from hindsight_api.worker.exceptions import DeferOperation
from hindsight_api.worker.poller import WorkerPoller


@dataclass
class _Case:
    """隔离 schema 内的真实 PostgreSQL 队列与索引测试上下文，不加载模型。"""

    engine: MemoryEngine
    backend: PostgreSQLBackend
    schema: str


@pytest_asyncio.fixture
async def index_case(pg0_db_url: str, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[_Case]:
    """仅构建索引维护依赖的表，避免为数据库回归下载嵌入或重排序模型。"""

    backend = PostgreSQLBackend()
    await backend.initialize(pg0_db_url, min_size=1, max_size=4)
    schema = "idx_cooldown_" + uuid.uuid4().hex
    async with backend.acquire() as conn:
        # 索引目录与 worker 队列状态没有公开业务 API；只在随机 schema 构造最小持久化 seam。
        await conn.execute(f"""
            CREATE SCHEMA {schema};
            CREATE TABLE {schema}.banks (bank_id text PRIMARY KEY, internal_id uuid NOT NULL DEFAULT gen_random_uuid());
            CREATE TABLE {schema}.memory_units (
                id uuid PRIMARY KEY DEFAULT gen_random_uuid(), bank_id text NOT NULL, fact_type text NOT NULL,
                embedding vector(3));
            CREATE TABLE {schema}.async_operations (
                operation_id uuid PRIMARY KEY, bank_id text NOT NULL, operation_type text NOT NULL,
                status text NOT NULL, task_payload jsonb, result_metadata jsonb, serialization_key text,
                created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz DEFAULT now(),
                next_retry_at timestamptz, retry_count integer NOT NULL DEFAULT 0, error_message text,
                worker_id text, claimed_at timestamptz);
            INSERT INTO {schema}.banks (bank_id) VALUES ('bank');
            INSERT INTO {schema}.memory_units (bank_id, fact_type, embedding) VALUES ('bank', 'world', '[1,0,0]');
        """)
    monkeypatch.setattr(engine_module, "get_current_schema", lambda: schema)
    for module in (engine_module, vector_index_health, bank_utils):
        monkeypatch.setattr(module, "per_bank_indexes_are_eager", lambda: False)
    monkeypatch.setattr(vector_index_health, "per_bank_index_build_bound", lambda: 1)
    monkeypatch.setattr(vector_index_health, "per_bank_index_keep_bound", lambda: 1)
    monkeypatch.setattr(engine_module, "per_bank_index_min_submit_interval_seconds", lambda: 900)
    engine = object.__new__(MemoryEngine)
    engine._get_backend = AsyncMock(return_value=backend)
    engine._authenticate_tenant = AsyncMock()
    engine._operation_validator = None
    engine._task_backend = WorkerTaskBackend()
    try:
        yield _Case(engine, backend, schema)
    finally:
        async with backend.acquire() as conn:
            await conn.execute(f"DROP SCHEMA {schema} CASCADE")
        await backend.shutdown()


async def _payload(case: _Case, operation_id: str) -> dict:
    """读取真实持久任务载荷，确认延期无需进程内状态才能恢复。"""

    async with case.backend.acquire() as conn:
        value = await conn.fetchval(
            f"SELECT task_payload FROM {case.schema}.async_operations WHERE operation_id = $1", uuid.UUID(operation_id)
        )
    return json.loads(value) if isinstance(value, str) else value


@pytest.mark.asyncio
async def test_cooldown_job_is_deduped_deferred_and_eventually_covers_without_new_submit(
    index_case: _Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    """首个分区已建后，新类型在冷却内只排一个任务，到期补齐期间的所有缺口。"""

    case = index_case
    context = RequestContext(internal=True)
    first = await case.engine.submit_async_vector_index_maintenance(
        "bank", request_context=context, trigger=CoverageTrigger.GREW
    )
    await case.engine._handle_vector_index_maintenance(await _payload(case, first["operation_id"]))
    async with case.backend.acquire() as conn:
        await conn.execute(
            f"UPDATE {case.schema}.async_operations SET status = 'completed' WHERE operation_id = $1",
            uuid.UUID(first["operation_id"]),
        )
        await conn.execute(
            f"INSERT INTO {case.schema}.memory_units (bank_id, fact_type, embedding) VALUES ('bank','experience','[0,1,0]')"
        )
    submits = await asyncio.gather(
        *[
            case.engine.submit_async_vector_index_maintenance(
                "bank", request_context=context, trigger=CoverageTrigger.GREW
            )
            for _ in range(3)
        ]
    )
    assert all(result["operation_id"] is not None for result in submits)
    ids = {result["operation_id"] for result in submits}
    assert len(ids) == 1
    operation_id = ids.pop()
    payload = await _payload(case, operation_id)
    deadline = datetime.fromisoformat(payload["not_before"])
    with pytest.raises(DeferOperation) as deferred:
        await case.engine._handle_vector_index_maintenance(payload)
    assert deferred.value.exec_date == deadline
    poller = WorkerPoller(case.backend, "local-index-worker", case.engine._handle_vector_index_maintenance)
    await poller._defer_operation(operation_id, deadline, deferred.value.reason, case.schema)
    async with case.backend.acquire() as conn:
        row = await conn.fetchrow(
            f"SELECT status,next_retry_at,retry_count,error_message FROM {case.schema}.async_operations WHERE operation_id=$1",
            uuid.UUID(operation_id),
        )
        assert row["status"] == "pending" and row["next_retry_at"] == deadline
        assert row["retry_count"] == 0 and row["error_message"] is None
        await conn.execute(
            f"INSERT INTO {case.schema}.memory_units (bank_id,fact_type,embedding) VALUES ('bank','observation','[0,0,1]')"
        )

    class _DueClock(datetime):
        """直接推进到冷却到期，测试不等待真实的十五分钟。"""

        @classmethod
        def now(cls, tz=None) -> datetime:
            """返回已到期时间，同时保留调用方要求的时区。"""

            return (deadline + timedelta(seconds=1)).astimezone(tz or UTC)

    monkeypatch.setattr(engine_module, "datetime", _DueClock)
    await case.engine._handle_vector_index_maintenance(payload)
    async with case.backend.acquire() as conn:
        plan = await plan_bank_vector_indexes(conn, case.schema, "bank")
        assert plan.is_empty and plan.already_present == 3
        assert await conn.fetchval(f"SELECT count(*) FROM {case.schema}.async_operations") == 2


@pytest.mark.asyncio
async def test_handoff_and_zero_cooldown_do_not_delay(index_case: _Case, monkeypatch: pytest.MonkeyPatch) -> None:
    """任务自己的交接及显式零冷却保持立即执行，避免到期任务被重新延迟。"""

    case = index_case
    async with case.backend.acquire() as conn:
        previous = await conn.fetchval(f"""
            INSERT INTO {case.schema}.async_operations (operation_id,bank_id,operation_type,status)
            VALUES (gen_random_uuid(),'bank','vector_index_maintenance','completed') RETURNING operation_id
        """)
    for interval, exclude in ((900, str(previous)), (0, None)):
        monkeypatch.setattr(engine_module, "per_bank_index_min_submit_interval_seconds", lambda: interval)
        result = await case.engine.submit_async_vector_index_maintenance(
            "bank",
            request_context=RequestContext(internal=True),
            trigger=CoverageTrigger.FULL,
            dedupe_excludes_operation_id=exclude,
        )
        assert "not_before" not in await _payload(case, result["operation_id"])
        async with case.backend.acquire() as conn:
            await conn.execute(
                f"UPDATE {case.schema}.async_operations SET status='completed' WHERE operation_id=$1",
                uuid.UUID(result["operation_id"]),
            )


@pytest.mark.asyncio
async def test_sync_backend_has_no_worker_deferral(index_case: _Case) -> None:
    """同步嵌入后端直接维护，不把后台延期异常抛给同步写入调用方。"""

    case = index_case
    case.engine._task_backend = SyncTaskBackend()
    case.engine._task_backend.set_executor(case.engine._handle_vector_index_maintenance)
    async with case.backend.acquire() as conn:
        await conn.execute(f"""
            INSERT INTO {case.schema}.async_operations (operation_id,bank_id,operation_type,status)
            VALUES (gen_random_uuid(),'bank','vector_index_maintenance','completed')
        """)
    result = await case.engine.submit_async_vector_index_maintenance(
        "bank", request_context=RequestContext(internal=True), trigger=CoverageTrigger.GREW
    )
    assert "not_before" not in await _payload(case, result["operation_id"])
    async with case.backend.acquire() as conn:
        plan = await plan_bank_vector_indexes(conn, case.schema, "bank")
    assert plan.is_empty and plan.already_present == 1
