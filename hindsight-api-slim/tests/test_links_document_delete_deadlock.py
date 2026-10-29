"""用真实 PostgreSQL 事务保护重连与文档删除的关系／父行锁顺序。"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

import asyncpg
import pytest
import pytest_asyncio

from hindsight_api.engine.db.ops_postgresql import PostgreSQLOps
from hindsight_api.engine.db.postgresql import PostgresConnection
from hindsight_api.engine.memories.pg.links import _bulk_insert_links
from hindsight_api.engine.memory_engine import _current_schema

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

pytestmark = pytest.mark.memory_backend_incompatible

_LOW = uuid.UUID("00000000-0000-4000-8000-000000000001")
_MID = uuid.UUID("00000000-0000-4000-8000-000000000002")
_HIGH = uuid.UUID("00000000-0000-4000-8000-000000000003")
_NIL = "00000000-0000-0000-0000-000000000000"


@dataclass
class _Race:
    """隔离 schema 内两个写事务与只读锁观察连接。"""

    setup: asyncpg.Connection
    delete: asyncpg.Connection
    relink: asyncpg.Connection
    schema: str


@pytest_asyncio.fixture
async def link_race(pg0_db_url: str) -> AsyncIterator[_Race]:
    """只构造锁复现需要的表和真实延迟FK，不加载模型或调用LLM。"""
    schema = f"link_race_{uuid.uuid4().hex[:12]}"
    setup = await asyncpg.connect(pg0_db_url)
    delete = await asyncpg.connect(pg0_db_url)
    relink = await asyncpg.connect(pg0_db_url)
    token = _current_schema.set(schema)
    try:
        await setup.execute(f"CREATE SCHEMA {schema}")
        for conn in (setup, delete, relink):
            await conn.execute(f"SET search_path TO {schema}")
            await conn.execute("SET statement_timeout = '5s'")
        # 直接建表是为了精确控制级联FK和唯一键等待，公共API无法制造这个调度点。
        await setup.execute(f"""
            CREATE TABLE documents(id text, bank_id text, PRIMARY KEY(id, bank_id));
            CREATE TABLE memory_units(id uuid PRIMARY KEY, document_id text, bank_id text,
                FOREIGN KEY(document_id, bank_id) REFERENCES documents(id, bank_id) ON DELETE CASCADE);
            CREATE TABLE memory_links(
                from_unit_id uuid REFERENCES memory_units(id) ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED,
                to_unit_id uuid REFERENCES memory_units(id) ON DELETE CASCADE DEFERRABLE INITIALLY DEFERRED,
                link_type text, weight double precision, entity_id uuid, bank_id text);
            CREATE UNIQUE INDEX links_unique ON memory_links(
                from_unit_id, to_unit_id, link_type, COALESCE(entity_id, '{_NIL}'::uuid));
            CREATE TABLE graph_maintenance_queue(bank_id text, unit_id uuid,
                enqueued_at timestamptz DEFAULT now(), PRIMARY KEY(bank_id, unit_id));
            INSERT INTO documents VALUES('remove', 'bank'), ('keep', 'bank');
        """)
        await setup.execute(
            "INSERT INTO memory_units VALUES($1, 'remove', 'bank'), ($2, 'keep', 'bank'), ($3, 'keep', 'bank')",
            _LOW,
            _MID,
            _HIGH,
        )
        await setup.execute("INSERT INTO memory_links VALUES($1, $2, 'temporal', 1, NULL, 'bank')", _LOW, _HIGH)
        yield _Race(setup, delete, relink, schema)
    finally:
        await delete.close()
        await relink.close()
        await setup.execute(f"DROP SCHEMA {schema} CASCADE")
        await setup.close()
        _current_schema.reset(token)


async def _wait_blocked(monitor: asyncpg.Connection, pid: int, finished: asyncio.Event | None = None) -> None:
    """等待数据库真实阻塞或竞争方完成，避免靠固定sleep碰运气。"""
    for _ in range(300):
        if finished is not None and finished.is_set():
            return
        if await monitor.fetchval("SELECT cardinality(pg_blocking_pids($1)) > 0", pid):
            return
        await asyncio.sleep(0.01)
    pytest.fail("写事务未到达预期的数据库锁等待")


@pytest.mark.parametrize("chunk_size", [1, 5000])
async def test_relink_skips_existing_links_before_locking_parents(link_race: _Race, chunk_size: int) -> None:
    """删除已持旧关系时，重连跳过可见旧边，不能持父行锁等待其唯一键。"""
    race = link_race
    deleted_links = asyncio.Event()
    relink_finished = asyncio.Event()
    ops = PostgreSQLOps()

    async def remove_document() -> None:
        """固定线上先删关系、再删文档触发父事实级联的顺序。"""
        async with race.delete.transaction():
            await ops.delete_unit_links(PostgresConnection(race.delete), f"{race.schema}.memory_links", "bank", [_LOW])
            deleted_links.set()
            await _wait_blocked(race.setup, race.relink.get_server_pid(), relink_finished)
            await race.delete.execute("DELETE FROM documents WHERE id = 'remove' AND bank_id = 'bank'")

    async def relink() -> None:
        """第一个新关系与第二个旧关系分chunk，覆盖跨chunk保留父行锁的风险。"""
        await deleted_links.wait()
        async with race.relink.transaction():
            await _bulk_insert_links(
                PostgresConnection(race.relink),
                [(str(_LOW), str(_MID), "temporal", 1.0, None), (str(_LOW), str(_HIGH), "temporal", 1.0, None)],
                bank_id="bank",
                chunk_size=chunk_size,
                ops=ops,
            )
        relink_finished.set()

    results = await asyncio.gather(remove_document(), relink(), return_exceptions=True)
    assert not [r for r in results if isinstance(r, BaseException)], results
    # 公共图读会去重方向，不能证明级联后没有重新插入指向已删除端点的原始关系。
    assert await race.setup.fetchval("SELECT count(*) FROM memory_links") == 0
    assert await race.setup.fetchval("SELECT count(*) FROM documents") == 1


async def test_relink_keeps_parent_key_share_compatible_with_queue_enqueue(link_race: _Race) -> None:
    """保留删除侧NO KEY UPDATE与重连KEY SHARE的兼容性，避免queue/parent新死锁。"""
    race = link_race
    queue_claimed = asyncio.Event()
    ops = PostgreSQLOps()
    await race.setup.execute("INSERT INTO graph_maintenance_queue(bank_id, unit_id) VALUES('bank', $1)", _MID)

    async def relink() -> None:
        """先认领真实queue行，再等待删除持父行后重连。"""
        async with race.relink.transaction():
            await ops.claim_graph_maintenance_batch(
                PostgresConnection(race.relink),
                f"{race.schema}.graph_maintenance_queue",
                "bank",
                1,
            )
            queue_claimed.set()
            await _wait_blocked(race.setup, race.delete.get_server_pid())
            await _bulk_insert_links(
                PostgresConnection(race.relink),
                [(str(_LOW), str(_MID), "temporal", 1.0, None)],
                bank_id="bank",
                ops=ops,
            )

    async def remove_document() -> None:
        """删除持普通写锁后入队，同线上sweep→enqueue→cascade一致。"""
        await queue_claimed.wait()
        async with race.delete.transaction():
            await race.delete.fetch("SELECT id FROM memory_units WHERE id=$1 FOR NO KEY UPDATE", _LOW)
            await ops.enqueue_graph_maintenance(
                PostgresConnection(race.delete),
                f"{race.schema}.graph_maintenance_queue",
                "bank",
                [str(_MID)],
            )
            await ops.delete_unit_links(PostgresConnection(race.delete), f"{race.schema}.memory_links", "bank", [_LOW])
            await race.delete.execute("DELETE FROM documents WHERE id='remove' AND bank_id='bank'")

    results = await asyncio.gather(relink(), remove_document(), return_exceptions=True)
    assert not [r for r in results if isinstance(r, BaseException)], results
    # queue行在删除提交后仍存在，证明重连没有吞掉删除期间重新入队的修复信号。
    assert await race.setup.fetchval("SELECT count(*) FROM graph_maintenance_queue") == 1
    assert await race.setup.fetchval("SELECT count(*) FROM memory_links") == 0


async def test_existing_candidate_filter_preserves_full_link_identity(link_race: _Race) -> None:
    """旧边只排除完全相同的唯一键，不能吞掉同端点的其他类型或实体关系。"""
    race = link_race
    entity_id = uuid.UUID("00000000-0000-4000-8000-000000000004")
    async with race.relink.transaction():
        await _bulk_insert_links(
            PostgresConnection(race.relink),
            [
                (str(_LOW), str(_HIGH), "temporal", 1.0, None),
                (str(_LOW), str(_HIGH), "semantic", 0.9, None),
                (str(_LOW), str(_HIGH), "temporal", 0.8, str(entity_id)),
            ],
            bank_id="bank",
            ops=PostgreSQLOps(),
        )
    # 原始唯一键含entity_id，公共图会合并同端点关系，必须直接检验持久化键。
    rows = await race.setup.fetch(
        "SELECT link_type, entity_id FROM memory_links WHERE bank_id='bank' ORDER BY link_type, entity_id NULLS FIRST"
    )
    assert [(r["link_type"], r["entity_id"]) for r in rows] == [
        ("semantic", None),
        ("temporal", None),
        ("temporal", entity_id),
    ]


async def test_default_chunks_preserve_input_order_after_missing_candidates_reorder(link_race: _Race) -> None:
    """物化候选逆序后，两个默认5000分块写入仍必须按同一唯一键顺序推进。"""
    race = link_race
    ids = [uuid.UUID(int=10000 + i) for i in range(5010)]
    await race.setup.executemany("INSERT INTO memory_units VALUES($1, 'keep', 'bank')", [(unit_id,) for unit_id in ids])
    large_links = [(str(ids[i]), str(ids[i + 5]), "semantic", 0.9, None) for i in range(5000)]
    extras = [(str(ids[i]), str(ids[j]), "semantic", 0.9, None) for i in range(100) for j in range(i + 6, i + 56)]
    small_links = [large_links[0], *extras[1:], large_links[-1]]
    first_chunk_done = asyncio.Event()
    ops = PostgreSQLOps()

    class ReorderedConnection(PostgresConnection):
        """真实SQL反转物化候选，稳定模拟Hash分批等执行计划对输入顺序的破坏。"""

        async def execute(self, query: str, *args, **kwargs) -> str:
            """只改变中间CTE的输出顺序，保留生产最终INSERT及其排序规则。"""
            boundary = "), locked AS ("
            assert query.count(boundary) == 1
            reordered = query.replace(boundary, "ORDER BY u.f DESC, u.t DESC\n                " + boundary)
            return await super().execute(reordered, *args, **kwargs)

    class PausingConnection(ReorderedConnection):
        """首个5000块完成后，等竞争批持锁排队再推进第二块，固定等待环调度。"""

        calls = 0

        async def execute(self, query: str, *args, **kwargs) -> str:
            """数据库屏障确保修复前一定出现反向唯一键等待，修复后两边均可完成。"""
            result = await super().execute(query, *args, **kwargs)
            self.calls += 1
            if self.calls == 1:
                first_chunk_done.set()
                await _wait_blocked(race.setup, race.relink.get_server_pid())
            return result

    async def small_batch() -> None:
        """5001候选的首块持最小共享键，第二块才插入最大的共享键。"""
        async with race.delete.transaction():
            await _bulk_insert_links(PausingConnection(race.delete), small_links, bank_id="bank", ops=ops)

    async def large_batch() -> None:
        """5000候选必须先等最小共享键，不能因中间重排先持最大的共享键。"""
        await first_chunk_done.wait()
        async with race.relink.transaction():
            await _bulk_insert_links(ReorderedConnection(race.relink), large_links, bank_id="bank", ops=ops)

    results = await asyncio.gather(small_batch(), large_batch(), return_exceptions=True)
    assert not [r for r in results if isinstance(r, BaseException)], results
    # 两批只有首尾两个重复键；直接计数可证明等待解除后没有丢失任何非重复候选。
    assert await race.setup.fetchval("SELECT count(*) FROM memory_links WHERE link_type='semantic'") == 9999
