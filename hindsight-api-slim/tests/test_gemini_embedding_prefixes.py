"""验证原生入库和 Recall 最终发给 Google SDK 的文字及参数。"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.genai import types

from hindsight_api.engine.embeddings import GeminiEmbeddings
from hindsight_api.engine.retain.embedding_utils import generate_embeddings_batch


@pytest.mark.parametrize("vertexai", [False, True])
@pytest.mark.parametrize("prefixed", [False, True])
async def test_retain_and_recall_sdk_inputs(vertexai: bool, prefixed: bool) -> None:
    query_prefix = "task: search result | query: " if prefixed else ""
    passage_prefix = "title: none | text: " if prefixed else ""
    embeddings = GeminiEmbeddings(
        model="gemini-embedding-2",
        api_key=None if vertexai else "test-key",
        vertexai_project_id="test-project" if vertexai else None,
        output_dimensionality=768,
        query_prefix=query_prefix,
        passage_prefix=passage_prefix,
    )
    client = MagicMock()
    embed_content = AsyncMock(
        return_value=types.EmbedContentResponse(embeddings=[types.ContentEmbedding(values=[0.1] * 768)])
    )
    client.aio.models.embed_content = embed_content
    with patch("google.genai.Client", return_value=client):
        await embeddings.initialize()
    embed_content.reset_mock()

    # 使用 Retain / Recall 共用的入口，确保 input_type 一直传到 SDK，且只加一次前缀。
    document = "P001 和 P009 在芭堤雅合影 (happened in May 2024) [P001, P009]"
    query = "我和 P009 去哪里旅游过？"
    documents = await generate_embeddings_batch(embeddings, [document])
    queries = await generate_embeddings_batch(embeddings, [query], input_type="query")

    assert len(documents) == len(queries) == 1
    assert len(documents[0]) == len(queries[0]) == 768
    calls = embed_content.call_args_list
    assert len(calls) == 2
    assert calls[0].kwargs["contents"][0].parts[0].text == passage_prefix + document
    assert calls[1].kwargs["contents"][0].parts[0].text == query_prefix + query
    for call in calls:
        assert call.kwargs["model"] == "gemini-embedding-2"
        assert call.kwargs["config"].model_dump(exclude_none=True) == {"output_dimensionality": 768}
