"""Regression coverage for exporting existing underscore mental model IDs."""

import pytest
from pydantic import ValidationError

from hindsight_api.api.http import BankTemplateManifest, BankTemplateMentalModel


@pytest.mark.parametrize("model_id", ["chat_memory", "photo_memory-v2", "chat-memory", "0"])
def test_template_round_trip_preserves_model_id(model_id: str) -> None:
    model = BankTemplateMentalModel(id=model_id, name="Memory", source_query="Summarize memories")
    exported = BankTemplateManifest(version="1", mental_models=[model]).model_dump_json()
    imported = BankTemplateManifest.model_validate_json(exported)
    assert imported.mental_models[0].id == model_id


@pytest.mark.parametrize("model_id", ["", "_chat", "-chat", "Chat", "chat memory", "a/b", "a\n"])
def test_template_rejects_invalid_model_id(model_id: str) -> None:
    with pytest.raises(ValidationError):
        BankTemplateMentalModel(id=model_id, name="Memory", source_query="Summarize memories")


def test_export_existing_underscore_id_can_be_imported() -> None:
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from fastapi.testclient import TestClient

    from hindsight_api.api.http import create_app

    memory = SimpleNamespace(
        audit_logger=None,
        get_bank_profile=AsyncMock(return_value={"name": "Existing bank"}),
        _authenticate_tenant=AsyncMock(),
        _config_resolver=SimpleNamespace(_load_bank_config=AsyncMock(return_value={})),
        list_mental_models=AsyncMock(
            return_value=[{"id": "chat_memory", "name": "Chat memory", "source_query": "Summarize conversations"}]
        ),
        list_directives=AsyncMock(return_value=[]),
    )
    client = TestClient(create_app(memory, initialize_memory=False))
    exported = client.get("/v1/default/banks/source/export")
    assert exported.status_code == 200, exported.text
    assert exported.json()["mental_models"][0]["id"] == "chat_memory"
    imported = client.post("/v1/default/banks/target/import?dry_run=true", json=exported.json())
    assert imported.status_code == 200, imported.text
    assert imported.json()["mental_models_created"] == ["chat_memory"]
