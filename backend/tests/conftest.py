from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from inference.config import Settings
from inference.entrypoints.app import create_app
from tests.support import FakeInferenceEngine, FakeMediaProcessor


@pytest.fixture
def fake_engine() -> FakeInferenceEngine:
    return FakeInferenceEngine()


@pytest.fixture
def fake_media_processor() -> FakeMediaProcessor:
    return FakeMediaProcessor()


@pytest.fixture
def settings() -> Settings:
    return Settings(max_upload_bytes=1024)


@pytest.fixture
def client(
    fake_engine: FakeInferenceEngine,
    fake_media_processor: FakeMediaProcessor,
    settings: Settings,
) -> Iterator[TestClient]:
    app = create_app(
        engine=fake_engine,
        media_processor=fake_media_processor,
        settings=settings,
    )
    with TestClient(app) as test_client:
        yield test_client
