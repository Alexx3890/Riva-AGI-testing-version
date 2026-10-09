import pytest
import rag_knowledge.storage.qdrant_storage as qdrant_mod
import rag_knowledge.service as service_mod


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def reset_globals():
    qdrant_mod._global_qdrant_store = None
    service_mod._default_service = None
    yield
    qdrant_mod._global_qdrant_store = None
    service_mod._default_service = None
