from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def docs_dir() -> Path:
    return FIXTURES / "docs"
