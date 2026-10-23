import pytest

from biodiv.fixtures import make_inat_fixture


@pytest.fixture
def inat_fixture(tmp_path):
    return make_inat_fixture(tmp_path / "inat")
