import pytest

from tests.fixtures.make_fixture import make_inat_fixture


@pytest.fixture
def inat_fixture(tmp_path):
    return make_inat_fixture(tmp_path / "inat")
