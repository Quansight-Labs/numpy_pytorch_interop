import sys

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: very slow tests")


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", help="run slow tests")
    parser.addoption("--nonp", action="store_true", help="error when NumPy is accessed")


def pytest_sessionstart(session):
    if session.config.getoption("--nonp"):

        class Inaccessible:
            def __getattribute__(self, attr):
                raise RuntimeError(f"Using --nonp but accessed np.{attr}")

        sys.modules["numpy"] = Inaccessible()


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="slow test, use --runslow to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
